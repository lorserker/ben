import os
import asyncio
import hashlib
import logging

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# This import is only to help PyInstaller when generating the executables
import tensorflow as tf

import deck52
import pprint
import time
import json
import asyncio
import uuid
import shelve
import os 
import argparse
import numpy as np

import human
import bots
import conf

from sample import Sample
from bidding import bidding
from sample import Sample
from deck52 import decode_card
from bidding.binary import DealData
from objects import CardResp, Card, BidResp
from claim import Claimer
from pbn2ben import load

import os

def get_execution_path():
    # Get the directory where the program is started from either PyInstaller executable or the script
    return os.getcwd()


def random_deal():
    deal_str = deck52.random_deal()
    auction_str = deck52.random_dealer_vuln()

    return deal_str, auction_str


class AsyncBotBid(bots.BotBid):
    async def async_bid(self, auction):
        return self.bid(auction)

class AsyncBotLead(bots.BotLead):
    async def async_lead(self, auction):
        return self.find_opening_lead(auction)

class AsyncCardPlayer(bots.CardPlayer):
    async def async_play_card(self, trick_i, leader_i, current_trick52, players_states):
        return self.play_card(trick_i, leader_i, current_trick52, players_states)
    
    
class Driver:

    def __init__(self, models, factory, sampler, verbose):
        self.models = models
        self.sampler = sampler
        self.factory = factory
        self.confirmer = factory.create_confirmer()
        self.channel = factory.create_channel()

        print("Setting seed=42")
        np.random.seed(42)

        #Default is a Human South
        self.human = [0.1, 0.1, 1, 0.1]
        #Default is no system
        self.ns = -1
        self.ew = -1
        self.sampler = sampler
        self.verbose = verbose
        self.play_only = False

    def set_deal(self, board_number, deal_str, auction_str, ns, ew, play_only = None):
        self.board_number = board_number
        self.deal_str = deal_str
        self.hands = deal_str.split()
        self.deal_data = DealData.from_deal_auction_string(self.deal_str, auction_str, ns, ew, 32)
        self.deal_data_52 = DealData.from_deal_auction_string(self.deal_str, auction_str, ns, ew, 52)

        auction_part = auction_str.split(' ')
        if play_only == None and len(auction_part) > 2: play_only = True
        if play_only:
            self.auction = self.deal_data.auction
            self.play_only = play_only
        self.dealer_i = self.deal_data.dealer
        self.vuln_ns = self.deal_data.vuln_ns
        self.vuln_ew = self.deal_data.vuln_ew
        self.ns = ns
        self.ew = ew
        self.trick_winners = []

        # Calculate the SHA-256 hash
        hash_object = hashlib.sha256(deal_str.encode())
        hash_bytes = hash_object.digest()

        # Convert the first 4 bytes of the hash to an integer and take modulus
        hash_integer = int.from_bytes(hash_bytes[:4], byteorder='big') % (2**32 - 1)

        # Now you can use hash_integer as a seed
        print("Setting seed=",hash_integer)
        np.random.seed(hash_integer)


    async def run(self):
        await self.channel.send(json.dumps({
            'message': 'deal_start',
            'dealer': self.dealer_i,
            'vuln': [self.vuln_ns, self.vuln_ew],
            'hand': self.deal_str.split()[2]
        }))

        self.bid_responses = []
        self.card_responses = []

        if self.play_only:
            auction = self.auction
            for bid in auction:
                if bidding.BID2ID[bid] > 1:
                    self.bid_responses.append(BidResp(bid=bid, candidates=[], samples=[], shape=-1, hcp=-1, who="PlayOnly"))
        else:
            auction = await self.bidding()

        self.contract = bidding.get_contract(auction)

        #if self.verbose:
        #    print("****** Bid responses ******")
        #    for bid_resp in self.bid_responses:
        #        pprint.pprint(bid_resp.to_dict(), width=120)

        if self.contract is None:
            return

        strain_i = bidding.get_strain_i(self.contract)
        decl_i = bidding.get_decl_i(self.contract)

        await self.channel.send(json.dumps({
            'message': 'auction_end',
            'declarer': decl_i,
            'auction': auction,
            'strain': strain_i
        }))

        print("trick 1")

        opening_lead52 = (await self.opening_lead(auction)).card.code()

        await self.channel.send(json.dumps({
            'message': 'card_played',
            'player': (decl_i + 1) % 4,
            'card': decode_card(opening_lead52)
        }))

        await self.channel.send(json.dumps({
            'message': 'show_dummy',
            'player': (decl_i + 1) % 4,
            'dummy': self.deal_str.split()[0] if decl_i == 0 else self.deal_str.split()[(decl_i + 2) % 4]
        }))

        if self.verbose: 
            for card_resp in self.card_responses:
                pprint.pprint(card_resp.to_dict(), width=200)
        
        await self.play(auction, opening_lead52)

        await self.channel.send(json.dumps({
            'message': 'deal_end',
            'pbn': self.deal_str
        }))

    def to_dict(self):
        return {
            'timestamp': time.time(),
            'dealer': self.dealer_i,
            'vuln_ns': self.vuln_ns,
            'vuln_ew': self.vuln_ew,
            'hands': self.deal_str,
            'bids': [b.to_dict() for b in self.bid_responses],
            'contract': self.contract,
            'play': [c.to_dict() for c in self.card_responses],
            'trick_winners': self.trick_winners,
            'board_number' : self.board_number
        }

    async def play(self, auction, opening_lead52):
        contract = bidding.get_contract(auction)
        
        level = int(contract[0])
        strain_i = bidding.get_strain_i(contract)
        decl_i = bidding.get_decl_i(contract)
        is_decl_vuln = [self.vuln_ns, self.vuln_ew, self.vuln_ns, self.vuln_ew][decl_i]

        lefty_hand = self.hands[(decl_i + 1) % 4]
        dummy_hand = self.hands[(decl_i + 2) % 4]
        righty_hand = self.hands[(decl_i + 3) % 4]
        decl_hand = self.hands[decl_i]

        card_players = [
            AsyncCardPlayer(self.models.player_models, 0, lefty_hand, dummy_hand, contract, is_decl_vuln, self.verbose),
            AsyncCardPlayer(self.models.player_models, 1, dummy_hand, decl_hand, contract, is_decl_vuln, self.verbose),
            AsyncCardPlayer(self.models.player_models, 2, righty_hand, dummy_hand, contract, is_decl_vuln, self.verbose),
            AsyncCardPlayer(self.models.player_models, 3, decl_hand, dummy_hand, contract, is_decl_vuln, self.verbose)
        ]

        # check if user is playing and is declarer
        if self.human[2] == 1:
            if decl_i == 2:
                card_players[3] = self.factory.create_human_cardplayer(self.models.player_models, 3, decl_hand, dummy_hand, contract, is_decl_vuln)
                card_players[1] = self.factory.create_human_cardplayer(self.models.player_models, 1, dummy_hand, decl_hand, contract, is_decl_vuln)
            elif decl_i == 0:
                card_players[1] = self.factory.create_human_cardplayer(self.models.player_models, 1, dummy_hand, decl_hand, contract, is_decl_vuln)
                card_players[3] = self.factory.create_human_cardplayer(self.models.player_models, 3, decl_hand, dummy_hand, contract, is_decl_vuln)
            elif decl_i == 1:
                card_players[0] = self.factory.create_human_cardplayer(self.models.player_models, 0, lefty_hand, dummy_hand, contract, is_decl_vuln)
            elif decl_i == 3:
                card_players[2] = self.factory.create_human_cardplayer(self.models.player_models, 2, righty_hand, dummy_hand, contract, is_decl_vuln)

        claimer = Claimer(self.verbose)

        player_cards_played = [[] for _ in range(4)]
        shown_out_suits = [set() for _ in range(4)]

        leader_i = 0

        tricks = []
        tricks52 = []
        trick_won_by = []

        opening_lead = deck52.card52to32(opening_lead52)

        current_trick = [opening_lead]
        current_trick52 = [opening_lead52]

        card_players[0].hand52[opening_lead52] -= 1

        for trick_i in range(12):
            if trick_i != 0:
                print("trick {}".format(trick_i+1))

            for player_i in map(lambda x: x % 4, range(leader_i, leader_i + 4)):
                if self.verbose:
                    print('player {}'.format(player_i))
                
                if trick_i == 0 and player_i == 0:
                    # To get the state right we ask for the play when using Tf.2X
                    if self.verbose:
                        print('skipping opening lead for ',player_i)
                    for i, card_player in enumerate(card_players):
                        card_player.set_card_played(trick_i=trick_i, leader_i=leader_i, i=0, card=opening_lead)
                    continue

                # Don't calculate claim before trick 6    
                if trick_i > 5 and len(current_trick) == 0 and player_i in (1, 3):
                    claimer.claim(
                        strain_i=strain_i,
                        player_i=player_i,
                        hands52=[card_player.hand52 for card_player in card_players],
                        n_samples=20
                    )
                rollout_states = None
                c_hcp = -1
                c_shp = -1
                if isinstance(card_players[player_i], bots.CardPlayer):
                    rollout_states, c_hcp, c_shp = self.sampler.init_rollout_states(trick_i, player_i, card_players, player_cards_played, shown_out_suits, current_trick, auction, card_players[player_i].hand.reshape(
                        (-1, 32)), [self.vuln_ns, self.vuln_ew], self.models, self.ns, self.ew)

                await asyncio.sleep(0.01)

                card_resp = await card_players[player_i].async_play_card(trick_i, leader_i, current_trick52, rollout_states)
                card_resp.hcp = c_hcp
                card_resp.shape = c_shp
                if self.verbose:
                    pprint.pprint(card_resp.to_dict(), width=200)
                
                await asyncio.sleep(0.01)

                self.card_responses.append(card_resp)

                card52 = card_resp.card.code()

                await self.channel.send(json.dumps({
                    'message': 'card_played',
                    'player': ((decl_i + 1) % 4 + player_i) % 4,
                    'card': card_resp.card.symbol()
                }))

                card = deck52.card52to32(card52)

                for card_player in card_players:
                    card_player.set_card_played(trick_i=trick_i, leader_i=leader_i, i=player_i, card=card)

                current_trick.append(card)

                current_trick52.append(card52)

                card_players[player_i].set_own_card_played52(card52)
                if player_i == 1:
                    for i in [0, 2, 3]:
                        card_players[i].set_public_card_played52(card52)
                if player_i == 3:
                    card_players[1].set_public_card_played52(card52)

                # update shown out state
                if card // 8 != current_trick[0] // 8:  # card is different suit than lead card
                    shown_out_suits[player_i].add(current_trick[0] // 8)

            # sanity checks after trick completed
            assert len(current_trick) == 4

            for i, card_player in enumerate(card_players):
                assert np.min(card_player.hand52) == 0
                assert np.min(card_player.public52) == 0
                assert np.sum(card_player.hand52) == 13 - trick_i - 1
                assert np.sum(card_player.public52) == 13 - trick_i - 1

            tricks.append(current_trick)
            tricks52.append(current_trick52)

            # initializing for the next trick
            # initialize hands
            for i, card in enumerate(current_trick):
                card_players[(leader_i + i) % 4].x_play[:, trick_i + 1, 0:32] = card_players[(leader_i + i) % 4].x_play[:, trick_i, 0:32]
                card_players[(leader_i + i) % 4].x_play[:, trick_i + 1, 0 + card] -= 1

            # initialize public hands
            for i in (0, 2, 3):
                card_players[i].x_play[:, trick_i + 1, 32:64] = card_players[1].x_play[:, trick_i + 1, 0:32]
            card_players[1].x_play[:, trick_i + 1, 32:64] = card_players[3].x_play[:, trick_i + 1, 0:32]

            for card_player in card_players:
                # initialize last trick
                for i, card in enumerate(current_trick):
                    card_player.x_play[:, trick_i + 1, 64 + i * 32 + card] = 1
                    
                # initialize last trick leader
                card_player.x_play[:, trick_i + 1, 288 + leader_i] = 1

                # initialize level
                card_player.x_play[:, trick_i + 1, 292] = level

                # initialize strain
                card_player.x_play[:, trick_i + 1, 293 + strain_i] = 1

            # sanity checks for next trick
            for i, card_player in enumerate(card_players):
                assert np.min(card_player.x_play[:, trick_i + 1, 0:32]) == 0
                assert np.min(card_player.x_play[:, trick_i + 1, 32:64]) == 0
                assert np.sum(card_player.x_play[:, trick_i + 1, 0:32], axis=1) == 13 - trick_i - 1
                assert np.sum(card_player.x_play[:, trick_i + 1, 32:64], axis=1) == 13 - trick_i - 1

            trick_winner = (leader_i + deck52.get_trick_winner_i(current_trick52, (strain_i - 1) % 5)) % 4
            trick_won_by.append(trick_winner)

            if trick_winner % 2 == 0:
                card_players[0].n_tricks_taken += 1
                card_players[2].n_tricks_taken += 1
            else:
                card_players[1].n_tricks_taken += 1
                card_players[3].n_tricks_taken += 1

            #print('trick52 {} cards={}. won by {}'.format(trick_i+1, list(map(decode_card, current_trick52)), trick_winner))
            if np.any(np.array(self.human) == 1):
                key = await self.confirmer.confirm()
                if key == 'q':
                    print(self.deal_str)
                    return

            # update cards shown
            for i, card in enumerate(current_trick):
                player_cards_played[(leader_i + i) % 4].append(card)
            
            leader_i = trick_winner
            current_trick = []
            current_trick52 = []

        # play last trick
        print("trick 13")
        for player_i in map(lambda x: x % 4, range(leader_i, leader_i + 4)):
            
            if not isinstance(card_players[player_i], bots.CardPlayer):
                await card_players[player_i].get_card_input()

            card52 = np.nonzero(card_players[player_i].hand52)[0][0]
            card =deck52.card52to32(card52)

            card_resp = CardResp(card=Card.from_code(card52), candidates=[], samples=[], shape=-1, hcp=-1)

            await self.channel.send(json.dumps({
                'message': 'card_played',
                'player': ((decl_i + 1) % 4 + player_i) % 4,
                'card': card_resp.card.symbol()
            }))

            self.card_responses.append(card_resp)
            
            current_trick.append(card)
            current_trick52.append(card52)

        if np.any(np.array(self.human) == 1):
            await self.confirmer.confirm()

        tricks.append(current_trick)
        tricks52.append(current_trick52)

        trick_winner = (leader_i + deck52.get_trick_winner_i(current_trick52, (strain_i - 1) % 5)) % 4
        trick_won_by.append(trick_winner)

        # Decode each element of tricks52
        decoded_tricks52 = [[deck52.decode_card(item) for item in inner] for inner in tricks52]
        pprint.pprint(list(zip(decoded_tricks52, trick_won_by)))

        self.trick_winners = trick_won_by

    
    async def opening_lead(self, auction):
        contract = bidding.get_contract(auction)
        decl_i = bidding.get_decl_i(contract)

        hands_str = self.deal_str.split()

        await asyncio.sleep(0.01)

        if self.human[(decl_i + 1) % 4] == 1:
            card_resp = await self.factory.create_human_leader().async_lead()
        else:
            bot_lead = AsyncBotLead(
                [self.vuln_ns, self.vuln_ew], 
                hands_str[(decl_i + 1) % 4], 
                self.models,
                self.ns,
                self.ew,
                self.models.lead_threshold,
                self.sampler,
                self.verbose
            )
            card_resp = await bot_lead.async_lead(auction)

        await asyncio.sleep(0.01)

        self.card_responses.append(card_resp)

        return card_resp

    async def bidding(self):
        hands_str = self.deal_str.split()
        
        vuln = [self.vuln_ns, self.vuln_ew]

        players = []
        for i, level in enumerate(self.human):
            if level == 99:
                from bba.BBA import BBABotBid
                players.append(BBABotBid(1,1,i,hands_str[i],vuln, self.dealer_i))
            elif level == 1:
                players.append(self.factory.create_human_bidder(vuln, hands_str[i]))
            else:
                # Overrule configuration for search threshold
                if level != -1:
                    self.models.search_threshold = level
                bot = AsyncBotBid(vuln, hands_str[i], self.models, self.ns, self.ew, self.sampler, self.verbose)
                players.append(bot)

        auction = ['PAD_START'] * self.dealer_i

        player_i = self.dealer_i

        while not bidding.auction_over(auction):
            bid_resp = await players[player_i].async_bid(auction)

            self.bid_responses.append(bid_resp)

            auction.append(bid_resp.bid)

            await self.channel.send(json.dumps({
                'message': 'bid_made',
                'auction': auction
            }))

            player_i = (player_i + 1) % 4

        return auction

def random_deal_source():
    while True:
        yield random_deal()

async def main():
    random = True
    #For some strange reason parameters parsed to the handler must be an array
    board_no = []
    board_no.append(0) 

    # Get the path to the config file
    config_path = get_execution_path()
    
    base_path = os.getenv('BEN_HOME') or config_path

    parser = argparse.ArgumentParser(description="Game server")
    parser.add_argument("--boards", default="", help="Filename for configuration")
    parser.add_argument("--auto", type=bool, default=False, help="Continue without user confirmation. If a file is provided it will stop at end of file")
    parser.add_argument("--boardno", default=0, type=int, help="Board number to start from")
    parser.add_argument("--config", default=f"{base_path}/config/default.conf", help="Filename for configuration")
    parser.add_argument("--ns", type=int, default=-1, help="System for NS")
    parser.add_argument("--ew", type=int, default=-1, help="System for EW")
    parser.add_argument("--playonly", type=bool, default=False, help="Just play, no bidding")
    parser.add_argument("--verbose", type=bool, default=False, help="Output samples and other information during play")

    args = parser.parse_args()

    configfile = args.config
    verbose = args.verbose
    auto = args.auto
    play_only = args.playonly
    boards = []

    if args.boards:
        filename = args.boards
        file_extension = os.path.splitext(filename)[1].lower()  
        if file_extension == '.ben':
            with open(filename, "r") as file:
                board_no.append(0) 
                lines = file.readlines()  # 
                # Loop through the lines, grouping them into objects
                for i in range(0, len(lines), 2):
                    board = {
                        'deal': lines[i].strip(),      
                        'auction': lines[i+1].strip().replace('NT','N')  
                    }
                    boards.append(board)            
            print(f"{len(boards)} boards loaded from file")
            random = False
        if file_extension == '.pbn':
            with open(filename, "r") as file:
                lines = file.readlines()
                boards = load(lines)
                print(f"{len(boards)} boards loaded from file")
            random = False

    if args.boardno:
        print(f"Starting from {args.boardno}")
        board_no[0] = args.boardno - 1

    if random:
        print("Playing random deals or deals from the client")
 
    ns = args.ns
    ew = args.ew

    np.set_printoptions(precision=1, suppress=True, linewidth=200)

    configuration = conf.load(configfile)

    try:
        if (configuration["models"]['tf_version'] == "2"):
            print("Loading version 2")
            from nn.models_tf2 import Models
        else: 
            # Default to version 1. of Tensorflow
            from nn.models import Models
    except KeyError:
            # Default to version 1. of Tensorflow
            from nn.models import Models

    models = Models.from_conf(configuration, base_path.replace(os.path.sep + "src",""))

    driver = Driver(models, human.ConsoleFactory(), Sample.from_conf(configuration, verbose), verbose)

    while True:
        if random: 
            #Just take a random"
            rdeal = random_deal()

            # example of to use a fixed deal
            # rdeal = ('AQ9.543.6.AKJ876 762.A96.KQJ42.Q2 KJ83.KJ2.T753.T5 T54.QT87.A98.943', 'S Both')

            driver.set_deal(None, *rdeal, ns, ew, False)
        else:
            rdeal = boards[board_no[0]]['deal']
            auction = boards[board_no[0]]['auction']
            print(f"Board: {board_no[0]+1} {rdeal}")
            driver.set_deal(board_no[0] + 1, rdeal, auction, ns, ew, play_only)
            board_no[0] = (board_no[0] + 1) % len(boards)

        # BEN is handling all 4 hands
        driver.human = [-1, -1, -1, -1]
        # BBA is handling all 4 hands
        # driver.human = [99, 99, 99, 99]
        await driver.run()

        with shelve.open(f"{base_path}/gamedb") as db:
            deal = driver.to_dict()
            db[uuid.uuid4().hex] = deal

        if not auto:
            user_input = input("\n Q to quit or any other key for next deal ")
            if user_input.lower() == "q":
                break
        else:
            if args.boards and board_no[0] >= len(boards):
                break

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
