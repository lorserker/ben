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
import sample
import conf

from bidding import bidding
from sample import Sample
from nn.models import Models
from deck52 import decode_card
from bidding.binary import DealData
from objects import CardResp, Card
from claim import Claimer
from pbn2ben import load



def random_deal():
    deal_str = deck52.random_deal()
    auction_str = deck52.random_dealer_vuln()

    return deal_str, auction_str


class AsyncBotBid(bots.BotBid):
    async def async_bid(self, auction):
        return self.bid(auction)

class AsyncBotLead(bots.BotLead):
    async def async_lead(self, auction):
        return self.lead(auction)

class AsyncCardPlayer(bots.CardPlayer):
    async def async_play_card(self, trick_i, leader_i, current_trick52, players_states):
        return self.play_card(trick_i, leader_i, current_trick52, players_states)
    
    
class Driver:

    def __init__(self, models, factory, sampler):
        self.models = models
        self.sampler = sampler
        self.factory = factory
        self.confirmer = factory.create_confirmer()
        self.channel = factory.create_channel()

        print("Setting seed=42")
        np.random.seed(42)

        #print('confirmer', self.confirmer)

        #Default is a Human South
        self.human = [0.1, 0.1, 1, 0.1]
        #Default is no system
        self.ns = -1
        self.ew = -1
        self.sampler = sampler

    def set_deal(self, deal_str, auction_str, ns, ew):
        self.deal_str = deal_str
        self.hands = deal_str.split()

        self.deal_data = DealData.from_deal_auction_string(self.deal_str, auction_str, ns, ew, 32)
        self.deal_data_52 = DealData.from_deal_auction_string(self.deal_str, auction_str, ns, ew, 52)

        self.dealer_i = self.deal_data.dealer
        self.vuln_ns = self.deal_data.vuln_ns
        self.vuln_ew = self.deal_data.vuln_ew
        self.ns = ns
        self.ew = ew
        self.trick_winners = []

    async def run(self):
        await self.channel.send(json.dumps({
            'message': 'deal_start',
            'dealer': self.dealer_i,
            'vuln': [self.vuln_ns, self.vuln_ew],
            'hand': self.deal_str.split()[2]
        }))

        self.bid_responses = []
        self.card_responses = []
        print(f"Deal={self.deal_str}")

        auction = await self.bidding()
        self.contract = bidding.get_contract(auction)

        for bid_resp in self.bid_responses:
            pprint.pprint(bid_resp.to_dict(), width=120)

        if self.contract is None:
            return

        #print(auction)
        #print(self.contract)

        #print([self.dealer_i, self.vuln_ns, self.vuln_ew])
        #print(self.deal_str)

        #level = int(self.contract[0])
        strain_i = bidding.get_strain_i(self.contract)
        decl_i = bidding.get_decl_i(self.contract)

        await self.channel.send(json.dumps({
            'message': 'auction_end',
            'declarer': decl_i,
            'auction': auction,
            'strain': strain_i
        }))

        opening_lead52 = (await self.opening_lead(auction)).card.code()

        await self.channel.send(json.dumps({
            'message': 'card_played',
            'player': (decl_i + 1) % 4,
            'card': decode_card(opening_lead52)
        }))

        await self.channel.send(json.dumps({
            'message': 'opening_lead',
            'player': (decl_i + 1) % 4,
            'card': decode_card(opening_lead52),
            'dummy': self.deal_str.split()[0] if decl_i == 0 else self.deal_str.split()[(decl_i + 2) % 4]
        }))

        print('opening lead:', decode_card(opening_lead52))

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
            'trick_winners': self.trick_winners
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
            AsyncCardPlayer(self.models.player_models, 0, lefty_hand, dummy_hand, contract, is_decl_vuln),
            AsyncCardPlayer(self.models.player_models, 1, dummy_hand, decl_hand, contract, is_decl_vuln),
            AsyncCardPlayer(self.models.player_models, 2, righty_hand, dummy_hand, contract, is_decl_vuln),
            AsyncCardPlayer(self.models.player_models, 3, decl_hand, dummy_hand, contract, is_decl_vuln)
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

        claimer = Claimer()

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
            print("trick {}".format(trick_i+1))

            for player_i in map(lambda x: x % 4, range(leader_i, leader_i + 4)):
                print('player {}'.format(player_i))
                
                if trick_i == 0 and player_i == 0:
                    #print('skipping')
                    for i, card_player in enumerate(card_players):
                        card_player.set_card_played(trick_i=trick_i, leader_i=leader_i, i=0, card=opening_lead)

                    continue

                if trick_i > 0 and len(current_trick) == 0 and player_i in (1, 3):
                    claimer.claim(
                        strain_i=strain_i,
                        player_i=player_i,
                        hands52=[card_player.hand52 for card_player in card_players],
                        n_samples=20
                    )

                rollout_states = None
                if isinstance(card_players[player_i], bots.CardPlayer):
                    rollout_states = self.sampler.init_rollout_states(trick_i, player_i, card_players, player_cards_played, shown_out_suits, current_trick, 100, auction, card_players[player_i].hand.reshape((-1, 32)), [self.vuln_ns, self.vuln_ew], self.models, self.ns, self.ew)

                await asyncio.sleep(0.01)

                card_resp = await card_players[player_i].async_play_card(trick_i, leader_i, current_trick52, rollout_states)
                
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
        for player_i in map(lambda x: x % 4, range(leader_i, leader_i + 4)):
            
            if not isinstance(card_players[player_i], bots.CardPlayer):
                await card_players[player_i].get_card_input()

            card52 = np.nonzero(card_players[player_i].hand52)[0][0]
            card =deck52.card52to32(card52)

            card_resp = CardResp(card=Card.from_code(card52), candidates=[], samples=[])

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

        #print('last trick')
        #print(current_trick)
        #print(current_trick52)
        #print(trick_won_by)

        #pprint.pprint(list(zip(tricks, trick_won_by)))

        self.trick_winners = trick_won_by

        print('\n%s\n' % self.deal_str)

    
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
                self.sampler
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
            if level == 1:
                players.append(self.factory.create_human_bidder(vuln, hands_str[i]))
            else:
                bot = AsyncBotBid(vuln, hands_str[i], self.models, self.ns, self.ew, level, self.sampler, False)
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

    parser = argparse.ArgumentParser(description="Game server")
    parser.add_argument("--boards", default="", help="Filename for configuration")
    parser.add_argument("--boardno", default=0, type=int, help="Board number to start from")
    parser.add_argument("--config", default="./config/default.conf", help="Filename for configuration")
    parser.add_argument("--ns", type=int, default=-1, help="System for NS")
    parser.add_argument("--ew", type=int, default=-1, help="System for EW")

    args = parser.parse_args()

    configfile = args.config

    if args.boards:
        filename = args.boards
        file_extension = os.path.splitext(filename)[1].lower()  
        if file_extension == '.ben':
            with open(filename, "r") as file:
                board_no.append(0) 
                boards = file.readlines()
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
        board_no[0] = args.boardno

    if random:
        print("Playing random deals or deals from the client")
 
    ns = args.ns
    ew = args.ew


    models = Models.from_conf(conf.load(configfile))

    driver = Driver(models, human.ConsoleFactory(), Sample.from_conf(conf.load(configfile)))

    deal_source = random_deal_source()

    while True:
        if random: 
            deal_str, auction_str = next(deal_source)
            driver.set_deal(deal_str, auction_str, ns, ew)
        else:
            rdeal = tuple(boards[board_no[0]].replace("'","").rstrip('\n').split(','))
            print(f"Board: {board_no[0]+1}" )
            print(rdeal)
            driver.set_deal(*rdeal, ns, ew)

        driver.human = [0.1, 0.1, 0.1, 0.1]
        await driver.run()

        with shelve.open('gamedb') as db:
            deal = driver.to_dict()
            db[uuid.uuid4().hex] = deal

        input('\nPress Enter for next deal...')


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
