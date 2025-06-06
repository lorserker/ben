import os
import sys
import platform

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
import traceback
import util
# Intil fixed in Keras, this is needed to remove a wrong warning
import warnings
warnings.filterwarnings("ignore")

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)

# Configure absl logging to suppress logs
import absl.logging
# Suppress Abseil logs
absl.logging.get_absl_handler().python_handler.stream = open(os.devnull, 'w')
absl.logging.set_verbosity(absl.logging.FATAL)
absl.logging.set_stderrthreshold(absl.logging.FATAL)

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU available and memory growth enabled.")
    except RuntimeError as e:
        print("Error setting memory growth:", e)
else:
    print("No GPU detected, using CPU.")


import socket
import shelve

import ipaddress
import argparse
import re
import time
import asyncio
import numpy as np
from sample import Sample
import botbidder
import botopeninglead
import botcardplayer
import conf
import datetime
import pprint
from objects import Card, CardResp, BidResp
from util import get_play_status, get_singleton, get_possible_cards
from nn.opponents import Opponents

from deck52 import card52to32, decode_card, get_trick_winner_i, hand_to_str
from bidding import bidding

from colorama import Fore, Back, Style, init
import gc

import faulthandler
faulthandler.enable()

version = '0.8.7.1'
init()

SEATS = ['North', 'East', 'South', 'West']

def handle_exception(e):
    sys.stderr.write(f"{str(e)}\n")
    traceback_str = traceback.format_exception(type(e), e, e.__traceback__)
    traceback_lines = "".join(traceback_str).splitlines()
    file_traceback = []
    for line in reversed(traceback_lines):
        if line.startswith("  File"):
            file_traceback.append(line.strip()) 
    if file_traceback:
        sys.stderr.write(f"{Fore.RED}")
        sys.stderr.write('\n'.join(file_traceback)+'\n')
        sys.stderr.write(f"{Fore.RESET}")

class TMClient:

    def __init__(self, name, seat, models, sampler, ddsolver, send_info, send_card_info, verbose):
        self.name = name
        self.seat = seat
        self.player_i = SEATS.index(self.seat)
        self.reader = None
        self.writer = None
        self.models = models
        self.sampler = sampler
        self._is_connected = False
        self.verbose = verbose
        self.deal_str = None
        self.trick_winners = None
        self.opponents = None
        self.dds = ddsolver
        self.partner = None
        self.last_explanations = ""
        self.send_info = send_info
        self.send_card_info = send_card_info

    @property
    def is_connected(self):
        return self._is_connected

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
            'board_number': self.board_number,
            'seat': self.seat,
            'opponents': self.opponents,
            'partner': self.partner,
            'models': self.models.name,
            'version': version
        }

    async def run(self, biddingonly, restart, send_info, send_card_info):

        self.bid_responses = []
        self.card_responses = []

        self.dealer_i, self.vuln_ns, self.vuln_ew, self.hand_str = await self.receive_deal(restart)

        auction = await self.bidding()

        await asyncio.sleep(0.01)

        self.contract = bidding.get_contract(auction)
        if  self.contract is None:
            await self.receive_line()
            return

        # level = int(self.contract[0])
        # strain_i = bidding.get_strain_i(self.contract)
        self.decl_i = bidding.get_decl_i(self.contract)
        auction_str = "-".join(auction).replace('PAD_START-', '').replace('PASS','P')

        print(f'{Fore.LIGHTGREEN_EX}{datetime.datetime.now().strftime("%H:%M:%S")} Bidding: {auction_str} {self.contract}{Fore.RESET}')
        if (self.verbose):
            print(f'{datetime.datetime.now().strftime("%H:%M:%S")} Contract {self.contract}')

        if  biddingonly:
            print(f'{datetime.datetime.now().strftime("%H:%M:%S")} Ready to start new board')
            return

        features = {}
        aceking = {}
        if self.bot.bbabot is not None and self.models.use_bba_to_count_aces:
            aceking = self.bot.bbabot.find_aces(auction)
            explanation, _, preempted = self.bot.bbabot.explain_auction(auction)
            features["Explanation"] = explanation
            features["preempted"] = preempted
        features["aceking"] = aceking

        if self.verbose:
            if self.bot.bbabot is not None:
                self.bot.bbabot.get_sample(auction)

        opening_lead_card = await self.opening_lead(auction, aceking)
        opening_lead52 = Card.from_symbol(opening_lead_card).code()

        if self.player_i != (self.decl_i + 2) % 4:
            self.dummy_hand_str = await self.receive_dummy()

        await self.play(auction, opening_lead52, features, send_card_info)

        await self.receive_line()
        

    async def connect(self, host, port, send_info, send_card_info, config):
        try:
            # Check if the host is already an IP address
            ipaddress.ip_address(host)
            resolved_ip = host  # It's already an IP address
        except ValueError:
            try:
                # Not a valid IP, resolve it as a hostname
                resolved_ip = socket.gethostbyname(host)
                print(f"Resolved hostname '{host}' to IP: {resolved_ip}")
            except Exception as e:
                print(f"Failed to connect to {host}:{port} - {e}")            
                handle_exception(e)
                self._is_connected = False
                print(Style.RESET_ALL)
                sys.exit(1)

        self.reader, self.writer = await asyncio.open_connection(resolved_ip, port)
        self._is_connected = True
        print(f"Connected to {host}:{port}")

        await self.send_message(f'Connecting "{self.name}" as {self.seat} using protocol version 18{" " + config if send_info else ""}')

        # Validate response Blue Chip can send: Error Team name mismatch
        await self.receive_line()
        
        await self.send_message(f'{self.seat} ready for teams')

        match_details = await self.receive_line()
        
        # Regular expression pattern to match text in quotes
        pattern = r'N/S : "([^"]+)" E/W : "([^"]+)"(?:.*?Playing (\w+))?'   
        # Find all matches in the string
        details = re.findall(pattern, match_details)

        ns_name, ew_name, imp = details[0]  # Unpack the tuple
        # Extracted text from the second set of quotes
        if len(details[0]) > 1:
            if self.seat == "North" or self.seat == "South":
                self.partner = ns_name
                self.opponents = ew_name
            else:
                self.partner = ew_name
                self.opponents = ns_name
        print(f"{Fore.LIGHTGREEN_EX}Partner: {self.partner}, Opponents: {self.opponents}{Fore.RESET}")        
        # Perhaps we should use the IMP/MP from table manager
        # print(f"IMP/MP: {imp}")
        if self.models.matchpoint:
            if imp == "IMP":
                print("Using MP configuration")
                sys.exit(1)
                

    async def bidding(self):
        vuln = [self.vuln_ns, self.vuln_ew]

        self.bot = botbidder.BotBid(vuln, self.hand_str, self.models, self.sampler, self.player_i, self.dealer_i, self.dds, False, self.verbose)

        auction = ['PAD_START'] * self.dealer_i

        player_i = self.dealer_i

        while not bidding.auction_over(auction):
            if player_i == self.player_i:
                # now it's this player's turn to bid
                bid_resp = self.bot.bid(auction)
                if (self.verbose):
                    print(bid_resp)
                auction.append(bid_resp.bid)
                # We read the explanations from BBA
                if self.models.consult_bba:
                    explanation, alert = self.bot.explain(auction)
                    bid_resp.explanation = explanation
                    bid_resp.alert = alert
                    # Are we in a keycard sequence
                    if not self.bot.bba_is_controlling:
                        self.bot.bba_is_controlling = self.bot.is_bba_controlling(bid_resp.bid, explanation)
                        
                self.bid_responses.append(bid_resp)
                await self.send_own_bid(bid_resp)
            else:
                # just wait for the other player's bid
                bid, alert, explanation = await self.receive_bid_for(player_i)
                if (player_i + 2) % 4 == self.player_i:
                    bid_resp = BidResp(bid=bid, candidates=[], samples=[], shape=-1, hcp=-1, who=self.partner, quality=None, alert=alert, explanation=explanation)
                else:
                    bid_resp = BidResp(bid=bid, candidates=[], samples=[], shape=-1, hcp=-1, who=self.opponents, quality=None, alert=alert, explanation=explanation)
                self.bid_responses.append(bid_resp)
                # We will have to save the explanation, so it can be used in the sampling
                auction.append(bid)

            player_i = (player_i + 1) % 4

        return auction

    async def opening_lead(self, auction, aceking):

        contract = bidding.get_contract(auction)
        decl_i = bidding.get_decl_i(contract)
        on_lead_i = (decl_i + 1) % 4
        
        if self.player_i == on_lead_i:
            # this player is on lead
            await self.receive_line()

            bot_lead = botopeninglead.BotLead(
                [self.vuln_ns, self.vuln_ew], 
                self.hand_str,
                self.models,
                self.sampler,
                on_lead_i,
                self.dealer_i,
                self.dds,
                self.verbose
            )
            card_resp = bot_lead.find_opening_lead(auction, aceking)
            self.card_responses.append(card_resp)
            card_symbol = card_resp.card.symbol()

            await asyncio.sleep(0.01)

            await self.send_card_played(card_symbol, card_resp.who)

            await asyncio.sleep(0.01)

            return card_symbol
        else:
            # just send that we are ready for the opening lead
            return await self.receive_card_play_for(on_lead_i, 0)

    async def play(self, auction, opening_lead52, features, send_card_info ):
        self.send_card_info = send_card_info
        contract = bidding.get_contract(auction)
        
        level = int(contract[0])
        strain_i = bidding.get_strain_i(contract)
        decl_i = bidding.get_decl_i(contract)
        is_decl_vuln = [self.vuln_ns, self.vuln_ew, self.vuln_ns, self.vuln_ew][decl_i]
        cardplayer_i = (self.player_i + 3 - decl_i) % 4  # lefty=0, dummy=1, righty=2, decl=3

        own_hand_str = self.hand_str
        dummy_hand_str = '...'

        if cardplayer_i != 1:
            dummy_hand_str = self.dummy_hand_str

        lefty_hand_str = '...'
        if cardplayer_i == 0:
            lefty_hand_str = own_hand_str
        
        righty_hand_str = '...'
        if cardplayer_i == 2:
            righty_hand_str = own_hand_str
        
        decl_hand_str = '...'
        if cardplayer_i == 3:
            decl_hand_str = own_hand_str

        pimc = [None, None, None, None]

        # We should only instantiate the PIMC for the position we are playing
        if self.models.pimc_use_declaring and (cardplayer_i == 1 or cardplayer_i == 3): 
            from pimc.PIMC import BGADLL
            declarer = BGADLL(self.models, dummy_hand_str, decl_hand_str, contract, is_decl_vuln, self.sampler, self.verbose)
            pimc[1] = declarer
            pimc[3] = declarer
            if self.verbose:
                print("PIMC",dummy_hand_str, decl_hand_str, contract)
        else:
            pimc[1] = None
            pimc[3] = None
        if self.models.pimc_use_defending and cardplayer_i == 0:
            from pimc.PIMCDef import BGADefDLL
            pimc[0] = BGADefDLL(self.models, dummy_hand_str, lefty_hand_str, contract, is_decl_vuln, 0, self.sampler, self.verbose)
            if self.verbose:
                print("PIMC",dummy_hand_str, lefty_hand_str, righty_hand_str, contract)
        else:
            pimc[0] = None
        if self.models.pimc_use_defending and cardplayer_i == 2:
            from pimc.PIMCDef import BGADefDLL
            pimc[2] = BGADefDLL(self.models, dummy_hand_str, righty_hand_str, contract, is_decl_vuln, 2, self.sampler, self.verbose)
            if self.verbose:
                print("PIMC",dummy_hand_str, lefty_hand_str, righty_hand_str, contract)
        else:
            pimc[2] = None

        card_players = [
            botcardplayer.CardPlayer(self.models, 0, lefty_hand_str, dummy_hand_str, contract, is_decl_vuln, self.sampler, pimc[0], self.dds, self.verbose),
            botcardplayer.CardPlayer(self.models, 1, dummy_hand_str, decl_hand_str, contract, is_decl_vuln, self.sampler, pimc[1], self.dds, self.verbose),
            botcardplayer.CardPlayer(self.models, 2, righty_hand_str, dummy_hand_str, contract, is_decl_vuln, self.sampler, pimc[2], self.dds, self.verbose),
            botcardplayer.CardPlayer(self.models, 3, decl_hand_str, dummy_hand_str, contract, is_decl_vuln, self.sampler, pimc[3], self.dds, self.verbose)
        ]

        player_cards_played = [[] for _ in range(4)]
        player_cards_played52 = [[] for _ in range(4)]
        shown_out_suits = [set() for _ in range(4)]
        discards = [set() for _ in range(4)]
        
        leader_i = 0

        tricks = []
        tricks52 = []
        trick_won_by = []

        opening_lead = card52to32(opening_lead52)

        current_trick = [opening_lead]
        current_trick52 = [opening_lead52]

        card_players[0].hand52[opening_lead52] -= 1

        for trick_i in range(12):
            print("{}            Playing trick {}".format(datetime.datetime.now().strftime("%H:%M:%S"),trick_i+1))

            for player_i in map(lambda x: x % 4, range(leader_i, leader_i + 4)):
                if (self.verbose):
                    print('player {}'.format(player_i))

                nesw_i = (decl_i + player_i + 1) % 4 # N=0, E=1, S=2, W=3
                
                if trick_i == 0 and player_i == 0:
                    #print('skipping')
                    for i, card_player in enumerate(card_players):
                        card_player.set_real_card_played(opening_lead52, player_i)
                        card_player.set_card_played(trick_i=trick_i, leader_i=leader_i, i=0, card=opening_lead)

                    continue

                card52 = None
                card_resp = None
                # it's dummy's turn and this is the declarer
                if (player_i == 1 and cardplayer_i == 3):
                    print('{}            Declarers turn for dummy'.format(datetime.datetime.now().strftime("%H:%M:%S")))

                if (player_i == cardplayer_i and player_i != 1) or (player_i == 1 and cardplayer_i == 3):
                    play_status = get_play_status(card_players[player_i].hand52,current_trick52, strain_i)

                    if play_status == "Forced":
                        card = get_singleton(card_players[player_i].hand52,current_trick52)
                        card_resp = CardResp(
                            card=Card.from_code(card),
                            candidates=[],
                            samples=[],
                            shape=-1,
                            hcp=-1, 
                            quality=None,
                            who="Forced",
                            claim = -1
                        )
                    # if play status = follow 
                    # and all out cards are equal value (like JT9)
                    # the play lowest if defending and highest if declaring
                    if play_status == "Follow" and card_resp == None:
                        result = get_possible_cards(card_players[player_i].hand52,current_trick52)
                        if result[0] != -1:
                            card = result[0] if player_i == 3 else result[1]
                            card_resp = CardResp(
                                card=Card.from_code(card),
                                candidates=[],
                                samples=[],
                                shape=-1,
                                hcp=-1,
                                quality=None,
                                who="Follow",
                                claim = -1
                            )                        

                    # if card_resp is None, we have to rollout
                    if card_resp == None:
                        vuln = [self.vuln_ns, self.vuln_ew]
                        played_cards = [card for row in player_cards_played52 for card in row] + current_trick52
                        rollout_states, bidding_scores, c_hcp, c_shp, quality, probability_of_occurence, lead_scores, play_scores, logical_play_scores, discard_scores, worlds = self.sampler.init_rollout_states(trick_i, player_i, card_players, played_cards, player_cards_played, shown_out_suits, discards, features["aceking"], current_trick, opening_lead52,auction, card_players[player_i].hand_str, card_players[player_i].public_hand_str, vuln, self.models, card_players[player_i].get_random_generator())
                        card_players[player_i].check_pimc_constraints(trick_i, rollout_states, quality)
                        card_resp = card_players[player_i].play_card(trick_i, leader_i, current_trick52, tricks52, rollout_states, worlds, bidding_scores, quality, probability_of_occurence, shown_out_suits, play_status, lead_scores, play_scores, logical_play_scores, discard_scores, features)
                        card_resp.hcp = c_hcp
                        card_resp.shape = c_shp

                        if (self.verbose):
                            for sample in enumerate(card_resp.samples):                  
                                print(f"{sample}")

                    self.card_responses.append(card_resp)

                    card52 = card_resp.card.code()
                    
                    if (player_i == 1 and cardplayer_i == 3):
                        await self.send_card_played_for_dummy(card_resp.card.symbol(), card_resp.who) 
                    else:
                        await self.send_card_played(card_resp.card.symbol(),card_resp.who) 
                        
                    await asyncio.sleep(0.01)

                else:
                    # another player is on play, we just have to wait for their card
                    card52_symbol = await self.receive_card_play_for(nesw_i, trick_i)
                    card52 = Card.from_symbol(card52_symbol).code()
                
                card32 = card52to32(card52)
                
                for card_player in card_players:
                    card_player.set_real_card_played(card52, player_i)
                    card_player.set_card_played(trick_i=trick_i, leader_i=leader_i, i=player_i, card=card32)

                current_trick.append(card32)

                current_trick52.append(card52)

                card_players[player_i].set_own_card_played52(card52)
                if player_i == 1:
                    for i in [0, 2, 3]:
                        card_players[i].set_public_card_played52(card52)
                if player_i == 3:
                    card_players[1].set_public_card_played52(card52)

                # update shown out state
                if card32 // 8 != current_trick[0] // 8:  # card is different suit than lead card
                    shown_out_suits[player_i].add(current_trick[0] // 8)
                    discards[player_i].add((trick_i,card32))
                        

            # sanity checks after trick completed
            assert len(current_trick) == 4

            for i in [cardplayer_i] + ([1] if cardplayer_i == 3 else []):
                if cardplayer_i == 1:
                    break
                assert np.min(card_players[i].hand52) == 0
                assert np.min(card_players[i].public52) == 0
                assert np.sum(card_players[i].hand52) == 13 - trick_i - 1
                assert np.sum(card_players[i].public52) == 13 - trick_i - 1

            tricks.append(current_trick)
            tricks52.append(current_trick52)

            if self.models.pimc_use_declaring or self.models.pimc_use_defending:
                for card_player in card_players:
                    if isinstance(card_player, botcardplayer.CardPlayer) and card_player.pimc:
                        card_player.pimc.reset_trick()

            # initializing for the next trick
            # initialize hands
            for i, card32 in enumerate(current_trick):
                card_players[(leader_i + i) % 4].x_play[:, trick_i + 1, 0:32] = card_players[(leader_i + i) % 4].x_play[:, trick_i, 0:32]
                card_players[(leader_i + i) % 4].x_play[:, trick_i + 1, 0 + card32] -= 1

            # initialize public hands
            for i in (0, 2, 3):
                card_players[i].x_play[:, trick_i + 1, 32:64] = card_players[1].x_play[:, trick_i + 1, 0:32]
            card_players[1].x_play[:, trick_i + 1, 32:64] = card_players[3].x_play[:, trick_i + 1, 0:32]

            for card_player in card_players:
                # initialize last trick
                for i, card32 in enumerate(current_trick):
                    card_player.x_play[:, trick_i + 1, 64 + i * 32 + card32] = 1
                    
                # initialize last trick leader
                card_player.x_play[:, trick_i + 1, 288 + leader_i] = 1

                # initialize level
                card_player.x_play[:, trick_i + 1, 292] = level

                # initialize strain
                card_player.x_play[:, trick_i + 1, 293 + strain_i] = 1

            # sanity checks for next trick
            for i in [cardplayer_i] + ([1] if cardplayer_i == 3 else []):
                if cardplayer_i == 1:
                    break
                assert np.min(card_players[i].x_play[:, trick_i + 1, 0:32]) == 0
                assert np.min(card_players[i].x_play[:, trick_i + 1, 32:64]) == 0
                assert np.sum(card_players[i].x_play[:, trick_i + 1, 0:32], axis=1) == 13 - trick_i - 1
                assert np.sum(card_players[i].x_play[:, trick_i + 1, 32:64], axis=1) == 13 - trick_i - 1

            trick_winner = (leader_i + get_trick_winner_i(current_trick52, (strain_i - 1) % 5)) % 4
            trick_won_by.append(trick_winner)

            if trick_winner % 2 == 0:
                card_players[0].n_tricks_taken += 1
                card_players[2].n_tricks_taken += 1
                if self.models.pimc_use_defending:
                    if isinstance(card_players[0], botcardplayer.CardPlayer) and card_players[0].pimc:
                        card_players[0].pimc.update_trick_needed()
                    if isinstance(card_players[2], botcardplayer.CardPlayer) and card_players[2].pimc:
                        card_players[2].pimc.update_trick_needed()
            else:
                card_players[1].n_tricks_taken += 1
                card_players[3].n_tricks_taken += 1
                if self.models.pimc_use_declaring:
                    if isinstance(card_players[3], botcardplayer.CardPlayer) and card_players[3].pimc :
                        card_players[3].pimc.update_trick_needed()

            print('{}            trick {} cards={} won by {}'.format(datetime.datetime.now().strftime("%H:%M:%S"),trick_i+1, list(map(decode_card, current_trick52)), "NESW"[(trick_winner + self.decl_i + 1) % 4]))

            # update cards shown
            for i, card32 in enumerate(current_trick):
                player_cards_played[(leader_i + i) % 4].append(card32)
            for i, card52 in enumerate(current_trick52):
                player_cards_played52[(leader_i + i) % 4].append(card52)
            
            leader_i = trick_winner
            current_trick = []
            current_trick52 = []

            # player on lead will receive message (or decl if dummy on lead)
            if leader_i == 1:
                if cardplayer_i == 3:
                    #print("waiting for message for lead")
                    await self.receive_line()
            elif leader_i == cardplayer_i:
                #print("waiting for message for lead")
                await self.receive_line()
            # Give player to lead to receive message
            await asyncio.sleep(0.1)

        # play last trick
        trick_i += 1
        for player_i in map(lambda x: x % 4, range(leader_i, leader_i + 4)):
            nesw_i = (decl_i + player_i + 1) % 4 # N=0, E=1, S=2, W=3
            card52 = None
            #If we are declarer and it is dummy to lead we must send "direction for dummy"
            if player_i == 1 and cardplayer_i == 3 or player_i == cardplayer_i and player_i != 1:
                # we are on play
                card52 = int(np.nonzero(card_players[player_i].hand52)[0][0])
                card52_symbol = Card.from_code(card52).symbol()

                cr = CardResp(
                    card=Card.from_symbol(card52_symbol),
                    candidates=[],
                    samples=[],
                    shape=-1,
                    hcp=-1,
                    quality=None,
                    who = "Forced",
                    claim = -1
                )
                self.card_responses.append(cr)

                await asyncio.sleep(0.01)
                
                if player_i == 1 and cardplayer_i == 3:
                    await self.send_card_played_for_dummy(card52_symbol, cr.who)
                else:    
                    await self.send_card_played(card52_symbol, cr.who)

                await asyncio.sleep(0.01)
                self.card_responses
            else:
                # someone else is on play. we just have to wait for their card
                card52_symbol = await self.receive_card_play_for(nesw_i, trick_i)
                card52 = Card.from_symbol(card52_symbol).code()

            card32 = card52to32(card52)

            current_trick.append(card32)
            current_trick52.append(card52)

        # update cards shown
        for i, card32 in enumerate(current_trick):
            player_cards_played[(leader_i + i) % 4].append(card32)
        for i, card52 in enumerate(current_trick52):
            player_cards_played52[(leader_i + i) % 4].append(card52)

        tricks.append(current_trick)
        tricks52.append(current_trick52)

        trick_winner = (leader_i + get_trick_winner_i(current_trick52, (strain_i - 1) % 5)) % 4
        trick_won_by.append(trick_winner)

        self.trick_winners = trick_won_by
        
        if (self.verbose):
            #pprint.pprint(list(zip(tricks, trick_won_by)))
            # Decode each element of tricks52
            decoded_tricks52 = [[decode_card(item) for item in inner] for inner in tricks52]
            pprint.pprint(list(zip(decoded_tricks52, trick_won_by)))

        concatenated_str = ""

        for i in range(len(player_cards_played52)):
            index = (i + (3-decl_i)) % len(player_cards_played52)
            concatenated_str += hand_to_str(player_cards_played52[index]) + " "

        # Remove the trailing space
        self.deal_str = concatenated_str.strip()


    async def send_card_played(self, card_symbol, who):
        msg_card = f'{self.seat} plays {card_symbol[::-1]}{". " + who if self.send_card_info else ""}'
        await self.send_message(msg_card)

    async def send_card_played_for_dummy(self, card_symbol, who):
        dummy_i = (self.decl_i + 2) % 4
        seat = SEATS[dummy_i]
        msg_card = f'{seat} plays {card_symbol[::-1]}{". " + who if self.send_card_info else ""}'
        await self.send_message(msg_card)

    async def send_own_bid(self, bid_resp):
        bid = bid_resp.bid.replace('N', 'NT')
        msg_bid = f'{SEATS[self.player_i]} bids {bid}'
        if bid == 'PASS':
            msg_bid = f'{SEATS[self.player_i]} passes'
        elif bid == 'X':
            msg_bid = f'{SEATS[self.player_i]} doubles'
        elif bid == 'XX':
            msg_bid = f'{SEATS[self.player_i]} redoubles'
        if bid_resp.alert:
            msg_bid += ' Alert. ' + bid_resp.explanation
            if bid_resp.who == "Rescue":
                msg_bid += ' Rescue.'
            self.last_explanations = bid_resp.explanation
        else:
            if self.send_info:
                if bid_resp.explanation:
                    if bid_resp.explanation not in self.last_explanations:
                        msg_bid += ' Infos. ' + bid_resp.explanation
                        if bid_resp.who == "Rescue":
                            msg_bid += ' Rescue.'
                        self.last_explanations = bid_resp.explanation
                    else:
                        if bid_resp.who == "Rescue":
                            msg_bid += ' Infos. Rescue.'
            
        await self.send_message(msg_bid)

    async def receive_card_play_for(self, player_i, trick_i):
        # We need to find out if it is dummy we are waiting for
        if ((self.decl_i + 2) % 4 == player_i):
            waiting_for = "dummy"
        else:
            waiting_for = SEATS[player_i]
        msg_ready = f"{self.seat} ready for {waiting_for}'s card to trick {trick_i + 1}"
        await self.send_message(msg_ready)

        card_resp = await self.receive_line()

        card_resp_parts = card_resp.strip().split('.')
        card_info = card_resp_parts[1] if len(card_resp_parts) > 1 else None
        card_resp_parts = card_resp_parts[0].strip().split()
        if self.verbose:
            print("card_resp_parts", card_resp_parts)
        if card_resp.lower() == "start of board":
            raise RestartLogicException(card_resp)

        assert card_resp_parts[0] == SEATS[player_i], f"Received {card_resp_parts} - was expecting card for {SEATS[player_i]}"
        if player_i % 2 == self.player_i:
            who = self.partner
        else:
            who = self.opponents

        if card_info:
            who += " " + card_info

        cr = CardResp(
            card=Card.from_symbol(card_resp_parts[2][::-1].upper().replace(".", "")),
            candidates=[],
            samples=[],
            shape=-1,
            hcp=-1, 
            quality=None,
            who=who,
            claim = -1
        )
        self.card_responses.append(cr)

        return card_resp_parts[-1][::-1].upper()

    async def receive_bid_for(self, player_i):
        msg_ready = f"{SEATS[self.player_i]} ready for {SEATS[player_i]}'s bid"
        await self.send_message(msg_ready)
        
        bid_resp = await self.receive_line()
        bid_resp_parts = bid_resp.strip().split()

        if bid_resp.lower() == "start of board":
            raise RestartLogicException(bid_resp)

        assert bid_resp_parts[0] == SEATS[player_i], f"Received {bid_resp} - was expecting bid for {SEATS[player_i]} - restart client"

        # This is to prevent the client failing, when receiving an alert
        if (bid_resp_parts[1].upper() not in ['PASSES', 'DOUBLES', 'REDOUBLES']):
            bid = bid_resp_parts[2].rstrip('.').upper().replace('NT', 'N')
        else:
            bid = bid_resp_parts[1].upper()

        explanation = None
        alert = False
        # Check for the presence of "Alert." or "Infos."
        if "Alert." in bid_resp_parts:
            # Find the index of "Alert."
            index = bid_resp_parts.index("Alert.")
            # Get all parts after "Alert."
            explanation = ' '.join(bid_resp_parts[index + 1:])
            alert = True
        elif "Infos." in bid_resp_parts:
            # Find the index of "Infos."
            index = bid_resp_parts.index("Infos.")
            # Get all parts after "Infos."
            explanation = ' '.join(bid_resp_parts[index + 1:])

        return {
            'PASSES': 'PASS',
            'DOUBLES': 'X',
            'REDOUBLES': 'XX'
        }.get(bid, bid), alert, explanation

    async def receive_dummy(self):
        dummy_i = (self.decl_i + 2) % 4

        if self.player_i == dummy_i:
            return self.hand_str
        else:
            msg_ready = f'{self.seat} ready for dummy'
            await self.send_message(msg_ready)
            line = await self.receive_line()
            if (line.lower() == "start of board"):
                raise RestartLogicException(line)

            # Dummy's cards : S A Q T 8 2. H K 7. D K 5 2. C A 7 6.
            return TMClient.parse_hand(line)

    async def send_ready(self):
        await self.send_message(f'{self.seat} ready to start')

    async def receive_deal(self, restart):
        np.random.seed(42)

        if restart:
            deal_line_1 = "Start of Board"
        else:
            deal_line_1 = await self.receive_line()

        while deal_line_1.lower() != "start of board":
            await self.send_message(f'{self.seat} ready to start')
            deal_line_1 = await self.receive_line()
            if deal_line_1 is None or deal_line_1 == "":
                print("Empty response from TM, terminating")
                sys.exit(1)

        while deal_line_1.lower() == "start of board":
            await self.send_message(f'{self.seat} ready for deal')
            deal_line_1 = await self.receive_line()

        if "number" not in deal_line_1:
            # Restart so we get the hand next
            print("Protocol error - expecting board number", deal_line_1)
            raise ValueError("Deal not received")

        pattern = r'Board number (\d+)\.'

        match = re.search(pattern, deal_line_1)

        if match:
            self.board_number = match.group(1)

        await self.send_message(f'{self.seat} ready for cards')
        # "South's cards : S K J 9 3. H K 7 6. D A J. C A Q 8 7. \r\n"
        # "North's cards : S 9 3. H -. D J 7 5. C A T 9 8 6 4 3 2."
        deal_line_2 = await self.receive_line()
        if deal_line_2.lower() == "start of board":
            raise RestartLogicException(deal_line_2)

        rx_dealer_vuln = r'(?P<dealer>[a-zA-Z]+?)\.\s(?P<vuln>.+?)\svulnerable'
        match = re.search(rx_dealer_vuln, deal_line_1)

        if deal_line_2 is None or deal_line_2 == "":
            print("Deal not received", deal_line_2)
            raise ValueError("Deal not received")
        
        hand_str = TMClient.parse_hand(deal_line_2)
        
        dealer_i = 'NESW'.index(match.groupdict()['dealer'][0])
        vuln_str = match.groupdict()['vuln']
        assert vuln_str in {'Neither', 'N/S', 'E/W', 'Both'}
        vuln_ns = vuln_str == 'N/S' or vuln_str == 'Both'
        vuln_ew = vuln_str == 'E/W' or vuln_str == 'Both'
        self.last_explanations = ""
        return dealer_i, vuln_ns, vuln_ew, hand_str
    
    @staticmethod
    def parse_hand(s):
        # Translate hand from Tablemanager format to PBN
        try:
            hand = s[s.index(':') + 1 : s.rindex('.')] \
                .replace(' ', '').replace('-', '').replace('S', '').replace('H', '').replace('D', '').replace('C', '')
            return hand
        except Exception as ex:
            print("Parse hand",s)
            print(ex)
            print(f"Protocol error. Received {s} - Expected a hand")


    async def send_message(self, message: str):
        try:
            print(f'{datetime.datetime.now().strftime("%H:%M:%S")} sending:   {message.ljust(57)}', end='')

            self.writer.write((message + "\r\n").encode())
            await self.writer.drain()
            print(' ...sent successfully.')
            await asyncio.sleep(0.05)
        except ConnectionAbortedError as ex:
            print(f'Error: {str(ex)}')
            # Handle the error gracefully, such as logging it or notifying the user
            # Close the connection (in case it's not already closed)
            self._is_connected = False
            # Stop the event loop to terminate the application
            print(Style.RESET_ALL)
            sys.exit()
        except Exception as ex:
            print(f'Error: {str(ex)}')
            # Handle the error gracefully, such as logging it or notifying the user
            # Close the connection (in case it's not already closed)
            self._is_connected = False
            # Stop the event loop to terminate the application
            print(Style.RESET_ALL)
            sys.exit()

    async def receive_line(self) -> str:
        try:
            message = await self.reader.readline()
            print('{} receiving: '.format(datetime.datetime.now().strftime("%H:%M:%S")), end='')
            msg = message.decode().replace('\r', '').replace('\n', '')
            if msg.startswith('Timing'):
                msg = msg.replace('E/W','\n                             E/W') + "        "
            print(f'{msg.ljust(57)} ...received.')
            if (msg == "End of session"):
                # Stop the event loop to terminate the application
                self._is_connected = False
                print(Style.RESET_ALL)
                sys.exit(0)
            return msg
        except ConnectionAbortedError as ex:
            print(f'Match terminated:\n {str(ex)}\n')    
            # Close the connection (in case it's not already closed)
            self._is_connected = False
            # Stop the event loop to terminate the application
            print(Style.RESET_ALL)
            sys.exit(0)
        except Exception as ex:
            print(f'Match terminated:\n {str(ex)}\n')    
            # Close the connection (in case it's not already closed)
            self._is_connected = False
            # Stop the event loop to terminate the application
            print(Style.RESET_ALL)
            sys.exit(0)

def validate_ip(ip_str):
    try:
        ip = ipaddress.ip_address(ip_str)
        return str(ip)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{ip_str}' is not a valid IP address")
    
def get_execution_path():
    # Get the directory where the program is started from either PyInstaller executable or the script
    return os.getcwd()


def cleanup_shelf(shelf_filename):
    # Delete the shelf file if it exists
    if os.path.exists(shelf_filename + '.dat'):
        os.remove(shelf_filename + '.dat')
        # Remove other associated files generated by shelve
        for ext in ('.bak', '.dat', '.dir'):
            filename = shelf_filename + ext
            if os.path.exists(filename):
                os.remove(filename)


def validate_ip_or_hostname(value):
    """Validate that the value is either an IP address or a hostname."""
    # Check if the value is a valid IP address
    try:
        socket.inet_aton(value)
        return value
    except socket.error:
        pass
    
    print("Validating hostname",value)
    # Validate hostname using a regex
    hostname_regex = re.compile(
        r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)(?:\.[A-Za-z0-9-]{1,63})*$"
    )
    if hostname_regex.match(value):
        return value

    raise argparse.ArgumentTypeError(f"Invalid IP address or hostname: {value}")

# Custom function to convert string to boolean
def str_to_bool(value):
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    raise ValueError("Invalid boolean value")

class RestartLogicException(Exception):
    """Custom exception to signal a restart of the main application logic."""
    pass

#  Examples of how to start the table manager
# python table_manager_client.py --name BEN --seat North
# python table_manager_client.py --name BEN --seat South

async def main():
    
    # Get the path to the config file
    config_path = get_execution_path()

    parser = argparse.ArgumentParser(description="Table manager interface")
    parser.add_argument("--host", type=validate_ip_or_hostname, default="127.0.0.1", help="IP address or hostname for Table Manager")    
    parser.add_argument("--port", type=int, default=2000, help="Port for Table Manager")
    parser.add_argument("--name", required=True, help="Name in Table Manager")
    parser.add_argument("--seat", required=True, help="Where to sit (North, East, South or West)")
    parser.add_argument("--config", default=f"{config_path}/config/default.conf", help="Filename for configuration")
    parser.add_argument("--opponent", default="", help="Filename for configuration pf opponents")
    parser.add_argument("--biddingonly", type=str_to_bool, default=False, help="Only bid, no play")
    parser.add_argument("--nosearch", type=str_to_bool, default=False, help="Just use neural network")
    parser.add_argument("--matchpoint", type=str_to_bool, default=None, help="Playing match point")
    parser.add_argument("--sendinfo", type=str_to_bool, default=False, help="Send information to Table Manager")
    parser.add_argument("--sendcardinfo", type=str_to_bool, default=False, help="Send card information to Table Manager")
    parser.add_argument("--verbose", type=str_to_bool, default=False, help="Output samples and other information during play")

    args = parser.parse_args()

    host = args.host
    port = args.port
    name = args.name
    seat = args.seat
    configfile = args.config
    opponentfile = args.opponent
    matchpoint = args.matchpoint
    verbose = args.verbose
    biddingonly = args.biddingonly
    nosearch = args.nosearch
    send_info = args.sendinfo
    send_card_info = args.sendcardinfo

    np.set_printoptions(precision=2, suppress=True, linewidth=200)

    print("BEN_HOME=",os.getenv('BEN_HOME'))

    print(f"{Fore.CYAN}{datetime.datetime.now():%Y-%m-%d %H:%M:%S} table_manager_client.py - Version {version}")
    if util.is_pyinstaller_executable():
        print(f"Running inside a PyInstaller-built executable. {platform.python_version()}")
    else:
        print(f"Running in a standard Python environment: {platform.python_version()}")

    print(f"Python version: {sys.version}{Fore.RESET}")

    if sys.platform == 'win32':
        # Print the PythonNet version
        sys.stderr.write(f"PythonNet: {util.get_pythonnet_version()}\n") 
        sys.stderr.write(f"{util.check_dotnet_version()}\n") 

    # Try to fetch Keras version or handle older TensorFlow versions
    try:
        keras_version = tf.keras.__version__
    except AttributeError:
        keras_version = "Not integrated with TensorFlow"
        configfile = configfile.replace("default.conf", "TF1.x/default_tf1x.conf")

    # Write to stderr
    sys.stderr.write(f"Loading TensorFlow {tf.__version__} - Keras version: {keras_version}\n")
    sys.stderr.write(f"NumPy Version : {np.__version__}\n")

    configuration = conf.load(configfile)

    try:
        if (configuration["models"]['tf_version'] == "2"):
            from nn.models_tf2 import Models
        else: 
            # Default to version 1. of Tensorflow
            from nn.models_tf2 import Models
    except KeyError:
            # Default to version 1. of Tensorflow
            from nn.models_tf2 import Models

    print("Config:", configfile)
    if opponentfile != "":
        # Override with information from opponent file
        print("Opponent:", opponentfile)
        configuration.read(opponentfile)
        opponents = Opponents.from_conf(configuration, config_path.replace(os.path.sep + "src",""))
        sys.stderr.write(f"Expecting opponent: {opponents.name}\n")

    models = Models.from_conf(configuration, config_path.replace(os.path.sep + "src",""))

    if sys.platform != 'win32':
        print("Disabling PIMC/BBA/SuitC as platform is not win32")
        models.pimc_use_declaring = False
        models.pimc_use_defending = False
        models.use_bba = False
        models.consult_bba = False
        models.use_bba_rollout = False
        models.use_bba_to_count_aces = False
        #models.use_suitc = False
        
    if models.use_bba:
        print("Using BBA for bidding")
    else:
        print("Model:   ", os.path.basename(models.bidder_model.model_path))
        print("Opponent:", os.path.basename(models.opponent_model.model_path))

    if models.use_bba or models.use_bba_to_count_aces or models.consult_bba or models.use_bba_rollout:
        from bba.BBA import BBABotBid
        bot = BBABotBid(None, None ,None, None, None, None, None, verbose)
        print(f"BBA enabled. Version {bot.version()}")    
        print(f"Our BBA convention card: {models.bba_our_cc}")
        print(f"Their BBA convention card: {models.bba_their_cc}")

    if models.use_suitc:
        from suitc.SuitC import SuitCLib
        suitc = SuitCLib(verbose)
        print(f"SuitC enabled. Version {suitc.version()}")

    if models.pimc_use_declaring or models.pimc_use_defending:
        from pimc.PIMC import BGADLL
        pimc = BGADLL(None, None, None, None, None, None, verbose)
        from pimc.PIMCDef import BGADefDLL
        pimcdef = BGADefDLL(None, None, None, None, None, None, None, verbose)
        print(f"PIMC enabled. Version {pimc.version()}")
        print(f"PIMCDef enabled. Version {pimcdef.version()}")

    from ddsolver import ddsolver
    dds = ddsolver.DDSolver()
    print(f"DDSolver enabled. Version {dds.version()}")
    
    # Not supported by TM, so no need to calculate
    models.claim = False

    if matchpoint is not None:
        models.matchpoint = matchpoint

    if nosearch:
        print("Simulation disabled")
        models.search_threshold = -1

    if models.matchpoint:
        print("Matchpoint mode on")
    else:
        print("Playing IMPS mode")

    sampling = Sample.from_conf(configuration, verbose=verbose)
    client = TMClient(name, seat, models, sampling, dds, send_info, send_card_info, verbose)
    print(f"Connecting to {host}:{port} as {seat} {"sending info" if send_info else ""}...")
    config = f"{models.name} {version} {configfile} {opponentfile}" 
    await client.connect(host, port, send_info, send_card_info, config)
    first = True

    if client.is_connected:
        await client.send_ready()

    shelf_filename = f"{config_path}/{seat}-{name}"
    restart = False
    while client.is_connected:
        t_start = time.time()
        try:
            await client.run(biddingonly, restart, send_info, send_card_info)
        except RestartLogicException as e:
            print(f"{Fore.CYAN}{datetime.datetime.now():%H:%M:%S} Communication restarted from Table Manager{Fore.RESET}")
            restart = True
        # The deal just played is saved for later review
        # if bidding only we do not save the deal
        if restart:
            continue
        if not biddingonly:
            # If we just played board 1 we assume a new match
            deal = client.to_dict()
            if deal["board_number"] == "1" and first:
                cleanup_shelf(shelf_filename)
                first = False
            with shelve.open(shelf_filename) as db:
                print(f"{datetime.datetime.now():%H:%M:%S} Saving Board:",client.deal_str)
                print(f'{Fore.CYAN}{datetime.datetime.now():%H:%M:%S} Board played in {time.time() - t_start:0.1f} seconds.{Fore.RESET}')  
                if deal["board_number"]+"-Open" not in db:
                    db[deal["board_number"]+"-Open"] = deal
                else:
                    db[deal["board_number"]+"-Closed"] = deal
        np.empty(0) 
        gc.collect()

def initialize_logging(argv):
    # Empty function just to initialize absl logging
    pass


if __name__ == "__main__":
    print(Back.BLACK)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
        loop.run_forever()
    except ConnectionRefusedError:
        sys.stderr.write(f"{Fore.RED}Connection refused. Is the server running?{Fore.RESET}")
        sys.exit(1)
    except OSError as e:
        sys.stderr.write(f"{Fore.RED}Table manager teminated{Fore.RESET}\n")
        handle_exception(e)
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    except ValueError as e:
        sys.stderr.write(f"{Fore.RED}Error in configuration or communication - typical the models do not match the configuration.{Fore.RESET}\n")
        handle_exception(e)
        sys.exit(1)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        print(Style.RESET_ALL)
