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

import asyncio
import compare
import scoring

import pprint
import time
import datetime
import json
import asyncio
import uuid
import shelve
import re
import argparse
import numpy as np

import human
import botbidder
import botopeninglead
import botcardplayer
import conf

from sample import Sample
from bidding import bidding
from deck52 import board_dealer_vuln, decode_card, card52to32, get_trick_winner_i, random_deal
from bidding.binary import DealData
from objects import CardResp, Card, BidResp
from claim import Claimer
from pbn2ben import load
from util import calculate_seed, get_play_status, get_singleton, get_possible_cards
from colorama import Fore, Back, Style, init
import gc
import psutil
from nn.opponents import Opponents

import faulthandler
faulthandler.enable()

version = '0.8.7.1'

init()

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

def get_execution_path():
    # Get the directory where the program is started from either PyInstaller executable or the script
    return os.getcwd()


def random_deal_board(number=None):
    deal_str = random_deal()
    if number == None:
        number = np.random.randint(1, 17)
    auction_str = board_dealer_vuln(number)

    return deal_str, auction_str


class AsyncBotBid(botbidder.BotBid):
    async def async_bid(self, auction, alert=None):
        return self.bid(auction)

class AsyncBotLead(botopeninglead.BotLead):
    async def async_opening_lead(self, auction, aceking):
        return self.find_opening_lead(auction, aceking)

class AsyncCardPlayer(botcardplayer.CardPlayer):
    async def async_play_card(self, trick_i, leader_i, current_trick52, tricks52, players_states, worlds, bidding_scores, quality, probability_of_occurence, shown_out_suits, play_status, lead_scores, play_scores, logical_play_scores, discard_scores, features):
        return self.play_card(trick_i, leader_i, current_trick52, tricks52, players_states, worlds, bidding_scores, quality, probability_of_occurence, shown_out_suits, play_status, lead_scores, play_scores, logical_play_scores, discard_scores, features)
    
    
class Driver:

    def __init__(self, models, factory, sampler, seed, ddsolver, verbose):
        self.models = models
        self.sampler = sampler
        self.factory = factory
        self.confirmer = factory.create_confirmer()
        self.channel = factory.create_channel()

        if seed is not None:
            print(f"Setting seed = {seed}")
            np.random.seed(seed)
            tf.random.set_seed(seed)

        #Default is a Human South
        self.human = [False, False, True, False]
        self.human_declare = False
        self.rotate = False
        self.name = models.name
        self.ns = models.ns
        self.ew = models.ew
        self.verbose = verbose
        self.play_only = False
        self.claim = models.claim
        self.claimed = None
        self.claimedbydeclarer = None
        self.conceed = None
        self.decl_i = None
        self.strain_i = None
        self.facit_score = None
        self.facit_total = 0
        self.actual_score = 0
        self.card_play = None
        self.dds = ddsolver


    def set_deal(self, board_number, deal_str, auction_str, play_only = None, bidding_only="False"):
        self.play_only = play_only
        self.bidding_only = bidding_only
        self.board_number = board_number
        self.deal_str = deal_str
        self.hands = deal_str.split()
        self.deal_data = DealData.from_deal_auction_string(self.deal_str, auction_str, "", self.ns, self.ew,  self.models.n_cards_bidding)

        if bidding_only != "False":
            auction_part = auction_str.split(' ')
            if play_only == None and len(auction_part) > 2: play_only = True
            self.fixed_auction = auction_part[2:]
            if self.verbose:
                print("Fixed auction: ", self.fixed_auction)
        else:
            auction_part = auction_str.split(' ')
            if play_only == None and len(auction_part) > 2: play_only = True
            if play_only:
                self.auction = self.deal_data.auction
                self.play_only = play_only
            self.fixed_auction = None
                
        if self.rotate:
            self.dealer_i = (self.deal_data.dealer + 1) % 4
            self.vuln_ns = self.deal_data.vuln_ew
            self.vuln_ew = self.deal_data.vuln_ns
            self.hands.insert(0, self.hands.pop())
            self.deal_str = " ".join(self.hands)
            print("Rotated deal: "+self.deal_str)
        else:
            self.dealer_i = self.deal_data.dealer
            self.vuln_ns = self.deal_data.vuln_ns
            self.vuln_ew = self.deal_data.vuln_ew
        self.trick_winners = []
        self.tricks_taken = 0
        self.parscore = 0

        # Now you can use hash_integer as a seed
        hash_integer = calculate_seed(deal_str)
        if self.verbose:
            print("Setting seed (Full deal) =",hash_integer)
        np.random.seed(hash_integer)


    async def run(self, t_start):
        result_list = self.hands.copy()

        # If human involved hide the unseen hands
        if np.any(np.array(self.human)):
            for i in range(len(self.human)):
                if not self.human[i]:
                    result_list[i] = ""

        await self.channel.send(json.dumps({
            'message': 'deal_start',
            'dealer': self.dealer_i,
            'vuln': [self.vuln_ns, self.vuln_ew],
            'hand': result_list,
            #'name': self.name,
            'board_no' : self.board_number
        }))

        self.bid_responses = []
        self.card_responses = []

        if self.play_only:
            for bid in self.auction:
                if bidding.BID2ID[bid] > 1:
                    self.bid_responses.append(BidResp(bid=bid, candidates=[], samples=[], shape=-1, hcp=-1, who="PlayOnly", quality=None, alert=None, explanation=None))
        else:
            self.auction = await self.bidding(self.bidding_only, self.fixed_auction)

        self.contract = bidding.get_contract(self.auction)
        self.strain_i = bidding.get_strain_i(self.contract)
        self.decl_i = bidding.get_decl_i(self.contract)

        if self.bidding_only != "False":
            print("Bidding only,  saving result and going to next board")
            return

        if self.contract is None:
            await self.channel.send(json.dumps({
                'message': 'deal_end',
                'pbn': self.deal_str,
                'dict': self.to_dict() 
            }))
            print('{1} Bidding took {0:0.1f} seconds.'.format(time.time() - t_start, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            return

        await self.channel.send(json.dumps({
            'message': 'auction_end',
            'declarer': self.decl_i,
            'auction': self.auction,
            'strain': self.strain_i
        }))

        print('{1} Bidding took {0:0.1f} seconds.'.format(time.time() - t_start, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        features = {}
        aceking = {}
        if self.models.use_bba_to_count_aces:
            if self.bot is not None and self.bot.bbabot is not None:
                aceking = self.bot.bbabot.find_aces(self.auction)
                explanation, _, preempted = self.bot.bbabot.explain_auction(self.auction)
                features["Explanation"] = explanation
                features["preempted"] = preempted

        features["aceking"] = aceking
        
        if self.verbose:
            if self.bot is not None and self.bot.bbabot is not None:
                self.bot.bbabot.get_sample(self.auction)

        print("trick 1")

        opening_lead52 = (await self.opening_lead(self.auction, aceking))

        if str(opening_lead52.card).startswith("Conceed"):

            self.conceed = True
            self.claimed = 0
            self.claimedbydeclarer = False
            await self.channel.send(json.dumps({
                'message': 'deal_end',
                'pbn': self.deal_str,
                'dict': self.to_dict() 
            }))
            return
        
        self.card_responses.append(opening_lead52)

        opening_lead52 = opening_lead52.card.code()
        await self.channel.send(json.dumps({
            'message': 'card_played',
            'player': (self.decl_i + 1) % 4,
            'card': decode_card(opening_lead52)
        }))

        print('{1} Opening lead after {0:0.1f} seconds.'.format(time.time() - t_start, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        # If human is dummy display declarers hand unless declarer also is human
        if self.human[(self.decl_i + 2) % 4] and not self.human[self.decl_i]:
            hand = self.deal_str.split()[self.decl_i]
            await self.channel.send(json.dumps({
                'message': 'show_dummy',
                'player': self.decl_i,
                'dummy': hand
            }))
        else:
            # dummys hand
            hand = self.deal_str.split()[(self.decl_i + 2) % 4]
            await self.channel.send(json.dumps({
                'message': 'show_dummy',
                'player': (self.decl_i + 2) % 4,
                'dummy': hand
            }))

        if self.verbose: 
            for card_resp in self.card_responses:
                print(f"{Fore.LIGHTCYAN_EX}")
                pprint.pprint(card_resp.to_dict(), width=200)
                print(f"{Fore.RESET}")

        
        self.card_play = await self.play(self.contract, self.strain_i, self.decl_i, self.auction, opening_lead52, features)

        await self.channel.send(json.dumps({
            'message': 'deal_end',
            'pbn': self.deal_str,
            'dict': self.to_dict()
        }))

    def pbn_header(self, event):
        pbn_str = ""
        pbn_str += '% PBN 2.1\n'
        pbn_str += '% EXPORT\n'
        pbn_str += '%BCOptions\n'
        pbn_str += '%PipColors #0000ff,#ff0000,#ffc000,#008000\n'
        pbn_str += '%PipFont "Symbol","Symbol",2,0xAA,0xA9,0xA8,0xA7\n'
        pbn_str += '%Font:FixedPitch "Courier New",14,700,0\n'
        pbn_str += '%Margins 1000,0800,500,0800\n\n'
        pbn_str += f'[Event "##{event}"]\n'
        pbn_str += f'[Site "##BEN {version}"]\n'
        date = datetime.datetime.now().date().isoformat().replace('-', '.')
        pbn_str += f'[Date "##{date}"]\n'
        return pbn_str

    def asPBN(self):
        dealer = "NESW"[self.dealer_i]
        pbn_str = ""
        pbn_str += '[BCFlags "801f"]\n'
        pbn_str += f'[Board "{self.board_number}"]\n'
        if self.bidding_only == "NS":
            pbn_str += '[West "Facit"]\n'
            pbn_str += '[East "Facit"]\n'
        else:
            pbn_str += '[West "BEN"]\n'
            pbn_str += '[East "BEN"]\n'
        pbn_str += '[North "BEN"]\n'
        pbn_str += '[South "BEN"]\n'
        pbn_str += f'[Dealer "{dealer}"]\n'
        if self.vuln_ns and self.vuln_ew:
            pbn_str += '[Vulnerable "All"]\n'
        if self.vuln_ns and not self.vuln_ew:
            pbn_str += '[Vulnerable "NS"]\n'
        if not self.vuln_ns and self.vuln_ew:
            pbn_str += '[Vulnerable "EW"]\n'
        if not self.vuln_ns and not self.vuln_ew:
            pbn_str += '[Vulnerable "None"]\n'
        pbn_str += f'[Deal "N:{self.deal_str}"]\n'
        if self.contract is None:
            pbn_str += '[Contract ""]\n'
            pbn_str += f'[Declarer ""]\n'
        else:
            declarer = self.contract[-1]
            pbn_str += f'[Declarer "{declarer}"]\n'
            # Remove declarer from contract
            pbn_str += f'[Contract "{self.contract[:-1]}"]\n'

        if self.bidding_only == "NS":
            pbn_str += f'[Scoring "Facit"]\n'
        else:
            pbn_str += f'[Result "{self.tricks_taken}"]\n'
            if self.models.matchpoint:
                pbn_str += '[Scoring "MP"]\n'
            else:
                pbn_str += '[Scoring "IMP"]\n'
            if self.contract is not None:
                if (self.contract[-1] == "N" or self.contract[-1] =="S"):
                    pbn_str += f'[Score "NS {scoring.score(self.contract, self.vuln_ns, self.tricks_taken)}"]\n'
                else:
                    pbn_str += f'[Score "EW {scoring.score(self.contract, self.vuln_ew, self.tricks_taken)}"]\n'

        pbn_str += f'[ParScore "{self.parscore}"]\n'
        pbn_str += f'[Auction "{dealer}"]\n'
        auctionlines = ((len(self.bid_responses) + (self.dealer_i + 1) % 4) + 3)// 4
        alerts = 1
        notes = []
        for i, b in enumerate(self.bid_responses, start=1):
            pbn_str += b.bid
            if b.alert or (b.explanation != None and b.explanation != ""):
                pbn_str += f" ={alerts}="
                note = f'[Note "{alerts}:'
                note += '*' if b.alert else ''
                note += f' {b.explanation}' if b.explanation != None else ''
                note += '"]'
                notes.append(note)
                alerts += 1
            if i % 4 == 0:
                pbn_str += "\n"
            else:
                pbn_str += " "
        # Add an additional line break if the total number of bids is not divisible by 4
        if i % 4 != 0:
            pbn_str += "\n"
            
        if len(notes) > 0:
            pbn_str +=  "\n".join(notes) + "\n"
        auctionlines += len(notes)
        if self.contract is not None and self.card_play:
            declarer_i = "NESW".index(declarer)
            leader = "NESW"[(declarer_i + 1) % 4]            
            pbn_str += f'[Play "{leader}"]\n'
            for i in range(13):
                for j in range(4):
                    pbn_str += Card.from_code(self.card_play[j][i]).symbol() + " "
                pbn_str += "\n"
        else:
            pbn_str += f'[Play ""]\n'

        if self.bidding_only == "NS":
            pbn_str += '[Hidden "EW"]\n'
            pbn_str += '{'
            for i in range(max(0,(9-auctionlines))):
                pbn_str += '\\n'
            pbn_str += 'Score Table:\n\n'
            elements = self.facit_score[self.board_number - 1]
            # Add a newline after every second element
            str = ""
            for i in range(len(elements) // 2):
                contract = elements[i*2]
                contract_length = len(contract)
                str += ' '
                contract = contract[0] + contract[1:].replace("S", "\\s").replace("H", "\\h").replace("D", "\\d").replace("C", "\\c")            
                str += contract + ' ' * (5 - contract_length)
                score_str = elements[i*2 +1]
                str += ' ' * (5 - len(score_str))
                str += score_str
                str += '\\n'

            for i in range(12 - (len(elements) // 2)):
                str += '\\n'

            # Replace the suit characters
            pbn_str += str + '\n\n'
            pbn_str += 'Facit Score:   ' + f"{self.actual_score:>3}" + '\\n'            
            pbn_str += 'Running Score: ' + f"{self.facit_total:>3}"             
            pbn_str += '\n}\n'
        else:
            pbn_str += '[HomeTeam ""]\n'
            pbn_str += '[VisitTeam ""]\n'
            pbn_str += '[ScoreIMP ""]\n'
            pbn_str += '[Room ""]\n'
        pbn_str += '\n'
        return pbn_str
    
    def to_dict(self):
        result = {
            'timestamp': time.time(),
            'dealer': self.dealer_i,
            'vuln_ns': self.vuln_ns,
            'vuln_ew': self.vuln_ew,
            'hands': self.deal_str,
            'bids': [b.to_dict() for b in self.bid_responses],
            'contract': self.contract,
            'play': [c.to_dict() for c in self.card_responses],
            'trick_winners': self.trick_winners,
            'board_number' : self.board_number,
            'player': self.name,
            'rotated': self.rotate,
            'play_only': self.play_only,
            'bidding_only': self.bidding_only,
            'human': self.human,
            'opponents': "BEN",
            'partner': "BEN",
            'model': self.models.name,
            'version': f'{version}'
        }
        if self.decl_i is not None:
            result['declarer'] = self.decl_i
        if self.strain_i is not None:
            result['strain'] = self.strain_i
        if self.claimed is not None:
            result['claimed'] = self.claimed
        if self.claimedbydeclarer is not None:
            result['claimedbydeclarer'] = self.claimedbydeclarer
        if self.conceed is not None:
            result['conceed'] = self.conceed
        return result

# trick_i : 
# 	the number of the trick (0-12)
# player_i : 
# 	the number of the player (0-3)
# card_players : 
# 	a list of bots.CardPlayer(left,dummy,right,declarer). Updated trough time
# 	init :
#         card_players = [
#                     bots.CardPlayer(self.models.player_models, 0, lefty_hand, dummy_hand, contract, is_decl_vuln),
#                     bots.CardPlayer(self.models.player_models, 1, dummy_hand, decl_hand, contract, is_decl_vuln),
#                     bots.CardPlayer(self.models.player_models, 2, righty_hand, dummy_hand, contract, is_decl_vuln),
#                     bots.CardPlayer(self.models.player_models, 3, decl_hand, dummy_hand, contract, is_decl_vuln)
#                 ] 
#         Note : for some reason the declarer hand is set up as the public hand for the dummy.
#         self.x_play contains the cards played as 13 list of binaries repr, one for each card played
#         self.hand52 contains the hand of the player as a list of binaries (except for dummy)
#         self.public52 contains the dummy hand as a list of binaries (except for dummy)
#     each card played :
#         card_player[player_i].set_own_card_played52(card52)
#         if player_i==1 : (dummy)
#             card_player.set_public_card_played52(card52) for all cards players except dummy
#         if player_i==3 : (declarer)
#             dummy.set_public_card_played52(card52) # Why this ? Because of the curious point mentionned before
# shown_out_suit :
#     A list of set, each set containing the shown_out suit as a number(spades=0,clubs=3)
#     Updated for each card play     
# players_cards played:
# 	a list of 4 list(left,dummy,right,declarer). 32 repr
# 	Updated only when the trick is finished
# current_trick : 
# 	the current trick, is the order the card have been played. 32 repr
# self.padded_auction : 
# 	The auction, with the pad at the begining
# card_players[player_i].hand.reshape((-1, 32)) :
# 	The 13 cards of a player, reshaped into a list. Not updated trough tricks. 32 repr 
# self.vuln :
# 	The vuls, as a list of two bools

# After each trick is done :
#     for each card played, init the x_play slice of the next trick. Pain in the ass

    async def play(self, contract, strain_i, decl_i, auction, opening_lead52, features):
        
        level = int(contract[0])
        is_decl_vuln = [self.vuln_ns, self.vuln_ew, self.vuln_ns, self.vuln_ew][decl_i]

        lefty_hand = self.hands[(decl_i + 1) % 4]
        dummy_hand = self.hands[(decl_i + 2) % 4]
        righty_hand = self.hands[(decl_i + 3) % 4]
        decl_hand = self.hands[decl_i]

        pimc = [None, None, None, None]

        if self.models.pimc_use_declaring: 
            # No PIMC for dummy, we want declarer to play both hands
            from pimc.PIMC import BGADLL
            declarer = BGADLL(self.models, dummy_hand, decl_hand, contract, is_decl_vuln, self.sampler, self.verbose)
            pimc[1] = declarer
            pimc[3] = declarer
            if self.verbose:
                print("PIMC",dummy_hand, decl_hand, contract)
        else:
            pimc[1] = None
            pimc[3] = None
        if self.models.pimc_use_defending:
            from pimc.PIMCDef import BGADefDLL
            pimc[0] = BGADefDLL(self.models, dummy_hand, lefty_hand, contract, is_decl_vuln, 0, self.sampler, self.verbose)
            pimc[2] = BGADefDLL(self.models, dummy_hand, righty_hand, contract, is_decl_vuln, 2, self.sampler, self.verbose)
            if self.verbose:
                print("PIMC",dummy_hand, lefty_hand, righty_hand, contract)
        else:
            pimc[0] = None
            pimc[2] = None

        card_players = [
            AsyncCardPlayer(self.models, 0, lefty_hand, dummy_hand, contract, is_decl_vuln, self.sampler, pimc[0], self.dds, self.verbose),
            AsyncCardPlayer(self.models, 1, dummy_hand, decl_hand, contract, is_decl_vuln, self.sampler, pimc[1], self.dds, self.verbose),
            AsyncCardPlayer(self.models, 2, righty_hand, dummy_hand, contract, is_decl_vuln, self.sampler, pimc[2], self.dds, self.verbose),
            AsyncCardPlayer(self.models, 3, decl_hand, dummy_hand, contract, is_decl_vuln, self.sampler, pimc[3], self.dds, self.verbose)
        ]

        # check if user is playing and update card players accordingly
        # the card players are allways positioned relative to declarer (lefty = 0, dummy = 1 ...)
        for i in range(4): 
            if self.human[i]:
                # We are declarer or human declare and dummy
                if decl_i == i or self.human_declare and decl_i == (i + 2) % 4:
                    card_players[3] = self.factory.create_human_cardplayer(self.models, 3, decl_hand, dummy_hand, contract, is_decl_vuln)
                    card_players[1] = self.factory.create_human_cardplayer(self.models, 1, dummy_hand, decl_hand, contract, is_decl_vuln)
                    
                # We are lefty
                if i == (decl_i + 1) % 4:
                    card_players[0] = self.factory.create_human_cardplayer(self.models, 0, lefty_hand, dummy_hand, contract, is_decl_vuln)
                # We are righty
                if i == (decl_i + 3) % 4:
                    card_players[2] = self.factory.create_human_cardplayer(self.models, 2, righty_hand, dummy_hand, contract, is_decl_vuln)

        claimer = Claimer(self.verbose, self.dds)

        player_cards_played = [[] for _ in range(4)]
        player_cards_played52 = [[] for _ in range(4)]
        shown_out_suits = [set() for _ in range(4)]
        discards = [set() for _ in range(4)]
        card_play = [[] for _ in range(4)]

        leader_i = 0

        tricks = []
        tricks52 = []
        trick_won_by = []

        opening_lead = card52to32(opening_lead52)

        current_trick = [opening_lead]
        current_trick52 = [opening_lead52]
        card_play[0].append(opening_lead52)

        card_players[0].hand52[opening_lead52] -= 1
        for trick_i in range(12):
            if trick_i != 0:
                print(f"trick {trick_i+1} lead:{leader_i}")

            for player_i in map(lambda x: x % 4, range(leader_i, leader_i + 4)):
                if self.verbose:
                    print('player {}'.format(player_i))
                
                if trick_i == 0 and player_i == 0:
                    for i, card_player in enumerate(card_players):
                        card_player.set_real_card_played(opening_lead52, player_i)
                        card_player.set_card_played(trick_i=trick_i, leader_i=leader_i, i=0, card=opening_lead)
                    continue

                card52 = None
                card_resp = None
                play_status = get_play_status(card_players[player_i].hand52,current_trick52, strain_i)
                if self.verbose:
                    print('play status', play_status)

                if isinstance(card_players[player_i], botcardplayer.CardPlayer):
                    if play_status == "Forced" and self.models.autoplaysingleton:
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
                    if isinstance(card_players[player_i], botcardplayer.CardPlayer):
                        played_cards = [card for row in player_cards_played52 for card in row] + current_trick52
                        rollout_states, bidding_scores, c_hcp, c_shp, quality, probability_of_occurence, lead_scores, play_scores, logical_play_scores, discard_scores, worlds = self.sampler.init_rollout_states(trick_i, player_i, card_players, played_cards, player_cards_played, shown_out_suits, discards, features["aceking"], current_trick, opening_lead52, auction, card_players[player_i].hand_str, card_players[player_i].public_hand_str, [self.vuln_ns, self.vuln_ew], self.models, card_players[player_i].get_random_generator())
                        assert rollout_states[0].shape[0] > 0, "No samples for DDSolver"
                        card_players[player_i].check_pimc_constraints(trick_i, rollout_states, quality)
                    else: 
                        rollout_states = []
                        bidding_scores = []
                        lead_scores = []
                        play_scores = []
                        c_hcp = -1
                        c_shp = -1
                        quality = 1
                        probability_of_occurence = []
                        logical_play_scores = []
                        discard_scores = []
                        worlds = []
                        
                    await asyncio.sleep(0.01)

                    while card_resp is None:
                        card_resp =  await card_players[player_i].async_play_card(trick_i, leader_i, current_trick52, tricks52, rollout_states, worlds, bidding_scores, quality, probability_of_occurence, shown_out_suits, play_status, lead_scores, play_scores, logical_play_scores, discard_scores, features)

                        if (str(card_resp.card).startswith("Conceed")) :
                                self.claimedbydeclarer = (player_i == 3) or (player_i == 1)
                                self.claimed = 0
                                self.conceed = True
                                self.trick_winners = trick_won_by
                                print(f"Contract: {self.contract} Accepted conceed½")
                                return

                        if (str(card_resp.card).startswith("Claim")) :
                            tricks_claimed = int(re.search(r'\d+', card_resp.card).group()) if re.search(r'\d+', card_resp.card) else None
                            
                            self.canclaim = claimer.claim(
                                strain_i=strain_i,
                                player_i=player_i,
                                hands52=[card_player.hand52 for card_player in card_players],
                                n_samples=50
                            )
                            if (tricks_claimed <= self.canclaim):
                                # player_i is relative to declarer
                                print(f"Claimed {tricks_claimed} can claim {self.canclaim} {player_i} {decl_i}")
                                self.claimedbydeclarer = (player_i == 3) or (player_i == 1)
                                self.claimed = tricks_claimed

                                # Trick winners until claim is saved
                                self.trick_winners = trick_won_by
                                # Print contract and result
                                if self.claimedbydeclarer:
                                    print(f"Contract: {self.contract} Accepted declarers claim of {tricks_claimed} tricks")
                                else:
                                    print(f"Contract: {self.contract} Accepted opponents claim of {tricks_claimed} tricks")
                                return
                            else:
                                if self.claimedbydeclarer:
                                    print(f"Declarer claimed {tricks_claimed} tricks - rejected {self.canclaim}")
                                else:
                                    print(f"Opponents claimed {tricks_claimed} tricks - rejected {self.canclaim}")
                                await self.channel.send(json.dumps({
                                    'message': 'claim_rejected',
                                }))
                                card_resp = None


                    card_resp.hcp = c_hcp
                    card_resp.shape = c_shp
                    if self.verbose:
                        print(f"{Fore.LIGHTCYAN_EX}")
                        pprint.pprint(card_resp.to_dict(), width=200)
                        print(f"{Fore.RESET}")
                    
                    await asyncio.sleep(0.01)

                self.card_responses.append(card_resp)

                card52 = card_resp.card.code()

                await self.channel.send(json.dumps({
                    'message': 'card_played',
                    'player': ((decl_i + 1) % 4 + player_i) % 4,
                    'card': card_resp.card.symbol()
                }))

                card32 = card52to32(card52)

                for card_player in card_players:
                    card_player.set_real_card_played(card52, player_i)
                    card_player.set_card_played(trick_i=trick_i, leader_i=leader_i, i=player_i, card=card32)

                current_trick.append(card32)

                current_trick52.append(card52)

                card_play[player_i].append(card52)

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

            for i, card_player in enumerate(card_players):
                assert np.min(card_player.hand52) == 0
                assert np.min(card_player.public52) == 0
                assert np.sum(card_player.hand52) == 13 - trick_i - 1
                assert np.sum(card_player.public52) == 13 - trick_i - 1

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
                if self.verbose:
                    print(f"Initialize last trick {card_player.player_i}")
                for i, card32 in enumerate(current_trick):
                    card_player.x_play[:, trick_i + 1, 64 + i * 32 + card32] = 1
                    
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

            if self.verbose:
                print('trick52 {} cards={}. won by {}'.format(trick_i+1, list(map(decode_card, current_trick52)), trick_winner))

            # update cards shown
            for i, card32 in enumerate(current_trick):
                player_cards_played[(leader_i + i) % 4].append(card32)           
            for i, card52 in enumerate(current_trick52):
                player_cards_played52[(leader_i + i) % 4].append(card52)

            leader_i = trick_winner
            current_trick = []
            current_trick52 = []

            if self.verbose:
                print(f"Human {self.human}")
            if np.any(np.array(self.human)):
                key = await self.confirmer.confirm()
                if key == 'q':
                    print(self.deal_str)
                    return
            else:
                await self.confirmer.confirm()

        # play last trick
        print("trick 13")
        for player_i in map(lambda x: x % 4, range(leader_i, leader_i + 4)):
            
            if not isinstance(card_players[player_i], botcardplayer.CardPlayer):
                await card_players[player_i].get_card_input()
                who = "Human"
            else:
                who = "NN"

            card52 = int(np.nonzero(card_players[player_i].hand52)[0][0])
            card32 = card52to32(card52)

            card_resp = CardResp(card=Card.from_code(card52), candidates=[], samples=[], shape=-1, hcp=-1, quality=None, who=who, claim = -1)

            await self.channel.send(json.dumps({
                'message': 'card_played',
                'player': ((decl_i + 1) % 4 + player_i) % 4,
                'card': card_resp.card.symbol()
            }))

            self.card_responses.append(card_resp)
            
            current_trick.append(card32)
            current_trick52.append(card52)
            card_play[player_i].append(card52)
            # Just updating to create better validation
            for card_player in card_players:
                card_player.set_real_card_played(card52, player_i)

        await self.confirmer.confirm()

        tricks.append(current_trick)
        tricks52.append(current_trick52)
        
        trick_winner = (leader_i + get_trick_winner_i(current_trick52, (strain_i - 1) % 5)) % 4
        if trick_winner % 2 == 0:
            card_players[0].n_tricks_taken += 1
            card_players[2].n_tricks_taken += 1
        else:
            card_players[1].n_tricks_taken += 1
            card_players[3].n_tricks_taken += 1


        trick_won_by.append(trick_winner)
        if self.verbose:
            # Decode each element of tricks52
            decoded_tricks52 = [[decode_card(item) for item in inner] for inner in tricks52]
            pprint.pprint(list(zip(decoded_tricks52, trick_won_by)))

        self.trick_winners = trick_won_by
        self.tricks_taken = card_players[3].n_tricks_taken

        # Print contract and result
        print("Contract: ",self.contract, card_players[3].n_tricks_taken, "tricks")

        return card_play

    
    async def opening_lead(self, auction, aceking):

        contract = bidding.get_contract(auction)
        decl_i = bidding.get_decl_i(contract)

        hands_str = self.deal_str.split()

        await asyncio.sleep(0.01)

        if self.human[(decl_i + 1) % 4]:
            card_resp = await self.factory.create_human_leader().async_lead()
            if self.verbose:
                print(f"{Fore.LIGHTCYAN_EX}{card_resp.card} selected by human{Fore.RESET}")
        else:
            bot_lead = AsyncBotLead(
                [self.vuln_ns, self.vuln_ew], 
                hands_str[(decl_i + 1) % 4], 
                self.models,
                self.sampler,
                (decl_i + 1) % 4,
                self.dealer_i,
                self.dds,
                self.verbose
            )
            card_resp = await bot_lead.async_opening_lead(auction, aceking)

        await asyncio.sleep(0.01)

        return card_resp

    async def bidding(self, bidding_only, bidding_only_auction):
        hands_str = self.deal_str.split()
        
        vuln = [self.vuln_ns, self.vuln_ew]

        players = []
        hint_bots = [None, None, None, None]

        for i, level in enumerate(self.human):
            if level == 1:
                hint_bots[i] = AsyncBotBid(vuln, hands_str[i], self.models, self.sampler, i, self.dealer_i, self.dds, False, self.verbose)
                players.append(self.factory.create_human_bidder(vuln, hands_str[i], self.name, hint_bots[i]))
                self.bot = None
            else:
                self.bot = AsyncBotBid(vuln, hands_str[i], self.models, self.sampler, i, self.dealer_i, self.dds, False, self.verbose)
                players.append(self.bot)

        if self.models.use_bba or self.models.use_bba_to_count_aces or self.models.consult_bba or self.models.use_bba_rollout:
            if self.verbose:
                print(f"Using BBA CC's {self.models.bba_our_cc} and {self.models.bba_their_cc}")
                print("Our conventions")
                print("\n".join([convention for convention, selected in players[0].bbabot.our_conventions.items() if selected]))
                print("Their conventions")
                print("\n".join([convention for convention, selected in players[0].bbabot.their_conventions.items() if selected]))
        auction = ['PAD_START'] * self.dealer_i

        player_i = self.dealer_i
        alert = False
        bid_no = 0

        while not bidding.auction_over(auction):
            if self.bidding_only == "NS" and (player_i == 1 or player_i == 3):
                if bidding_only_auction[0] != '' and len(bidding_only_auction) > bid_no:
                    if bidding.can_bid(bidding_only_auction[bid_no].replace("P","PASS"), auction):
                        bid_resp = BidResp(bid=bidding_only_auction[bid_no].replace("P","PASS"), candidates=[], samples=[], shape=-1, hcp=-1, who=self.name, quality=None, alert=alert, explanation=None)
                    else:
                        bid_resp = BidResp(bid="PASS", candidates=[], samples=[], shape=-1, hcp=-1, who=self.name, quality=None, alert=alert, explanation=None)
                else:
                    bid_resp = BidResp(bid="PASS", candidates=[], samples=[], shape=-1, hcp=-1, who=self.name, quality=None, alert=alert, explanation=None)
            else:
                bid_resp = await players[player_i].async_bid(auction, alert)
            if bid_resp.bid == "Alert": 
                alert = not alert
                await self.channel.send(json.dumps({
                    'message': 'alert',
                    'alert': str(alert)
                }))
                await asyncio.sleep(0.1)
            else:
                if bid_resp.bid == "Hint":
                    bid_resp = await hint_bots[player_i].async_bid(auction)
                    await self.channel.send(json.dumps({
                        'message': 'hint',
                        'bids': bid_resp.to_dict()
                    }))

                    await asyncio.sleep(0.1)
                else :
                    self.bid_responses.append(bid_resp)
                    auction.append(bid_resp.bid)
                    bid_no += 1

                    if self.bidding_only != "NS":
                        await self.channel.send(json.dumps({
                            'message': 'bid_made',
                            'auction': auction,
                            'alert': str(alert or bid_resp.alert),
                            'explanation': bid_resp.explanation
                        }))

                    player_i = (player_i + 1) % 4
                    # give time to client to redraw
                    await asyncio.sleep(0.1)
                    alert = False
                
        return auction

def random_deal_source():
    while True:
        yield random_deal_board()

# Custom function to convert string to boolean
def str_to_bool(value):
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    raise ValueError("Invalid boolean value")

def log_memory_usage():
    # Get system memory info
    virtual_memory = psutil.virtual_memory()
    available_memory = virtual_memory.available / (1024 ** 2)  # Convert bytes to MB
    print(f"Available memory: {available_memory:.2f} MB")


async def main():
    random = True
    #For some strange reason parameters parsed to the handler must be an array
    boardno = None
    board_no = []
    board_no.append(0) 

    # Get the path to the config file
    config_path = get_execution_path()

    parser = argparse.ArgumentParser(description="BEN Game server")
    parser.add_argument("--boards", default="", help="Filename for configuration")
    parser.add_argument("--auto", type=str_to_bool, default=False, help="Continue without user confirmation. If a file is provided it will stop at end of file")
    parser.add_argument("--boardno", default=0, type=int, help="Board number to start from")
    parser.add_argument("--config", default=f"{config_path}/config/default.conf", help="Filename for configuration")
    parser.add_argument("--opponent", default="", help="Filename for configuration pf opponents")
    parser.add_argument("--playonly", type=str_to_bool, default=False, help="Just play, no bidding")
    parser.add_argument("--biddingonly", default="False", help="Just bidding, no play, can be True, NS or EW")
    parser.add_argument("--outputpbn", default="", help="Save each board to this PBN file")
    parser.add_argument("--paronly", default=0, type=int, help="only record deals with this IMP difference from par")
    parser.add_argument("--facit", default=False, type=str_to_bool, help="Calcualte score for the bidding from facit")
    parser.add_argument("--matchpoint", type=str_to_bool, default=None, help="Playing match point")
    parser.add_argument("--verbose", type=str_to_bool, default=False, help="Output samples and other information during play")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for random")
    parser.add_argument("--db", default="paronly", help="Db for board records")

    args = parser.parse_args()

    configfile = args.config
    opponentfile = args.opponent
    verbose = args.verbose
    auto = args.auto
    playonly = args.playonly
    biddingonly = args.biddingonly
    seed = args.seed
    if seed == -1:
        seed = np.random.SeedSequence().generate_state(1)[0]
    outputpbn = args.outputpbn
    paronly = args.paronly
    matchpoint = args.matchpoint
    facit = args.facit
    facit_score = None
    boards = []
    event = ""
    DB_NAME =  args.db

    np.set_printoptions(precision=2, suppress=True, linewidth=200)

    print(f"{Fore.CYAN}{datetime.datetime.now():%Y-%m-%d %H:%M:%S} game.py - Version {version}")
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

    if facit:
            print("Playing Bidding contest")
    else:
        if matchpoint is not None:
            models.matchpoint = matchpoint
        if models.matchpoint:
            print("Matchpoint mode on")
        else:
            print("Playing IMPS mode")

    if models.use_bba or models.use_bba_to_count_aces or models.consult_bba or models.use_bba_rollout:
        from bba.BBA import BBABotBid
        bot = BBABotBid(None, None ,None, None, None, None, None, None)
        print(f"BBA enabled. Version {bot.version()}")    

    if models.use_suitc:
        from suitc.SuitC import SuitCLib
        suitc = SuitCLib(verbose)
        print(f"SuitC enabled. Version {suitc.version()}")

    if models.pimc_use_declaring or models.pimc_use_defending:
        from pimc.PIMC import BGADLL
        pimc = BGADLL(None, None, None, None, None, None, None)
        from pimc.PIMCDef import BGADefDLL
        pimcdef = BGADefDLL(None, None, None, None, None, None, None, None)
        print(f"PIMC enabled. Version {pimc.version()}")
        print(f"PIMCDef enabled. Version {pimcdef.version()}")

    from ddsolver import ddsolver
    dds = ddsolver.DDSolver()
    print(f"DDSolver enabled. Version {dds.version()}")

    if args.boards:
        filename = args.boards
        file_extension = os.path.splitext(filename)[1].lower()  
        event = os.path.splitext(os.path.basename(filename))[0]
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
                boards, facit_score = load(lines)
                print(f"{len(boards)} boards loaded from file")
            random = False

    if args.boardno:
        print(f"Starting from {args.boardno}")
        board_no[0] = args.boardno - 1
        boardno = args.boardno

    if random:
        event = "Random deals"
        print("Playing random deals or deals from the client")
 

    driver = Driver(models, human.ConsoleFactory(), Sample.from_conf(configuration, verbose), seed, dds, verbose)

    # If the format is score - contract we swap the columns
    if facit and facit_score[0][0].isdigit(): 
        for i in range(len(facit_score)):
            arr = np.array(facit_score[i])
            # Reshape the array to 2 columns (you can infer the number of rows automatically)
            reshaped_arr = arr.reshape(-1, 2)

            # Swap the columns by reversing the order along axis 1
            swapped_arr = reshaped_arr[:, ::-1]

            # Optionally, reshape back to 1D if needed
            facit_score[i] = swapped_arr.flatten()            
    driver.facit_score = facit_score
    while True:
        if random: 
            if boardno:
                np.random.seed(boardno)

            #Just take a random"
            rdeal = random_deal_board(boardno)

            # example of to use a fixed deal
            # rdeal = ('T962.86.K742.643 KJ43.Q9.QT9.T852 .AJT7532.AJ8.QJ7 AQ875.K4.653.AK9', 'N E-W')

            print(f"{Fore.CYAN}Playing Board: {rdeal}{Fore.RESET}")
            driver.set_deal(None, *rdeal, False, bidding_only=biddingonly)
        else:
            rdeal = boards[board_no[0]]['deal']
            auction = boards[board_no[0]]['auction']
            print(f"{Fore.LIGHTBLUE_EX}Board: {board_no[0]+1} {rdeal}{Fore.RESET}")
            #print("auction",auction)
            driver.set_deal(board_no[0] + 1, rdeal, auction, play_only=playonly, bidding_only=biddingonly)
            board_no[0] = (board_no[0] + 1)

        # BEN is handling all 4 hands
        driver.human = [False, False, False, False]
        t_start = time.time()
        await driver.run(t_start)

        score = 0
        imps = 0
        if facit:        
            if driver.facit_score == None:
                print("No score table provided")
            else:
                driver.actual_score = None
                contract = driver.contract[0:-1]
                declarer = driver.contract[-1]
                contract_adjustments = {
                    "1C": ["2C","3C","4C"],
                    "1D": ["2D","3D","4D"],
                    "1H": ["2H","3H"],
                    "1S": ["2S","3S"],
                    "1N": ["2N"],
                    "2C": ["3C","4C"],
                    "2D": ["3D","4D"],
                    "2H": ["3H"],
                    "2S": ["3S"],
                    "3C": ["4C"],
                    "3D": ["4D"],
                    "3N": ["4N", "5N"],
                    "4H": ["5H"],
                    "4S": ["5S"],
                    "4N": ["5N"]
                }
                print( driver.contract, driver.hands[0], driver.hands[2], str(driver.auction).replace("PASS","P"))
                # Loop through facit_score for the current board
                for i in range(len(driver.facit_score[board_no[0] - 1]) // 2):
                    score_contract = driver.facit_score[board_no[0] - 1][i * 2]
                    score_value = int(driver.facit_score[board_no[0] - 1][i * 2 + 1])

                    # Match contract or adjusted contract for declarer
                    if score_contract == contract or score_contract == f"{declarer}{contract}":
                        # Determine color based on score_value
                        if score_value < 4:
                            color = Fore.LIGHTYELLOW_EX  # Use this for an orange-like color
                        elif score_value == 10:
                            color = Fore.LIGHTGREEN_EX
                        else:
                            color = Fore.LIGHTBLUE_EX
                        print(f"{color}Score for {score_contract}: {score_value}{Style.RESET_ALL} ({driver.facit_score[board_no[0]-1]})")
                        driver.actual_score = score_value
                        driver.facit_total += score_value
                        break

                    # Check for adjusted contracts (with or without declarer)
                    if contract in contract_adjustments:
                        adjusted_contracts = contract_adjustments[contract]

                        # Match adjusted contract (with or without declarer)
                        if score_contract in adjusted_contracts or score_contract in [f"{declarer}{adj}" for adj in adjusted_contracts]:
                            # Determine color based on score_value
                            if score_value < 4:
                                color = Fore.LIGHTYELLOW_EX  # Use this for an orange-like color
                            elif score_value == 10:
                                color = Fore.LIGHTGREEN_EX
                            else:
                                color = Fore.LIGHTBLUE_EX
                            print(f"{color}Score for {score_contract}: {score_value}{Style.RESET_ALL} ({driver.facit_score[board_no[0]-1]})")
                            driver.actual_score = score_value
                            driver.facit_total += score_value
                            break
        
                if driver.actual_score is None:
                    print(f"{Fore.RED}No score - Scoring {driver.contract[:-1]} {driver.facit_score[board_no[0]-1]}{Style.RESET_ALL}")
                    driver.actual_score = 0
                else:
                    print(f"{Fore.LIGHTGREEN_EX}Running score: {driver.facit_total}{Style.RESET_ALL}")
        else:
            #print("Calculating PAR")
            par_score = dds.calculatepar(driver.deal_str, [driver.vuln_ns, driver.vuln_ew])
            if driver.contract != None:
                if (driver.contract[-1] == "N" or driver.contract[-1] =="S"):
                    score = scoring.score(driver.contract, driver.vuln_ns, driver.tricks_taken)
                if (driver.contract[-1] == "E" or driver.contract[-1] =="W"):
                    score = -scoring.score(driver.contract, driver.vuln_ew, driver.tricks_taken)

                imps = abs(compare.get_imps(score, par_score))
                print(f"{Fore.LIGHTBLUE_EX}Score: {driver.contract} {str(score)} Par: {str(par_score)} IMP: {str(imps)}{Style.RESET_ALL}")
            driver.parscore = par_score

        if biddingonly == "False":
            if paronly <= imps:
                with shelve.open(f"{config_path}/{DB_NAME}") as db:
                    deal = driver.to_dict()
                    print(f"{datetime.datetime.now():%H:%M:%S} Saving Board: {driver.hands} in {config_path}/{DB_NAME}")
                    db[uuid.uuid4().hex] = deal

        if outputpbn != "":
            if paronly <= imps:
                # Check if the file exists and its size
                if not os.path.exists(outputpbn) or os.path.getsize(outputpbn) == 0 or board_no[0] == 1:
                    # Write header if file is new or empty
                    with open(outputpbn, "w") as file:
                        file.write(driver.pbn_header(event))
                        file.write(driver.asPBN())
                else:
                    # Just append the content if the file already has content
                    with open(outputpbn, "a") as file:
                        file.write(driver.asPBN())

        print(f'{Fore.CYAN}{datetime.datetime.now():%Y-%m-%d %H:%M:%S} Board played in {time.time() - t_start:0.1f} seconds.{Fore.RESET}')  
        if not auto:
            user_input = input("\n Q to quit or any other key for next deal.\n ")
            if user_input.lower() == "q":
                break
        else:
            if args.boards and board_no[0] >= len(boards):
                break
        np.empty(0) 
        gc.collect()

if __name__ == '__main__':
    print(Back.BLACK)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except ValueError as e:
        sys.stderr.write(f"{Fore.RED}Error in configuration or communication - typical the models do not match the configuration.{Fore.RESET}\n")
        handle_exception(e)
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        print(Style.RESET_ALL)