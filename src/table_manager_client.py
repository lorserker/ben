import sys
import os
import logging

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import shelve
# This import is only to help PyInstaller when generating the executables
import tensorflow as tf
import ipaddress
import argparse
import re
import time
import asyncio
import numpy as np
from sample import Sample
import bots
import conf
import datetime
import pprint
from objects import Card, CardResp, BidResp

from deck52 import card52to32, decode_card, get_trick_winner_i, hand_to_str
from bidding import bidding

SEATS = ['North', 'East', 'South', 'West']

class TMClient:

    def __init__(self, name, seat, models, sampler, verbose):
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
            'opponents': self.opponents
        }

    async def run(self, biddingonly):

        self.bid_responses = []
        self.card_responses = []

        self.dealer_i, self.vuln_ns, self.vuln_ew, self.hand_str = await self.receive_deal()

        auction = await self.bidding()

        await asyncio.sleep(0.01)

        self.contract = bidding.get_contract(auction, self.dealer_i, self.models)
        if  self.contract is None:
            return

        # level = int(self.contract[0])
        # strain_i = bidding.get_strain_i(self.contract)
        self.decl_i = bidding.get_decl_i(self.contract)

        print(f'{datetime.datetime.now().strftime("%H:%M:%S")} Bidding ended: {auction}')
        if (self.verbose):
            print(f'{datetime.datetime.now().strftime("%H:%M:%S")} Contract {self.contract}')

        if  biddingonly:
            print(f'{datetime.datetime.now().strftime("%H:%M:%S")} Ready to start new board')
            return

        opening_lead_card = await self.opening_lead(auction)
        opening_lead52 = Card.from_symbol(opening_lead_card).code()

        if self.player_i != (self.decl_i + 2) % 4:
            self.dummy_hand_str = await self.receive_dummy()

        await self.play(auction, opening_lead52)

        
    async def connect(self, host, port):
        try:
            self.reader, self.writer = await asyncio.open_connection(host, port)
            self._is_connected = True
        except ConnectionRefusedError as ex: 
            print(f"Server not responding {str(ex)}")
            self._is_connected = False
            asyncio.get_event_loop().stop()
            return

        print('connected')

        await self.send_message(f'Connecting "{self.name}" as {self.seat} using protocol version 18')

        # Validate response Blue Chip can send: Error Team name mismatch
        await self.receive_line()
        
        await self.send_message(f'{self.seat} ready for teams')


        opponents = await self.receive_line()
        
        # Regular expression pattern to match text in quotes
        pattern = r'"([^"]*?)"'
        # Find all matches in the string
        matches = re.findall(pattern, opponents)

        # Extracted text from the second set of quotes
        if self.seat == "North" or self.seat == "South":
            self.opponents = matches[1]
        else:
            self.opponents = matches[0]

    async def bidding(self):
        vuln = [self.vuln_ns, self.vuln_ew]

        if self.models.use_bba:
            from bba.BBA import BBABotBid            
            bot = BBABotBid(self.models.bba_ns, self.models.bba_ew ,self.player_i,self.hand_str,vuln, self.dealer_i)
        else:
            bot = bots.BotBid(vuln, self.hand_str, self.models, self.sampler, self.player_i, self.dealer_i, self.verbose)

        auction = ['PAD_START'] * self.dealer_i

        player_i = self.dealer_i

        while not bidding.auction_over(auction):
            if player_i == self.player_i:
                # now it's this player's turn to bid
                bid_resp = bot.bid(auction)
                if (self.verbose):
                    pprint.pprint(bid_resp.samples, width=80)
                auction.append(bid_resp.bid)
                self.bid_responses.append(bid_resp)
                await self.send_own_bid(bid_resp.bid)
            else:
                # just wait for the other player's bid
                bid = await self.receive_bid_for(player_i)
                if (player_i + 2) % 4 == self.player_i:
                    bid_resp = BidResp(bid=bid, candidates=[], samples=[], shape=-1, hcp=-1, who = 'Partner', quality=None)
                else:
                    bid_resp = BidResp(bid=bid, candidates=[], samples=[], shape=-1, hcp=-1, who=self.opponents, quality=None)
                self.bid_responses.append(bid_resp)
                auction.append(bid)

            player_i = (player_i + 1) % 4

        return auction

    async def opening_lead(self, auction):

        contract = bidding.get_contract(auction, self.dealer_i, self.models)
        decl_i = bidding.get_decl_i(contract)
        on_lead_i = (decl_i + 1) % 4
        
        if self.player_i == on_lead_i:
            # this player is on lead
            await self.receive_line()

            bot_lead = bots.BotLead(
                [self.vuln_ns, self.vuln_ew], 
                self.hand_str,
                self.models,
                self.sampler,
                on_lead_i,
                self.dealer_i,
                self.verbose
            )
            card_resp = bot_lead.find_opening_lead(auction)
            self.card_responses.append(card_resp)
            card_symbol = card_resp.card.symbol()

            await asyncio.sleep(0.01)

            await self.send_card_played(card_symbol)

            await asyncio.sleep(0.01)

            return card_symbol
        else:
            # just send that we are ready for the opening lead
            return await self.receive_card_play_for(on_lead_i, 0)

    async def play(self, auction, opening_lead52):
        contract = bidding.get_contract(auction, self.dealer_i, self.models)
        
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

        if self.models.pimc_use:
            from pimc.PIMC import BGADLL
            pimc = BGADLL(self.models, dummy_hand_str, decl_hand_str, contract, is_decl_vuln, self.verbose)
            if self.verbose:
                print("PIMC",dummy_hand_str, decl_hand_str, contract)
        else:
            pimc = None


        card_players = [
            bots.CardPlayer(self.models, 0, lefty_hand_str, dummy_hand_str, contract, is_decl_vuln, self.sampler, pimc, self.verbose),
            bots.CardPlayer(self.models, 1, dummy_hand_str, decl_hand_str, contract, is_decl_vuln, self.sampler, pimc, self.verbose),
            bots.CardPlayer(self.models, 2, righty_hand_str, dummy_hand_str, contract, is_decl_vuln, self.sampler, pimc, self.verbose),
            bots.CardPlayer(self.models, 3, decl_hand_str, dummy_hand_str, contract, is_decl_vuln, self.sampler, pimc, self.verbose)
        ]

        player_cards_played = [[] for _ in range(4)]
        player_cards_played52 = [[] for _ in range(4)]
        shown_out_suits = [set() for _ in range(4)]

        leader_i = 0

        tricks = []
        tricks52 = []
        trick_won_by = []

        opening_lead = card52to32(opening_lead52)

        current_trick = [opening_lead]
        current_trick52 = [opening_lead52]

        card_players[0].hand52[opening_lead52] -= 1

        for trick_i in range(12):
            print("{} Playing trick {}".format(datetime.datetime.now().strftime("%H:%M:%S"),trick_i+1))

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
                # it's dummy's turn and this is the declarer
                if (player_i == 1 and cardplayer_i == 3):
                    print('{} declarers turn for dummy'.format(datetime.datetime.now().strftime("%H:%M:%S")))

                if (player_i == cardplayer_i and player_i != 1) or (player_i == 1 and cardplayer_i == 3):
                    rollout_states, bidding_scores, c_hcp, c_shp, good_quality, probability_of_occurence = self.sampler.init_rollout_states(trick_i, player_i, card_players, player_cards_played, shown_out_suits, current_trick, self.dealer_i, auction, card_players[player_i].hand_str, [self.vuln_ns, self.vuln_ew], self.models, card_players[player_i].rng)
                    card_resp = await card_players[player_i].play_card(trick_i, leader_i, current_trick52, rollout_states, bidding_scores, good_quality, probability_of_occurence, shown_out_suits)
                    card_resp.hcp = c_hcp
                    card_resp.shape = c_shp

                    self.card_responses.append(card_resp)

                    if (self.verbose):
                        for idx, candidate in enumerate(card_resp.candidates, start=1):
                            if candidate.expected_tricks_sd:
                                print(f"{candidate.card} Expected Score: {str(int(candidate.expected_score_sd)).ljust(5)} Tricks (SD) {candidate.expected_tricks_sd:.2f} #Insta_score {candidate.insta_score:.3f}")
                            if candidate.expected_tricks_dd:
                                print(f"{candidate.card} Expected Score: {str(int(candidate.expected_score_dd)).ljust(5)} Tricks (DD) {candidate.expected_tricks_dd:.2f} #Insta_score {candidate.insta_score:.3f}")
                        for idx, sample in enumerate(card_resp.samples, start=1):                  
                            print(f"{sample}")

                    card52 = card_resp.card.code()
                    
                    if (player_i == 1 and cardplayer_i == 3):
                        await self.send_card_played_for_dummy(card_resp.card.symbol()) 
                    else:
                        await self.send_card_played(card_resp.card.symbol()) 
                        
                    await asyncio.sleep(0.01)

                else:
                    # another player is on play, we just have to wait for their card
                    card52_symbol = await self.receive_card_play_for(nesw_i, trick_i)
                    card52 = Card.from_symbol(card52_symbol).code()
                
                card = card52to32(card52)
                
                for card_player in card_players:
                    card_player.set_real_card_played(card52, player_i)
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

            for i in [cardplayer_i] + ([1] if cardplayer_i == 3 else []):
                if cardplayer_i == 1:
                    break
                assert np.min(card_players[i].hand52) == 0
                assert np.min(card_players[i].public52) == 0
                assert np.sum(card_players[i].hand52) == 13 - trick_i - 1
                assert np.sum(card_players[i].public52) == 13 - trick_i - 1

            tricks.append(current_trick)
            tricks52.append(current_trick52)

            if self.models.pimc_use:
                # Only declarer and use PIMC
                if isinstance(card_players[3], bots.CardPlayer):
                    card_players[3].pimc.reset_trick()

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
            else:
                card_players[1].n_tricks_taken += 1
                card_players[3].n_tricks_taken += 1
                if self.models.pimc_use:
                    # Only declarer use PIMC
                    if isinstance(card_players[3], bots.CardPlayer):
                        card_players[3].pimc.update_trick_needed()


            print('{} trick52 {} cards={}. won by {}'.format(datetime.datetime.now().strftime("%H:%M:%S"),trick_i+1, list(map(decode_card, current_trick52)), trick_winner))

            # update cards shown
            for i, card in enumerate(current_trick):
                player_cards_played[(leader_i + i) % 4].append(card)
            for i, card in enumerate(current_trick52):
                player_cards_played52[(leader_i + i) % 4].append(card)
            
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
                card52 = np.nonzero(card_players[player_i].hand52)[0][0]
                card52_symbol = Card.from_code(card52).symbol()

                cr = CardResp(
                    card=Card.from_symbol(card52_symbol),
                    candidates=[],
                    samples=[],
                    shape=-1,
                    hcp=-1,
                    quality=None
                )
                self.card_responses.append(cr)

                await asyncio.sleep(0.01)
                
                if player_i == 1 and cardplayer_i == 3:
                    await self.send_card_played_for_dummy(card52_symbol)
                else:    
                    await self.send_card_played(card52_symbol)

                await asyncio.sleep(0.01)
                self.card_responses
            else:
                # someone else is on play. we just have to wait for their card
                card52_symbol = await self.receive_card_play_for(nesw_i, trick_i)
                card52 = Card.from_symbol(card52_symbol).code()

            card = card52to32(card52)

            current_trick.append(card)
            current_trick52.append(card52)

        # update cards shown
        for i, card in enumerate(current_trick):
            player_cards_played[(leader_i + i) % 4].append(card)
        for i, card in enumerate(current_trick52):
            player_cards_played52[(leader_i + i) % 4].append(card)

        tricks.append(current_trick)
        tricks52.append(current_trick52)

        trick_winner = (leader_i + get_trick_winner_i(current_trick52, (strain_i - 1) % 5)) % 4
        trick_won_by.append(trick_winner)

        if (self.verbose):
            pprint.pprint(list(zip(tricks, trick_won_by)))

        self.trick_winners = trick_won_by
        
        # Decode each element of tricks52
        decoded_tricks52 = [[decode_card(item) for item in inner] for inner in tricks52]
        pprint.pprint(list(zip(decoded_tricks52, trick_won_by)))

        concatenated_str = ""

        for i in range(len(player_cards_played52)):
            index = (i + (3-decl_i)) % len(player_cards_played52)
            concatenated_str += hand_to_str(player_cards_played52[index]) + " "

        # Remove the trailing space
        self.deal_str = concatenated_str.strip()


    async def send_card_played(self, card_symbol):
        msg_card = f'{self.seat} plays {card_symbol[::-1]}'
        await self.send_message(msg_card)

    async def send_card_played_for_dummy(self, card_symbol):
        dummy_i = (self.decl_i + 2) % 4
        seat = SEATS[dummy_i]
        msg_card = f'{seat} plays {card_symbol[::-1]}'
        await self.send_message(msg_card)

    async def send_own_bid(self, bid):
        bid = bid.replace('N', 'NT')
        msg_bid = f'{SEATS[self.player_i]} bids {bid}'
        if bid == 'PASS':
            msg_bid = f'{SEATS[self.player_i]} passes'
        elif bid == 'X':
            msg_bid = f'{SEATS[self.player_i]} doubles'
        elif bid == 'XX':
            msg_bid = f'{SEATS[self.player_i]} redoubles'
        
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

        card_resp_parts = card_resp.strip().split()

        assert card_resp_parts[0] == SEATS[player_i], f"{card_resp_parts[0]} != {SEATS[player_i]}"

        cr = CardResp(
            card=Card.from_symbol(card_resp_parts[-1][::-1].upper()),
            candidates=[],
            samples=[],
            shape=-1,
            hcp=-1, 
            quality=None
        )
        self.card_responses.append(cr)

        return card_resp_parts[-1][::-1].upper()

    async def receive_bid_for(self, player_i):
        msg_ready = f"{SEATS[self.player_i]} ready for {SEATS[player_i]}'s bid."
        await self.send_message(msg_ready)
        
        bid_resp = await self.receive_line()
        bid_resp_parts = bid_resp.strip().split()

        assert bid_resp_parts[0] == SEATS[player_i]

        # This is to prevent the client failing, when receiving an alert
        if (bid_resp_parts[1].upper() not in ['PASSES', 'DOUBLES', 'REDOUBLES']):
            bid = bid_resp_parts[2].rstrip('.').upper().replace('NT', 'N')
        else:
            bid = bid_resp_parts[1].upper()

        return {
            'PASSES': 'PASS',
            'DOUBLES': 'X',
            'REDOUBLES': 'XX'
        }.get(bid, bid)

    async def receive_dummy(self):
        dummy_i = (self.decl_i + 2) % 4

        if self.player_i == dummy_i:
            return self.hand_str
        else:
            msg_ready = f'{self.seat} ready for dummy.'
            await self.send_message(msg_ready)
            line = await self.receive_line()
            if (line == "Start of board"):
                return None
            # Dummy's cards : S A Q T 8 2. H K 7. D K 5 2. C A 7 6.
            return TMClient.parse_hand(line)

    async def send_ready(self):
        await self.send_message(f'{self.seat} ready to start.')

    async def receive_deal(self):
        await self.receive_line()

        await self.send_message(f'{self.seat} ready for deal.')
        np.random.seed(42)


        #If we are restarting a match we will receive 
        # 'Board number 1. Dealer North. Neither vulnerable. \r\n'
        deal_line_1 = await self.receive_line()
        if deal_line_1 == "Start of Board":
            await self.send_message(f'{self.seat} ready for deal.')
            deal_line_1 = await self.receive_line()

        pattern = r'Board number (\d+)\.'

        match = re.search(pattern, deal_line_1)

        if match:
            self.board_number = match.group(1)

        await self.send_message(f'{self.seat} ready for cards.')
        # "South's cards : S K J 9 3. H K 7 6. D A J. C A Q 8 7. \r\n"
        # "North's cards : S 9 3. H -. D J 7 5. C A T 9 8 6 4 3 2."
        deal_line_2 = await self.receive_line()

        rx_dealer_vuln = r'(?P<dealer>[a-zA-Z]+?)\.\s(?P<vuln>.+?)\svulnerable'
        match = re.search(rx_dealer_vuln, deal_line_1)

        if deal_line_2 is None or deal_line_2 == "":
            raise ValueError("Deal not received")
        
        hand_str = TMClient.parse_hand(deal_line_2)

        dealer_i = 'NESW'.index(match.groupdict()['dealer'][0])
        vuln_str = match.groupdict()['vuln']
        assert vuln_str in {'Neither', 'N/S', 'E/W', 'Both'}
        vuln_ns = vuln_str == 'N/S' or vuln_str == 'Both'
        vuln_ew = vuln_str == 'E/W' or vuln_str == 'Both'

        return dealer_i, vuln_ns, vuln_ew, hand_str
    
    @staticmethod
    def parse_hand(s):
        # Translate hand from Tablemanager format to PBN
        try:
            hand = s[s.index(':') + 1 : s.rindex('.')] \
                .replace(' ', '').replace('-', '').replace('S', '').replace('H', '').replace('D', '').replace('C', '')
            return hand
        except Exception as ex:
            print(ex)
            print(f"Protocol error. Received {hand} - Expected a hand")


    async def send_message(self, message: str):
        time.sleep(0.1)
        try:
            print(f'{datetime.datetime.now().strftime("%H:%M:%S")} sending:   {message.ljust(60)}', end='')

            self.writer.write((message + "\r\n").encode())
            await self.writer.drain()
            print(' ...sent successfully.')
        except ConnectionAbortedError as ex:
            print(f'Error: {str(ex)}')
            # Handle the error gracefully, such as logging it or notifying the user
            # Close the connection (in case it's not already closed)
            self._is_connected = False
            # Stop the event loop to terminate the application
            asyncio.get_event_loop().stop()

    async def receive_line(self) -> str:
        try:
            print('{} receiving: '.format(datetime.datetime.now().strftime("%H:%M:%S")), end='')
            message = await self.reader.readline()
            msg = message.decode().replace('\r', '').replace('\n', '')
            print(f'{msg.ljust(60)} ...received.')
            if (msg == "End of session"):
                sys.exit()
            return msg
        except ConnectionAbortedError as ex:
            print(f'Match terminated: {str(ex)}')    
            # Close the connection (in case it's not already closed)
            self._is_connected = False
            # Stop the event loop to terminate the application
            asyncio.get_event_loop().stop()

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



#  Examples of how to start the table manager
# python table_manager_client.py --name BEN --seat North
# python table_manager_client.py --name BEN --seat South

async def main():
    
    # Get the path to the config file
    config_path = get_execution_path()
    
    base_path = os.getenv('BEN_HOME') or config_path

    parser = argparse.ArgumentParser(description="Table manager interface")
    parser.add_argument("--host", type=validate_ip, default="127.0.0.1", help="IP for Table Manager")
    parser.add_argument("--port", type=int, default=2000, help="Port for Table Manager")
    parser.add_argument("--name", required=True, help="Name in Table Manager")
    parser.add_argument("--seat", required=True, help="Where to sit (North, East, South or West)")
    parser.add_argument("--config", default=f"{base_path}/config/default.conf", help="Filename for configuration")
    parser.add_argument("--biddingonly", type=bool, default=False, help="Only bid, no play")
    parser.add_argument("--verbose", type=bool, default=False, help="Output samples and other information during play")

    args = parser.parse_args()

    host = args.host
    port = args.port
    name = args.name
    seat = args.seat
    configfile = args.config

    verbose = args.verbose
    biddingonly = args.biddingonly

    np.set_printoptions(precision=2, suppress=True)

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

    client = TMClient(name, seat, models, Sample.from_conf(configuration, verbose), verbose)
    print(f"Connecting to {host}:{port}")
    await client.connect(host, port)
    first = True

    if client.is_connected:
        await client.send_ready()

    shelf_filename = f"{base_path}/{seat}-{name}"

    while client.is_connected:
        t_start = time.time()
        await client.run(biddingonly)
        # The deal just played is saved for later review
        # if bidding only we do not save the deal
        if not biddingonly:
            # If we just played board 1 we assume a new match
            deal = client.to_dict()
            if deal["board_number"] == "1" and first:
                cleanup_shelf(shelf_filename)
                first = False
            with shelve.open(shelf_filename) as db:
                print("Saving Board: ",client.deal_str)
                print('{1} Board played in {0:0.1f} seconds.'.format(time.time() - t_start, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                if deal["board_number"]+"-Open" not in db:
                    db[deal["board_number"]+"-Open"] = deal
                else:
                    db[deal["board_number"]+"-Closed"] = deal


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    except ValueError as e:
        print("Error in configuration - typical the models do not match the configuration.")
        print(e)
        sys.exit(0)
    finally:
        loop.close()

