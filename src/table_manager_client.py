import ipaddress
import sys
import argparse
import re
import pprint
import asyncio
import numpy as np
from sample import Sample
import bots
import deck52
import conf
import shelve
import uuid


from nn.models import Models
from deck52 import decode_card
from bidding import bidding
from objects import Card

SEATS = ['North', 'East', 'South', 'West']

class TMClient:

    def __init__(self, name, seat, models, ns, ew, sampler):
        self.name = name
        self.seat = seat
        self.player_i = SEATS.index(self.seat)
        self.reader = None
        self.writer = None
        self.ns = ns
        self.ew = ew
        self.models = models
        self.sampler = sampler
        self._is_connected = False

    @property
    def is_connected(self):
        return self._is_connected
    
    async def run(self):
        self.dealer_i, self.vuln_ns, self.vuln_ew, self.hand_str = await self.receive_deal()

        auction = await self.bidding()

        self.contract = bidding.get_contract(auction)
        if self.contract is None:
            return

        level = int(self.contract[0])
        strain_i = bidding.get_strain_i(self.contract)
        self.decl_i = bidding.get_decl_i(self.contract)

        print(auction)
        #print(self.contract)
        #print(self.decl_i)

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

        await self.send_message(f'Connecting "{self.name}" as {self.seat} using protocol version 18.')

        await self.receive_line()
        
        await self.send_message(f'{self.seat} ready for teams.')

        await self.receive_line()

    async def bidding(self):
        vuln = [self.vuln_ns, self.vuln_ew]
        bot = bots.BotBid(vuln, self.hand_str, self.models, self.ns, self.ew, 0.1, self.sampler)
        
        auction = ['PAD_START'] * self.dealer_i

        player_i = self.dealer_i

        while not bidding.auction_over(auction):
            if player_i == self.player_i:
                # now it's this player's turn to bid
                bid_resp = bot.bid(auction)
                pprint.pprint(bid_resp.samples, width=80)
                auction.append(bid_resp.bid)
                await self.send_own_bid(bid_resp.bid)
            else:
                # just wait for the other player's bid
                bid = await self.receive_bid_for(player_i)
                auction.append(bid)

            player_i = (player_i + 1) % 4

        return auction

    async def opening_lead(self, auction):
        contract = bidding.get_contract(auction)
        decl_i = bidding.get_decl_i(contract)
        on_lead_i = (decl_i + 1) % 4
        
        if self.player_i == on_lead_i:
            # this player is on lead
            await self.receive_line()

            bot_lead = bots.BotLead(
                [self.vuln_ns, self.vuln_ew], 
                self.hand_str,
                self.models,
                self.ns,
                self.ew,
                self.sampler
            )
            card_resp = bot_lead.lead(auction)
            card_symbol = card_resp.card.symbol()
            # card_symbol = 'D5'
            await self.send_card_played(card_symbol)
            return card_symbol
        else:
            # just send that we are ready for the opening lead
            return await self.receive_card_play_for(on_lead_i, 0)

    async def play(self, auction, opening_lead52):
        contract = bidding.get_contract(auction)
        
        level = int(contract[0])
        strain_i = bidding.get_strain_i(contract)
        decl_i = bidding.get_decl_i(contract)
        is_decl_vuln = [self.vuln_ns, self.vuln_ew, self.vuln_ns, self.vuln_ew][decl_i]
        cardplayer_i = (self.player_i + 3 - decl_i) % 4  # lefty=0, dummy=1, righty=2, decl=3
        print(f'play starts. decl_i={decl_i}, player_i={self.player_i}, cardplayer_i={cardplayer_i}')

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

        card_players = [
            bots.CardPlayer(self.models.player_models, 0, lefty_hand_str, dummy_hand_str, contract, is_decl_vuln),
            bots.CardPlayer(self.models.player_models, 1, dummy_hand_str, decl_hand_str, contract, is_decl_vuln),
            bots.CardPlayer(self.models.player_models, 2, righty_hand_str, dummy_hand_str, contract, is_decl_vuln),
            bots.CardPlayer(self.models.player_models, 3, decl_hand_str, dummy_hand_str, contract, is_decl_vuln)
        ]

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
            print("Playing trick {}".format(trick_i+1))

            for player_i in map(lambda x: x % 4, range(leader_i, leader_i + 4)):
                #print('player {}'.format(player_i))

                nesw_i = (decl_i + player_i + 1) % 4 # N=0, E=1, S=2, W=3
                
                if trick_i == 0 and player_i == 0:
                    #print('skipping')
                    for i, card_player in enumerate(card_players):
                        card_player.set_card_played(trick_i=trick_i, leader_i=leader_i, i=0, card=opening_lead)

                    continue

                card52 = None
                if player_i == 1 and cardplayer_i == 3:
                    # it's dummy's turn and this is the declarer
                    print('declarers turn for dummy')

                    rollout_states = self.sampler.init_rollout_states(trick_i, player_i, card_players, player_cards_played, shown_out_suits, current_trick, 100, auction, card_players[player_i].hand.reshape((-1, 32)), [self.vuln_ns, self.vuln_ew], self.models, self.ns, self.ew)

                    card_resp = card_players[player_i].play_card(trick_i, leader_i, current_trick52, rollout_states)

                    for idx, candidate in enumerate(card_resp.candidates, start=1):
                        print(f"{candidate.card} Expected Score: {str(int(candidate.expected_score)).ljust(5)} Tricks {candidate.expected_tricks:.2f} Insta_score {candidate.insta_score:.4f}")
                    for idx, sample in enumerate(card_resp.samples, start=1):                  
                        print(f"{sample}")
                        if idx == 20:
                            break

                    card52 = card_resp.card.code()
                    
                    await self.send_card_played(card_resp.card.symbol()) 
                elif player_i == cardplayer_i and player_i != 1:
                    # we are on play
                    #print(f'{player_i} turn')

                    rollout_states = self.sampler.init_rollout_states(trick_i, player_i, card_players, player_cards_played, shown_out_suits, current_trick, 100, auction, card_players[player_i].hand.reshape((-1, 32)), [self.vuln_ns, self.vuln_ew], self.models, self.ns, self.ew)

                    card_resp = card_players[player_i].play_card(trick_i, leader_i, current_trick52, rollout_states)

                    for idx, candidate in enumerate(card_resp.candidates, start=1):
                        print(f"{candidate.card} Expected Score: {str(int(candidate.expected_score)).ljust(5)} Tricks {candidate.expected_tricks:.2f} Insta_score {candidate.insta_score:.4f}")
                    for idx, candidate in enumerate(card_resp.samples, start=1):                  
                        print(f"{candidate}")
                        if idx == 20:
                            break

                    card52 = card_resp.card.code()
                    
                    await self.send_card_played(card_resp.card.symbol()) 
                else:
                    # another player is on play, we just have to wait for their card
                    card52_symbol = await self.receive_card_play_for(nesw_i, trick_i)
                    card52 = Card.from_symbol(card52_symbol).code()
                
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

            for i in [cardplayer_i] + ([1] if cardplayer_i == 3 else []):
                if cardplayer_i == 1:
                    break
                assert np.min(card_players[i].hand52) == 0
                assert np.min(card_players[i].public52) == 0
                assert np.sum(card_players[i].hand52) == 13 - trick_i - 1
                assert np.sum(card_players[i].public52) == 13 - trick_i - 1

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
            for i in [cardplayer_i] + ([1] if cardplayer_i == 3 else []):
                if cardplayer_i == 1:
                    break
                assert np.min(card_players[i].x_play[:, trick_i + 1, 0:32]) == 0
                assert np.min(card_players[i].x_play[:, trick_i + 1, 32:64]) == 0
                assert np.sum(card_players[i].x_play[:, trick_i + 1, 0:32], axis=1) == 13 - trick_i - 1
                assert np.sum(card_players[i].x_play[:, trick_i + 1, 32:64], axis=1) == 13 - trick_i - 1

            trick_winner = (leader_i + deck52.get_trick_winner_i(current_trick52, (strain_i - 1) % 5)) % 4
            trick_won_by.append(trick_winner)

            if trick_winner % 2 == 0:
                card_players[0].n_tricks_taken += 1
                card_players[2].n_tricks_taken += 1
            else:
                card_players[1].n_tricks_taken += 1
                card_players[3].n_tricks_taken += 1

            # Wonder why this is repeated
            #print('trick52 {} cards={}. won by {}'.format(trick_i, list(map(decode_card, #current_trick52)), trick_winner))

            print('trick52 {} cards={}. won by {}'.format(trick_i+1, list(map(decode_card, current_trick52)), trick_winner))

            # update cards shown
            for i, card in enumerate(current_trick):
                player_cards_played[(leader_i + i) % 4].append(card)
            
            leader_i = trick_winner
            current_trick = []
            current_trick52 = []

            # player on lead will receive message (or decl if dummy on lead)
            if leader_i == 1:
                if cardplayer_i == 3:
                    await self.receive_line()
            elif leader_i == cardplayer_i:
                await self.receive_line()

        # play last trick
        for player_i in map(lambda x: x % 4, range(leader_i, leader_i + 4)):
            nesw_i = (decl_i + player_i + 1) % 4 # N=0, E=1, S=2, W=3
            card52 = None
            if player_i == 1 and cardplayer_i == 3 or player_i == cardplayer_i and player_i != 1:
                # we are on play
                card52 = np.nonzero(card_players[player_i].hand52)[0][0]
                card52_symbol = Card.from_code(card52).symbol()
                await self.send_card_played(card52_symbol)
            else:
                # someone else is on play. we just have to wait for their card
                card52_symbol = await self.receive_card_play_for(nesw_i, trick_i)
                card52 = Card.from_symbol(card52_symbol).code()

            card = deck52.card52to32(card52)

            current_trick.append(card)
            current_trick52.append(card52)

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

    async def send_card_played(self, card_symbol):
        msg_card = f'{self.seat} plays {card_symbol[::-1]}'
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
        msg_ready = f"{self.seat} ready for {SEATS[player_i]}'s card to trick {trick_i + 1}."
        await self.send_message(msg_ready)

        card_resp = await self.receive_line()
        card_resp_parts = card_resp.strip().split()

        assert card_resp_parts[0] == SEATS[player_i]

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
            deal_line_1 = await self.receive_line()
        # "South's cards : S K J 9 3. H K 7 6. D A J. C A Q 8 7. \r\n"
        # "North's cards : S 9 3. H -. D J 7 5. C A T 9 8 6 4 3 2."
        deal_line_2 = await self.receive_line()
        print(f"deal_line_2 {deal_line_2}")

        rx_dealer_vuln = r'(?P<dealer>[a-zA-z]+?)\.\s(?P<vuln>.+?)\svulnerable'
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
        return s[s.index(':') + 1 : s.rindex('.')] \
            .replace(' ', '').replace('-', '').replace('S', '').replace('H', '').replace('D', '').replace('C', '')

    async def send_message(self, message: str):
        print(f'sending:   {message.ljust(60)}', end='')
        self.writer.write((message+"\n").encode())
        await self.writer.drain()
        print(' ...sent successfully.')

    async def receive_line(self) -> str:
        try:
            print('receiving: ', end='')
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

#  Examples of how to start the table manager
# python table_manager_client.py --host 127.0.0.1 --port 2000 --name SAYC --seat North --ns 1 --ew 1 --config config/sayc.conf
# python table_manager_client.py --host 127.0.0.1 --port 2000 --name SAYC --seat South --ns 1 --ew 1 --config config/sayc.conf

async def main():
    
    parser = argparse.ArgumentParser(description="Table manager interface")
    parser.add_argument("--host", type=validate_ip, required=True, help="IP for Table Manager")
    parser.add_argument("--port", type=int, required=True, help="Port for Table Manager")
    parser.add_argument("--name", required=True, help="Name in Table Manager")
    parser.add_argument("--seat", required=True, help="Where to sit (North, East, South or West)")
    parser.add_argument("--config", default="./config/default.conf", help="Filename for configuration")
    parser.add_argument("--ns", type=int, default=-1, help="System for NS")
    parser.add_argument("--ew", type=int, default=-1, help="System for EW")
    parser.add_argument("--is_continue", type=bool, default=False, help="Continuing a match")
    args = parser.parse_args()

    host = args.host
    port = args.port
    name = args.name
    seat = args.seat

    configfile = args.config

    ns = args.ns
    ew = args.ew

    is_continue = args.is_continue

    models = Models.from_conf(conf.load(configfile))

    client = TMClient(name, seat, models, ns, ew, Sample.from_conf(conf.load(configfile))
)
    print(f"Connecting to {host}:{port}")
    await client.connect(host, port)
    
    if client.is_connected:
        if is_continue:    
            await client.receive_line()
        await client.send_ready()

    while client.is_connected:
        await client.run()
        # The deal just played should be saved for later review


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
