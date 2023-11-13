import numpy as np
import bots
import deck52

from bidding import bidding
from sample import Sample
from objects import BidResp, Card, CardResp

class CardByCard:

    def __init__(self, dealer, vuln, hands, auction, play, models, ns, ew, sampler, verbose):
        self.dealer_i = {'N': 0, 'E': 1, 'S': 2, 'W': 3}[dealer]
        self.vuln = vuln
        self.hands = hands
        self.auction = auction
        self.padded_auction = ['PAD_START'] * self.dealer_i + self.auction
        self.play = play
        self.bid_responses = []
        self.card_responses = []
        self.cards = {}
        self.models = models
        self.sampler = sampler
        self.ns = ns
        self.ew = ew
        self.verbose = verbose

    def analyze(self):
        print('analyzing the bidding')
        self.analyze_bidding()
        print('analyzing the play')
        self.analyze_opening_lead()
        self.analyze_play()

    def analyze_bidding(self):
        bidder_bots = [bots.BotBid(self.vuln, hand, self.models, self.ns, self.ew, self.sampler, self.verbose) for hand in self.hands]

        player_i = self.dealer_i
        bid_i = self.dealer_i

        while bid_i < len(self.padded_auction):
            bid_resp = bidder_bots[player_i].bid(self.padded_auction[:bid_i])
            self.bid_responses.append(BidResp(self.padded_auction[bid_i], bid_resp.candidates, bid_resp.samples, -1, -1, "Analysis"))
            type(self).bid_eval(self.padded_auction[bid_i], bid_resp)
            bid_i += 1
            player_i = (player_i + 1) % 4
    
    @staticmethod
    def bid_eval(bid, bid_resp):
        qualifier = '.'
        if bid_resp.candidates[0].bid != bid:
            qualifier = '?'
        print(f'{bid} {qualifier}')

    @staticmethod
    def card_eval(card, card_resp):
        qualifier = '.'
        best_tricks = card_resp.candidates[0].expected_tricks
        for candidate in card_resp.candidates:
            if candidate.card.symbol() == card:
                if best_tricks - candidate.expected_tricks > 0.1:
                    qualifier = '?'
                if best_tricks - candidate.expected_tricks > 0.6:
                    qualifier = '??'
        print(f'{card} {qualifier}')

    def analyze_opening_lead(self):
        contract = bidding.get_contract(self.padded_auction)
        decl_i = bidding.get_decl_i(contract)

        print(self.play[0])

        bot_lead = bots.BotLead(self.vuln, self.hands[(decl_i + 1) % 4], self.models, -1, -1, 0.05, self.sampler, False)

        card_resp = bot_lead.find_opening_lead(self.padded_auction)
        card_resp = CardResp(Card.from_symbol(self.play[0]), card_resp.candidates, card_resp.samples, -1, -1)
        self.card_responses.append(card_resp)
        self.cards[card_resp.card.symbol()] = card_resp

    def analyze_play(self):
        contract = bidding.get_contract(self.padded_auction)
        
        level = int(contract[0])
        strain_i = bidding.get_strain_i(contract)
        decl_i = bidding.get_decl_i(contract)
        is_decl_vuln = [self.vuln[0], self.vuln[1], self.vuln[0], self.vuln[1]][decl_i]

        lefty_hand = self.hands[(decl_i + 1) % 4]
        dummy_hand = self.hands[(decl_i + 2) % 4]
        righty_hand = self.hands[(decl_i + 3) % 4]
        decl_hand = self.hands[decl_i]

        card_players = [
            bots.CardPlayer(self.models.player_models, 0, lefty_hand, dummy_hand, contract, is_decl_vuln),
            bots.CardPlayer(self.models.player_models, 1, dummy_hand, decl_hand, contract, is_decl_vuln),
            bots.CardPlayer(self.models.player_models, 2, righty_hand, dummy_hand, contract, is_decl_vuln),
            bots.CardPlayer(self.models.player_models, 3, decl_hand, dummy_hand, contract, is_decl_vuln)
        ]

        player_cards_played = [[] for _ in range(4)]
        shown_out_suits = [set() for _ in range(4)]

        leader_i = 0

        tricks = []
        tricks52 = []
        trick_won_by = []

        opening_lead52 = Card.from_symbol(self.play[0]).code()
        opening_lead = deck52.card52to32(opening_lead52)

        current_trick = [opening_lead]
        current_trick52 = [opening_lead52]

        card_players[0].hand52[opening_lead52] -= 1
        card_i = 1

        for trick_i in range(12):
            for player_i in map(lambda x: x % 4, range(leader_i, leader_i + 4)):
                if trick_i == 0 and player_i == 0:
                    for i, card_player in enumerate(card_players):
                        card_player.set_card_played(trick_i=trick_i, leader_i=leader_i, i=0, card=opening_lead)
                    continue
                
                rollout_states = None
                if isinstance(card_players[player_i], bots.CardPlayer):
                    rollout_states, c_hcp, c_shp = self.sampler.init_rollout_states(trick_i, player_i, card_players, player_cards_played, shown_out_suits, current_trick, self.padded_auction, card_players[player_i].hand.reshape((-1, 32)), self.vuln, self.models, self.ns, self.ew)

                card_resp = card_players[player_i].play_card(trick_i, leader_i, current_trick52, rollout_states)
                card_resp = CardResp(Card.from_symbol(self.play[card_i]), card_resp.candidates, card_resp.samples, c_shp, c_hcp)
                self.card_responses.append(card_resp)
                self.cards[self.play[card_i]] = card_resp

                type(self).card_eval(self.play[card_i], card_resp)

                card_i += 1
                if card_i >= len(self.play):
                    return
                
                card52 = card_resp.card.code()
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

            # update cards shown
            for i, card in enumerate(current_trick):
                player_cards_played[(leader_i + i) % 4].append(card)
            
            leader_i = trick_winner
            current_trick = []
            current_trick52 = []

