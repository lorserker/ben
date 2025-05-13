import numpy as np
import botbidder
import botopeninglead
import botcardplayer
import deck52

from bidding import bidding
from sample import Sample
from objects import BidResp, Card, CardResp
from util import get_play_status, get_singleton, get_possible_cards
class CardByCard:

    def __init__(self, dealer, vuln, hands, auction, play, models, sampler, verbose):
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
        self.verbose = verbose
        from ddsolver import ddsolver
        self.dds = ddsolver.DDSolver()

    async def analyze(self):
        print('analyzing the bidding')
        self.analyze_bidding()
        print('analyzing opening lead')
        self.analyze_opening_lead()
        print('analyzing play')
        await self.analyze_play()

    def analyze_bidding(self):
        from ddsolver import ddsolver
        dds = ddsolver.DDSolver()
        bidder_bots = [botbidder.BotBid(self.vuln, hand, self.models, self.sampler, idx, self.dealer_i, dds, False, self.verbose) for idx, hand in enumerate(self.hands)]

        player_i = self.dealer_i
        bid_i = self.dealer_i

        while bid_i < len(self.padded_auction):
            bid_resp = bidder_bots[player_i].bid(self.padded_auction[:bid_i])
            self.bid_responses.append(BidResp(self.padded_auction[bid_i], bid_resp.candidates, bid_resp.samples, bid_resp.hcp, bid_resp.shape, "Analysis", bid_resp.quality, bid_resp.alert, bid_resp.explanation))
            type(self).bid_eval(self.padded_auction[bid_i], bid_resp)
            bid_i += 1
            player_i = (player_i + 1) % 4
    
    @staticmethod
    def bid_eval(bid, bid_resp):
        if bid_resp.candidates[0].bid != bid:
            candidates_list = bid_resp.candidates
            print(f'{bid} Suggested bid from NN: {bid_resp.candidates[0]}')
            # Check if 'b' is in the "bid" property of any CandidateBid object and get the index
            for index, candidate in enumerate(candidates_list):
                if hasattr(candidate, "bid") and bid in candidate.bid:
                    print(f'{bid} NN-values:{candidate}')
                    break  # Break the loop if found
            else:
                print(f'{bid} is not in the bids from the neural network')
        else:
            print(f'{bid} OK NN-value: {bid_resp.candidates[0].insta_score:.3f}')

    @staticmethod
    def card_eval(card, card_resp):
        qualifier = 'OK'
        if len(card_resp.candidates) == 0:
            qualifier = card_resp.who
        else:
            best_tricks = card_resp.candidates[0].expected_tricks_dd
            for candidate in card_resp.candidates:
                if candidate.card.symbol() == card and candidate.expected_tricks_dd != None:
                    if best_tricks - candidate.expected_tricks_dd > 0.1:
                        qualifier = f'? losing: {best_tricks - candidate.expected_tricks_dd:.2f}'
                    if best_tricks - candidate.expected_tricks_dd > 0.6:
                        qualifier = f'?? losing: {best_tricks - candidate.expected_tricks_dd:.2f}'
        print(f'{card} {qualifier}')

    def analyze_opening_lead(self):
        contract = bidding.get_contract(self.padded_auction)
        decl_i = bidding.get_decl_i(contract)

        print(self.play[0])

        bot_lead = botopeninglead.BotLead(self.vuln, self.hands[(decl_i + 1) % 4], self.models, self.sampler, (decl_i + 1) % 4, self.dealer_i, self.dds, False)

        card_resp = bot_lead.find_opening_lead(self.padded_auction, {})
        card_resp = CardResp(Card.from_symbol(self.play[0]), card_resp.candidates, card_resp.samples, card_resp.hcp, card_resp.shape, card_resp.quality,'', claim = -1)
        self.card_responses.append(card_resp)
        self.cards[card_resp.card.symbol()] = card_resp
        type(self).card_eval(self.play[0], card_resp)

    async def analyze_play(self):
        contract = bidding.get_contract(self.padded_auction)
        level = int(contract[0])
        strain_i = bidding.get_strain_i(contract)
        decl_i = bidding.get_decl_i(contract)
        is_decl_vuln = [self.vuln[0], self.vuln[1], self.vuln[0], self.vuln[1]][decl_i]

        lefty_hand = self.hands[(decl_i + 1) % 4]
        dummy_hand = self.hands[(decl_i + 2) % 4]
        righty_hand = self.hands[(decl_i + 3) % 4]
        decl_hand = self.hands[decl_i]

        # Should be found based on bidding
        aceking = {}
        features = {}
        if self.models.pimc_use_declaring or self.models.pimc_use_defending:
            from pimc.PIMC import BGADLL
            pimc = BGADLL(self.models, dummy_hand, decl_hand, contract, is_decl_vuln, self.sampler, self.verbose)
            if self.verbose:
                print("PIMC",dummy_hand, decl_hand, contract)
        else:
            pimc = None
        from ddsolver import ddsolver
        dd = ddsolver.DDSolver()

        card_players = [
            botcardplayer.CardPlayer(self.models, 0, lefty_hand, dummy_hand, contract, is_decl_vuln, self.sampler, pimc, dd, self.verbose),
            botcardplayer.CardPlayer(self.models, 1, dummy_hand, decl_hand, contract, is_decl_vuln, self.sampler, pimc, dd, self.verbose),
            botcardplayer.CardPlayer(self.models, 2, righty_hand, dummy_hand, contract, is_decl_vuln, self.sampler, pimc, dd, self.verbose),
            botcardplayer.CardPlayer(self.models, 3, decl_hand, dummy_hand, contract, is_decl_vuln, self.sampler, pimc, dd, self.verbose)
        ]

        player_cards_played = [[] for _ in range(4)]
        player_cards_played52 = [[] for _ in range(4)]
        shown_out_suits = [set() for _ in range(4)]
        discards = [set() for _ in range(4)]

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
                        card_player.set_real_card_played(opening_lead52, 0, True)
                        card_player.set_card_played(trick_i=trick_i, leader_i=leader_i, i=0, card=opening_lead)
                    continue

                card52 = None
                card_resp = None               
                rollout_states = None
                if isinstance(card_players[player_i], botcardplayer.CardPlayer):
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
                        played_cards = [card for row in player_cards_played52 for card in row] + current_trick52

                        rollout_states, bidding_scores, c_hcp, c_shp, quality, probability_of_occurence, lead_scores, play_scores, logical_play_scores, discard_scores, worlds = self.sampler.init_rollout_states(trick_i, player_i, card_players, played_cards, player_cards_played, shown_out_suits, discards, aceking, current_trick, opening_lead52, self.padded_auction, card_players[player_i].hand_str, card_players[player_i].public_hand_str, self.vuln, self.models, card_players[player_i].get_random_generator())

                        card_players[player_i].check_pimc_constraints(trick_i, rollout_states, quality)
                        card_resp = card_players[player_i].play_card(trick_i, leader_i, current_trick52, tricks52, rollout_states, worlds, bidding_scores, quality, probability_of_occurence, shown_out_suits, play_status, lead_scores, play_scores, logical_play_scores, discard_scores, features)
                        card_resp.hcp = c_hcp
                        card_resp.shape = c_shp

                self.card_responses.append(card_resp)
                self.cards[self.play[card_i]] = card_resp

                type(self).card_eval(self.play[card_i], card_resp)

                card_i += 1
                if card_i >= len(self.play):
                    return
                
                card52 = card_resp.card.code()
                card32 = deck52.card52to32(card52)

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

            trick_winner = (leader_i + deck52.get_trick_winner_i(current_trick52, (strain_i - 1) % 5)) % 4
            trick_won_by.append(trick_winner)

            if trick_winner % 2 == 0:
                card_players[0].n_tricks_taken += 1
                card_players[2].n_tricks_taken += 1
            else:
                card_players[1].n_tricks_taken += 1
                card_players[3].n_tricks_taken += 1

            # update cards shown
            for i, card32 in enumerate(current_trick):
                player_cards_played[(leader_i + i) % 4].append(card32)           
            for i, card52 in enumerate(current_trick52):
                player_cards_played52[(leader_i + i) % 4].append(card52)

            leader_i = trick_winner
            current_trick = []
            current_trick52 = []

