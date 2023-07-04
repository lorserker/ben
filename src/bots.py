import time
import random
import pprint

import numpy as np

import binary
import nn.player as player
import deck52
import sample
import scoring

from objects import BidResp, CandidateBid, Card, CardResp, CandidateCard
from bidding import bidding
from bidding.binary import parse_hand_f

from util import hand_to_str, expected_tricks, p_make_contract


class BotBid:

    def __init__(self, vuln, hand_str, models):
        self.vuln = vuln
        self.hand_str = hand_str
        self.hand = binary.parse_hand_f(32)(hand_str)
        self.min_candidate_score = 0.1

        self.model = models.bidder_model
        self.state = models.bidder_model.zero_state
        self.lead_model = models.lead
        self.sd_model = models.sd_model

        self.binfo_model = models.binfo

    @staticmethod
    def get_n_steps_auction(auction):
        hand_i = len(auction) % 4
        i = hand_i
        while i < len(auction) and auction[i] == 'PAD_START':
            i += 4
        return 1 + (len(auction) - i) // 4

    def get_binary(self, auction):
        n_steps = BotBid.get_n_steps_auction(auction)
        hand_ix = len(auction) % 4
        
        X = binary.get_auction_binary(n_steps, auction, hand_ix, self.hand, self.vuln)

        return X[:,-1,:]

    def bid(self, auction):
        candidates = self.get_bid_candidates(auction)
        hands_np = self.sample_hands(auction)

        samples = []
        for i in range(min(10, hands_np.shape[0])):
            samples.append('%s %s %s %s' % (
                hand_to_str(hands_np[i,0,:]),
                hand_to_str(hands_np[i,1,:]),
                hand_to_str(hands_np[i,2,:]),
                hand_to_str(hands_np[i,3,:]),
            ))
 
        if BotBid.do_rollout(auction, candidates):
            ev_candidates = []
            for candidate in candidates:
                auctions_np = self.bidding_rollout(auction, candidate.bid, hands_np)
                contracts, decl_tricks_softmax = self.expected_tricks(hands_np, auctions_np)
                ev = self.expected_score(len(auction) % 4, contracts, decl_tricks_softmax)
                ev_c = candidate.with_expected_score(np.mean(ev))
                ev_candidates.append(ev_c)
            candidates = sorted(ev_candidates, key=lambda c: c.expected_score, reverse=True)

            return BidResp(bid=candidates[0].bid, candidates=candidates, samples=samples)
        
        return BidResp(bid=candidates[0].bid, candidates=candidates, samples=samples)
    
    @staticmethod
    def do_rollout(auction, candidates):
        if len(candidates) == 1:
            return False
        
        if BotBid.get_n_steps_auction(auction) > 1:
            return True
        
        if any(bid not in ('PASS', 'PAD_START') for bid in auction):
            return True
        
        if all(candidate.bid != 'PASS' for candidate in candidates):
            return True

        return False

    def get_bid_candidates(self, auction):
        bid_softmax = self.next_bid_np(auction)[0]

        candidates = []
        while True:
            bid_i = np.argmax(bid_softmax)
            if bid_softmax[bid_i] < self.min_candidate_score and len(candidates) > 0:
                break
            if bidding.can_bid(bidding.ID2BID[bid_i], auction):
                candidates.append(CandidateBid(bid=bidding.ID2BID[bid_i], insta_score=bid_softmax[bid_i]))
            bid_softmax[bid_i] = 0

        return candidates

    def next_bid_np(self, auction):
        x = self.get_binary(auction)
        bid_np, next_state = self.model.model(x, self.state)
        self.state = next_state

        return bid_np
    
    def sample_hands(self, auction_so_far):
        turn_to_bid = len(auction_so_far) % 4
        n_steps = BotBid.get_n_steps_auction(auction_so_far)
        lho_pard_rho = sample.sample_cards_auction(2048, n_steps, auction_so_far, turn_to_bid, self.hand, self.vuln, self.model, self.binfo_model)[:64]
        n_samples = lho_pard_rho.shape[0]
        
        hands_np = np.zeros((n_samples, 4, 32), dtype=np.int32)
        hands_np[:,turn_to_bid,:] = self.hand
        for i in range(1, 4):
            hands_np[:, (turn_to_bid + i) % 4, :] = lho_pard_rho[:,i-1,:]
 
        return hands_np

    def bidding_rollout(self, auction_so_far, candidate_bid, hands_np):
        auction = [*auction_so_far, candidate_bid]
        
        n_samples = hands_np.shape[0]
        
        n_steps_vals = [0, 0, 0, 0]
        for i in range(1, 5):
            n_steps_vals[(len(auction_so_far) % 4 + i) % 4] = BotBid.get_n_steps_auction(auction_so_far + ['?'] * i)  
        
        # initialize auction vector
        auction_np = np.ones((n_samples, 64), dtype=np.int32) * bidding.BID2ID['PAD_END']
        for i, bid in enumerate(auction):
            auction_np[:,i] = bidding.BID2ID[bid]
            
        bid_i = len(auction) - 1
        turn_i = len(auction) % 4
        while not np.all(auction_np[:,bid_i] == bidding.BID2ID['PAD_END']):
            X = binary.get_auction_binary(n_steps_vals[turn_i], auction_np, turn_i, hands_np[:,turn_i,:], self.vuln)
            bid_np = self.model.model_seq(X).reshape((n_samples, n_steps_vals[turn_i], -1))[:,-1,:]
            
            bid_i += 1
            auction_np[:,bid_i] = np.argmax(bid_np, axis=1)
            
            n_steps_vals[turn_i] += 1
            turn_i = (turn_i + 1) % 4
            
        return auction_np

    def expected_tricks(self, hands_np, auctions_np):
        n_samples = hands_np.shape[0]
        s_all = np.arange(n_samples, dtype=np.int32)
        auctions, contracts = [], []
        declarers = np.zeros(n_samples, dtype=np.int32)
        strains = np.zeros(n_samples, dtype=np.int32)
        X_ftrs = np.zeros((n_samples, 42))
        B_ftrs = np.zeros((n_samples, 15))
        
        for i in range(n_samples):
            sample_auction = [bidding.ID2BID[bid_i] for bid_i in list(auctions_np[i, :]) if bid_i != 1]
            auctions.append(sample_auction)
            contract = bidding.get_contract(sample_auction)
            contracts.append(contract)
            strains[i] = 'NSHDC'.index(contract[1])
            declarers[i] = 'NESW'.index(contract[-1])
            
            hand_on_lead = hands_np[i:i+1, (declarers[i] + 1) % 4, :]
            
            X_ftrs[i,:], B_ftrs[i,:] = binary.get_lead_binary(sample_auction, hand_on_lead, self.binfo_model, self.vuln)
        
        lead_softmax = self.lead_model.model(X_ftrs, B_ftrs)
        lead_cards = np.argmax(lead_softmax, axis=1)
        
        X_sd = np.zeros((n_samples, 32 + 5 + 4*32))

        X_sd[s_all,32 + strains] = 1
        # lefty
        X_sd[:,(32 + 5 + 0*32):(32 + 5 + 1*32)] = hands_np[s_all, (declarers + 1) % 4]
        # dummy
        X_sd[:,(32 + 5 + 1*32):(32 + 5 + 2*32)] = hands_np[s_all, (declarers + 2) % 4]
        # righty
        X_sd[:,(32 + 5 + 2*32):(32 + 5 + 3*32)] = hands_np[s_all, (declarers + 3) % 4]
        # declarer
        X_sd[:,(32 + 5 + 3*32):] = hands_np[s_all, declarers]
        
        X_sd[s_all, lead_cards] = 1
        
        decl_tricks_softmax = self.sd_model.model(X_sd)
        
        return contracts, decl_tricks_softmax
    
    def expected_score(self, turn_to_bid, contracts, decl_tricks_softmax):
        n_samples = len(contracts)
        scores_by_trick = np.zeros((n_samples, 14))
        for i, contract in enumerate(contracts):
            scores_by_trick[i] = scoring.contract_scores_by_trick(contract, tuple(self.vuln))
            decl_i = 'NESW'.index(contract[-1])
            if (turn_to_bid + decl_i) % 2 == 1:
                # the other side is playing the contract
                scores_by_trick[i,:] *= -1
        
        return np.sum(decl_tricks_softmax * scores_by_trick, axis=1)


class BotLead:

    def __init__(self, vuln, hand_str, models):
        self.vuln = vuln
        self.hand_str = hand_str
        self.hand = binary.parse_hand_f(32)(hand_str)
        self.hand52 = binary.parse_hand_f(52)(hand_str)

        self.lead_model = models.lead
        self.bidder_model = models.bidder_model
        self.binfo_model = models.binfo
        self.sd_model = models.sd_model

    def lead(self, auction):
        lead_card_indexes, lead_softmax = self.get_lead_candidates(auction)
        accepted_samples, tricks = self.simulate_outcomes(4096, auction, lead_card_indexes)

        candidate_cards = []
        for i, card_i in enumerate(lead_card_indexes):
            candidate_cards.append(CandidateCard(
                card=Card.from_code(card_i, xcards=True),
                insta_score=lead_softmax[0,card_i],
                expected_tricks=np.mean(tricks[:,i,0]),
                p_make_contract=np.mean(tricks[:,i,1])
            ))
        candidate_cards = sorted(candidate_cards, key=lambda c: c.p_make_contract)

        opening_lead = candidate_cards[0].card.code()

        if opening_lead % 8 == 7:
            # it's a pip ~> choose a random one
            pips_mask = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1])
            lefty_led_pips = self.hand52.reshape((4, 13))[opening_lead // 8] * pips_mask
            opening_lead52 = (opening_lead // 8) * 13 + random.choice(np.nonzero(lefty_led_pips)[0])
        else:
            opening_lead52 = deck52.card32to52(opening_lead)

        samples = []
        for i in range(min(100, accepted_samples.shape[0])):
            samples.append('%s %s %s' % (
                hand_to_str(accepted_samples[i,0,:]),
                hand_to_str(accepted_samples[i,1,:]),
                hand_to_str(accepted_samples[i,2,:]),
            ))

        return CardResp(
            card=Card.from_code(opening_lead52),
            candidates=candidate_cards,
            samples=samples
        )

    def get_lead_candidates(self, auction):
        x_ftrs, b_ftrs = binary.get_lead_binary(auction, self.hand, self.binfo_model, self.vuln)
        lead_softmax = self.lead_model.model(x_ftrs, b_ftrs)
        lead_softmax = player.follow_suit(lead_softmax, self.hand, np.array([[0, 0, 0, 0]]))

        candidates = []

        while True:
            c = np.argmax(lead_softmax[0])
            score = lead_softmax[0][c]
            if score < 0.05:
                break
            lead_softmax[0][c] = 0
            candidates.append(c)

        return candidates, lead_softmax

    def simulate_outcomes(self, n_samples, auction, lead_card_indexes):
        contract = bidding.get_contract(auction)

        decl_i = bidding.get_decl_i(contract)
        lead_index = (decl_i + 1) % 4

        n_steps = 1 + len(auction) // 4

        accepted_samples = sample.sample_cards_auction(n_samples, n_steps, auction, lead_index, self.hand, self.vuln, self.bidder_model, self.binfo_model)

        n_accepted = accepted_samples.shape[0]

        X_sd = np.zeros((n_accepted, 32 + 5 + 4*32))

        strain_i = bidding.get_strain_i(contract)

        X_sd[:,32 + strain_i] = 1
        # lefty
        X_sd[:,(32 + 5 + 0*32):(32 + 5 + 1*32)] = self.hand.reshape(32)
        # dummy
        X_sd[:,(32 + 5 + 1*32):(32 + 5 + 2*32)] = accepted_samples[:,0,:].reshape((n_accepted, 32))
        # righty
        X_sd[:,(32 + 5 + 2*32):(32 + 5 + 3*32)] = accepted_samples[:,1,:].reshape((n_accepted, 32))
        # declarer
        X_sd[:,(32 + 5 + 3*32):] = accepted_samples[:,2,:].reshape((n_accepted, 32))

        tricks = np.zeros((n_accepted, len(lead_card_indexes), 2))

        for j, lead_card_i in enumerate(lead_card_indexes):
            X_sd[:, :32] = 0
            X_sd[:, lead_card_i] = 1

            tricks_softmax = self.sd_model.model(X_sd)

            tricks[:, j, 0:1] = expected_tricks(tricks_softmax.copy())
            tricks[:, j, 1:2] = p_make_contract(contract, tricks_softmax.copy())

        return accepted_samples, tricks


class CardPlayer:

    def __init__(self, player_models, player_i, hand_str, public_hand_str, contract, is_decl_vuln):
        self.player_models = player_models
        self.model = player_models[player_i]
        self.player_i = player_i
        self.hand = parse_hand_f(32)(hand_str).reshape(32)
        self.hand52 = parse_hand_f(52)(hand_str).reshape(52)
        self.public52 = parse_hand_f(52)(public_hand_str).reshape(52)
        self.contract = contract
        self.is_decl_vuln = is_decl_vuln
        self.n_tricks_taken = 0
        self.verbose = False
        self.level = int(contract[0])
        self.strain_i = bidding.get_strain_i(contract)

        self.init_x_play(parse_hand_f(32)(public_hand_str), self.level, self.strain_i)

        self.score_by_tricks_taken = [scoring.score(self.contract, self.is_decl_vuln, n_tricks) for n_tricks in range(14)]

        from ddsolver import ddsolver
        self.dd = ddsolver.DDSolver()

    def init_x_play(self, public_hand, level, strain_i):
        self.x_play = np.zeros((1, 13, 298))
        binary.BinaryInput(self.x_play[:,0,:]).set_player_hand(self.hand)
        binary.BinaryInput(self.x_play[:,0,:]).set_public_hand(public_hand)
        self.x_play[:,0,292] = level
        self.x_play[:,0,293+strain_i] = 1

    def set_card_played(self, trick_i, leader_i, i, card):
        played_to_the_trick_already = (i - leader_i) % 4 > (self.player_i - leader_i) % 4

        if played_to_the_trick_already:
            return

        if self.player_i == i:
            return

        # update the public hand when the public hand played
        if self.player_i in (0, 2, 3) and i == 1 or self.player_i == 1 and i == 3:
            self.x_play[:, trick_i, 32 + card] -= 1

        # update the current trick
        offset = (self.player_i - i) % 4   # 1 = rho, 2 = partner, 3 = lho
        self.x_play[:, trick_i, 192 + (3 - offset) * 32 + card] = 1

    def set_own_card_played52(self, card52):
        self.hand52[card52] -= 1

    def set_public_card_played52(self, card52):
        self.public52[card52] -= 1

    def play_card(self, trick_i, leader_i, current_trick52, players_states):
        current_trick = [deck52.card52to32(c) for c in current_trick52]
        card52_dd = self.next_card52(trick_i, leader_i, current_trick52, players_states)
        card_resp = self.next_card(trick_i, leader_i, current_trick, players_states, card52_dd)

        return card_resp

    def next_card52(self, trick_i, leader_i, current_trick52, players_states):
        from ddsolver import ddsolver
        
        n_samples = players_states[0].shape[0]

        unavailable_cards = set(list(np.nonzero(self.hand52)[0]) + list(np.nonzero(self.public52)[0]) + current_trick52)

        pips = [
            [c for c in range(7, 13) if i*13+c not in unavailable_cards] for i in range(4)
        ]

        symbols = 'AKQJT98765432'

        current_trick_players = [(leader_i + i) % 4 for i in range(len(current_trick52))]

        hands_pbn = []
        for i in range(n_samples):
            hands = [None, None, None, None]
            for j in range(4):
                random.shuffle(pips[j])
            pip_i = [0, 0, 0, 0]

            hands[self.player_i] = deck52.hand_to_str(self.hand52)
            hands[[1,3,1,1][self.player_i]] = deck52.hand_to_str(self.public52)

            for j in range(4):
                if hands[j] is None:
                    suits = []
                    hand32 = players_states[j][i,trick_i,:32].copy().astype(int)

                    # if already played to the trick, subtract the card from the hand
                    if j in current_trick_players:
                        card_of_j = current_trick52[current_trick_players.index(j)]
                        hand32[deck52.card52to32(card_of_j)] -= 1
                    hand_suits = hand32.reshape((4, 8))

                    for suit_i in range(4):
                        suit = []
                        for card_i in np.nonzero(hand_suits[suit_i])[0]:
                            if card_i < 7:
                                if suit_i * 13 + card_i not in current_trick52:
                                    suit.append(card_i)
                            else:
                                for _ in range(hand_suits[suit_i,card_i]):
                                    try:
                                        if pip_i[suit_i] < len(pips[suit_i]):
                                            pip = pips[suit_i][pip_i[suit_i]]
                                            
                                            if suit_i * 13 + pip not in current_trick52:
                                                suit.append(pip)
                                                pip_i[suit_i] += 1
                                    except:
                                        import pdb; pdb.set_trace()
                                    
                        suits.append(''.join([symbols[card] for card in sorted(suit)]))
                    hands[j] = '.'.join(suits)

            hands_pbn.append('W:' + ' '.join(hands))
            if i < 5 and self.verbose:
                print(hands_pbn[-1])

        t_start = time.time()
        
        dd_solved = self.dd.solve(self.strain_i, leader_i, current_trick52, hands_pbn)
        card_tricks = ddsolver.expected_tricks(dd_solved)
        card_ev = self.get_card_ev(dd_solved)

        card_result = {}
        for key in dd_solved.keys():
            card_result[key] = (card_tricks[key], card_ev[key])

        if self.verbose:
            print('dds took', time.time() - t_start)

            print('dd card res', card_result)

        return card_result

    def get_card_ev(self, dd_solved):
        card_ev = {}
        for card, future_tricks in dd_solved.items():
            ev_sum = 0
            for ft in future_tricks:
                if ft < 0:
                    continue
                tot_tricks = self.n_tricks_taken + ft
                tot_decl_tricks = tot_tricks if self.player_i % 2 == 1 else 13 - tot_tricks
                sign = 1 if self.player_i % 2 == 1 else -1
                ev_sum += sign * self.score_by_tricks_taken[tot_decl_tricks]
            card_ev[card] = ev_sum / len(future_tricks)
                
        return card_ev

    def next_card_softmax(self, trick_i):
        return player.follow_suit(
            self.model.next_cards_softmax(self.x_play[:,:(trick_i + 1),:]),
            binary.BinaryInput(self.x_play[:,trick_i,:]).get_player_hand(),
            binary.BinaryInput(self.x_play[:,trick_i,:]).get_this_trick_lead_suit(),
        ).reshape(-1)

    def next_card(self, trick_i, leader_i, current_trick, players_states, card_dd):
        t_start = time.time()
        card_softmax = self.next_card_softmax(trick_i)
        if self.verbose:
            print('ncs', time.time() - t_start)

        all_cards = np.arange(32)
        s_opt = card_softmax > 0.01

        card_options, card_scores = all_cards[s_opt], card_softmax[s_opt]

        card_nn = {c:s for c, s in zip(card_options, card_scores)}
        if self.verbose:
            print(card_nn)

        candidate_cards = []
        
        for card52, (e_tricks, e_score) in card_dd.items():
            card32 = deck52.card52to32(card52)

            candidate_cards.append(CandidateCard(
                card=Card.from_code(card52),
                insta_score=card_nn.get(card32, 0),
                expected_tricks=e_tricks,
                p_make_contract=None,
                expected_score=e_score
            ))

        candidate_cards = sorted(candidate_cards, key=lambda c: (c.expected_score, c.insta_score + random.random() / 10000), reverse=True)

        samples = []
        for i in range(min(20, players_states[0].shape[0])):
            samples.append('%s %s %s %s' % (
                hand_to_str(players_states[0][i,0,:32].astype(int)),
                hand_to_str(players_states[1][i,0,:32].astype(int)),
                hand_to_str(players_states[2][i,0,:32].astype(int)),
                hand_to_str(players_states[3][i,0,:32].astype(int)),
            ))

        card_resp = CardResp(
            card=candidate_cards[0].card,
            candidates=candidate_cards,
            samples=samples
        )

        if self.verbose:
            pprint.pprint(card_resp.to_dict())

        return card_resp
