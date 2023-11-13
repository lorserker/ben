import time
import pprint
import sys
import numpy as np

import binary
import deck52
import scoring

from objects import BidResp, CandidateBid, Card, CardResp, CandidateCard
from bidding import bidding
from sample import Sample
from binary import parse_hand_f

from util import hand_to_str, expected_tricks, p_make_contract, follow_suit


class BotBid:

    def __init__(self, vuln, hand_str, models, ns, ew, sampler, verbose):
        self.vuln = vuln
        self.hand_str = hand_str
        self.hand = binary.parse_hand_f(32)(hand_str)
        self.min_candidate_score = models.search_threshold
        self.max_candidate_score = models.no_search_threshold
        self.model = models.bidder_model
        self.state = models.bidder_model.zero_state
        self.lead_model = models.lead
        self.sd_model = models.sd_model
        self.binfo_model = models.binfo
        self.ns = ns
        self.ew = ew
        self.sample = sampler
        self.verbose = verbose
        self.sample_boards_for_auction = sampler.sample_boards_for_auction

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
        X = binary.get_auction_binary(n_steps, auction, hand_ix, self.hand, self.vuln, self.ns, self.ew)

        x = X[:,-1,:]
        return x

    def bid(self, auction):
        candidates, passout = self.get_bid_candidates(auction)
        if self.verbose:
            print(f"Sampling for aution: {auction}")
        hands_np, p_hcp, p_shp = self.sample_hands(auction)
        samples = []
        for i in range(hands_np.shape[0]):
            samples.append('%s %s %s %s' % (
                hand_to_str(hands_np[i,0,:]),
                hand_to_str(hands_np[i,1,:]),
                hand_to_str(hands_np[i,2,:]),
                hand_to_str(hands_np[i,3,:]),
            ))

        if BotBid.do_rollout(auction, candidates, hands_np, self.max_candidate_score):
            ev_candidates = []
            for candidate in candidates:
                if self.verbose:
                    print(f" {candidate.bid.ljust(4)} {candidate.insta_score:.4f}")
                auctions_np = self.bidding_rollout(auction, candidate.bid, hands_np)
                contracts, decl_tricks_softmax = self.expected_tricks(hands_np, auctions_np)

                for idx, (auction2, (contract, trick)) in enumerate(zip(auctions_np, zip(contracts, decl_tricks_softmax))):
                    auc = bidding.get_action_as_string(auction2)
                    weighted_sum = sum(i * trick[i] for i in range(len(trick)))
                    average_tricks = round(weighted_sum, 1)
                    samples[idx] += " " + auc + " ("+ str(average_tricks) + ") "
                    if self.verbose:
                        print(samples[idx])
    
                ev = self.expected_score(len(auction) % 4, contracts, decl_tricks_softmax)
                expected_score = np.mean(ev)
                #if self.verbose:
                #    print(ev)
                adjust = 0
                # The result is sorted based on the simulation. Adding some bonus to the bid selected by the neural network
                # Should probably be configurable
                if candidate.insta_score < 0.002:
                    adjust += -200
                if candidate.insta_score < 0.0002:
                    adjust += -200
                
                adjust += 50*candidate.insta_score

                # If we are doubling as penalty in the pass out-situation
                # These adjustments should probably be configurable
                if passout and candidate.insta_score < self.min_candidate_score:
                    # If we are bidding in the passout situation, and is going down, assume we are doubled
                    if bidding.BID2ID[candidate.bid] > 4:
                        if expected_score < 0:
                            adjust += expected_score * 3

                if passout and candidate.bid == "X" and candidate.insta_score < self.min_candidate_score:
                    # Don't double unless the expected score is positive with a margin
                    if expected_score < 100:
                        adjust += -200
                    else:
                        adjust += -100
                        
                ev_c = candidate.with_expected_score(np.mean(ev), adjust)
                if self.verbose:
                    print(ev_c)
                ev_candidates.append(ev_c)

            candidates = sorted(ev_candidates, key=lambda c: c.expected_score + c.adjust, reverse=True)
            who = "Simulation"
            # Print candidates with their relevant information
            if self.verbose:
                for idx, candidate in enumerate(candidates, start=1):
                    print(f"{idx}: {candidate.bid.ljust(4)} Insta_score: {candidate.insta_score:.4f} Expected Score: {str(int(candidate.expected_score)).ljust(5)} Adjustment:{str(int(candidate.adjust)).ljust(5)}")
        else:
            who = "NN"
        # Without detailed logging we only save 20 samples
        if not self.verbose:
            samples=samples[:20]
        
        return BidResp(bid=candidates[0].bid, candidates=candidates, samples=samples, shape=p_shp, hcp=p_hcp, who = who)
    
    @staticmethod
    def do_rollout(auction, candidates, samples, max_candidate_score):
        #if candidates[0].insta_score > max_candidate_score:
        #   return False
        #print(candidates)
        # Just one candidate, so no need for rolling out the bidding
        if len(candidates) == 1:
            #print("Just one candidate, so no need for rolling out the bidding")
            return False

        # No samples, so no need for rolling out the bidding
        if len(samples) == 0:
            #print("No samples, so nothing to roll out")
            return False
        
        # If we are past 1st round then we will roll out
        if BotBid.get_n_steps_auction(auction) > 1:
            #print("Past 1st round then we will roll out")
            return True
        
        # If someone has bid before us
        if any(bid not in ('PASS', 'PAD_START') for bid in auction):
            #print("If someone has bid before us")
            return True
        
        # If all bids are real bids - this could result in a missing roll out
        if all(candidate.bid != 'PASS' for candidate in candidates):
            #print("If all bids are real bids - this could result in a missing roll out")
            return True

        return False

    def get_bid_candidates(self, auction):
        bid_softmax = self.next_bid_np(auction)[0]
        if self.verbose:
            index_highest = np.argmax(bid_softmax)
            print(f"bid {bidding.ID2BID[index_highest]} value {bid_softmax[index_highest]:.4f} is recommended by NN")

        candidates = []

        #print("self.min_candidate_score: ",self.min_candidate_score)
        # If self.min_candidate_score == -1 we will just take what the neural network suggest 
        if (self.min_candidate_score == -1):
            while True:
                # We have to loop to avoid returning an invalid bid
                bid_i = np.argmax(bid_softmax)
                if bidding.can_bid(bidding.ID2BID[bid_i], auction):
                    candidates.append(CandidateBid(bid=bidding.ID2BID[bid_i], insta_score=bid_softmax[bid_i]))
                    break
                else:
                    # Only report it if above threshold
                    if bid_softmax[bid_i] >= self.min_candidate_score:
                        # Seems to be an error in the training that needs to be solved
                        sys.stderr.write(f"Bid not valid {bidding.ID2BID[bid_i]} insta_score: {bid_softmax[bid_i]}\n")
                        #assert(bid_i > 1)
                # set the score for the bid just processed to zero so it is out of the loop
                bid_softmax[bid_i] = 0
            return candidates, False
        
        # Find the last index of 'PAD_START'
        pad_start_index = len(auction) - 1 - auction[::-1].index('PAD_START') if 'PAD_START' in auction else -1

        # Calculate the count of elements after the last 'PAD_START'
        no_bids  = len(auction) - pad_start_index - 1
        passout = False
        if no_bids > 3 and auction[-2:] == ['PASS', 'PASS']:
            # this is the final pass, so we wil have a second opinion
            min_candidates = 2
            passout = True
            # If we are doubled trust the bidding model
            if auction[-3:] == ['X','PASS', 'PASS']:
                min_candidates = 1
        else:    
            if no_bids > 12 and auction[-2] != "PASS":
                min_candidates = 2
            else:
                min_candidates = 1

        while True:
            bid_i = np.argmax(bid_softmax)
            #print(bid_i, bid_softmax[bid_i])
            if bid_softmax[bid_i] < self.min_candidate_score:
                if len(candidates) >= min_candidates:
                    break
            if bidding.can_bid(bidding.ID2BID[bid_i], auction):
                candidates.append(CandidateBid(bid=bidding.ID2BID[bid_i], insta_score=bid_softmax[bid_i]))
            else:
                # Seems to be an error in the training that needs to be solved
                # Only report it if above threshold
                if bid_softmax[bid_i] >= self.min_candidate_score:
                    sys.stderr.write(f"Bid not valid: {bidding.ID2BID[bid_i]} insta_score: {bid_softmax[bid_i]}\n")
                if len(candidates) > 0:
                    break
                #assert(bid_i > 1)

            # set the score for the bid just processed to zero so it is out of the loop
            bid_softmax[bid_i] = 0

        # After preempts the model is lacking some bidding, so we will try to get a second bid
        if no_bids < 4 and no_bids > 0:
            if bidding.BID2ID[auction[0]] > 14:
                if self.verbose:
                    print("Extra candidate after opponents preempt might be needed")
                elif no_bids > 1 and bidding.BID2ID[auction[1]] > 14:
                    if self.verbose:
                        print("Extra candidate after partners preempt might be needed")

        if self.verbose:
            print("\n".join(str(bid) for bid in candidates))
        return candidates, passout

    def next_bid_np(self, auction):
        x = self.get_binary(auction)
        bid_np, next_state = self.model.model(x, self.state)
        self.state = next_state

        return bid_np
    
    def sample_hands(self, auction_so_far):
        turn_to_bid = len(auction_so_far) % 4
        lho_pard_rho, p_hcp, p_shp = self.sample.sample_cards_auction(
            auction_so_far, turn_to_bid, self.hand, self.vuln, self.model, self.binfo_model, self.ns, self.ew, self.sample.sample_boards_for_auction)
        # We have more samples, than we want to calculate on
        # They are sorted according to the bidding trust, but above our threshold, so we pick random
        if lho_pard_rho.shape[0] > self.sample.sample_hands_auction:
            random_indices = np.random.permutation(lho_pard_rho.shape[0])
            lho_pard_rho = lho_pard_rho[random_indices[:self.sample.sample_hands_auction], :, :]
        n_samples = lho_pard_rho.shape[0]
        
        hands_np = np.zeros((n_samples, 4, 32), dtype=np.int32)
        hands_np[:,turn_to_bid,:] = self.hand
        for i in range(1, 4):
            hands_np[:, (turn_to_bid + i) % 4, :] = lho_pard_rho[:,i-1,:]

        return hands_np, p_hcp, p_shp 

    def bidding_rollout(self, auction_so_far, candidate_bid, hands_np):
        auction = [*auction_so_far, candidate_bid]
        
        n_samples = hands_np.shape[0]
        assert n_samples > 0
        
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
            X = binary.get_auction_binary(n_steps_vals[turn_i], auction_np, turn_i, hands_np[:,turn_i,:], self.vuln, self.ns, self.ew)
            y_bid_np = self.model.model_seq(X)
            if (y_bid_np.ndim == 2): 
                x_bid_np = y_bid_np.reshape((n_samples, n_steps_vals[turn_i], -1))
                bid_np = x_bid_np[:,-1,:]
            else:
                bid_np = y_bid_np[:,-1,:]
            assert bid_np.shape[1] == 40
            bid_i += 1
            auction_np[:,bid_i] = np.argmax(bid_np, axis=1)
            
            n_steps_vals[turn_i] += 1
            turn_i = (turn_i + 1) % 4
        assert len(auction_np) > 0
        
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
            
            X_ftrs[i,:], B_ftrs[i,:] = binary.get_lead_binary(sample_auction, hand_on_lead, self.binfo_model, self.vuln, self.ns, self.ew)
        
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

    def __init__(self, vuln, hand_str, models, ns, ew, lead_threshold, sample, verbose):
        self.vuln = vuln
        self.hand_str = hand_str
        self.hand = binary.parse_hand_f(32)(hand_str)
        self.hand52 = binary.parse_hand_f(52)(hand_str)

        self.lead_model = models.lead
        self.bidder_model = models.bidder_model
        self.binfo_model = models.binfo
        self.sd_model = models.sd_model
        self.ns = ns
        self.ew = ew
        self.sample = sample
        self.verbose = verbose
        self.lead_threshold = lead_threshold
        self.lead_accept_nn = models.lead_accept_nn

    def find_opening_lead(self, auction):
        lead_card_indexes, lead_softmax = self.get_lead_candidates(auction)
        accepted_samples, tricks, p_hcp, p_shp = self.simulate_outcomes_opening_lead(auction, lead_card_indexes)

        candidate_cards = []
        for i, card_i in enumerate(lead_card_indexes):
            assert(tricks[:,i,0].all() >= 0)
            candidate_cards.append(CandidateCard(
                card=Card.from_code(card_i, xcards=True),
                insta_score=lead_softmax[0,card_i],
                expected_tricks=np.mean(tricks[:,i,0]),
                p_make_contract=np.mean(tricks[:,i,1])
            ))

        print("Sorting by insta_score")
        candidate_cards = sorted(candidate_cards, key=lambda c: c.insta_score, reverse=True)
        print(candidate_cards[0].card)
        print(candidate_cards[0].insta_score)
        if (candidate_cards[0].insta_score > self.lead_accept_nn):
            opening_lead = candidate_cards[0].card.code() 
        else:
            candidate_cards = sorted(candidate_cards, key=lambda c: (round(c.p_make_contract, 2), -round(c.expected_tricks, 2)))
            opening_lead = candidate_cards[0].card.code()

        if opening_lead % 8 == 7:
            # it's a pip ~> choose a random one
            pips_mask = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1])
            lefty_led_pips = self.hand52.reshape((4, 13))[opening_lead // 8] * pips_mask
            opening_lead52 = (opening_lead // 8) * 13 + np.random.choice(np.nonzero(lefty_led_pips)[0])
        else:
            opening_lead52 = deck52.card32to52(opening_lead)

        samples = []
        if self.verbose:
            print(f"Accepted samples for opening lead: {accepted_samples.shape[0]}")
        for i in range(accepted_samples.shape[0]):
            samples.append('%s %s %s' % (
                hand_to_str(accepted_samples[i,0,:]),
                hand_to_str(accepted_samples[i,1,:]),
                hand_to_str(accepted_samples[i,2,:])
            ))

        return CardResp(
            card=Card.from_code(opening_lead52),
            candidates=candidate_cards,
            samples=samples, 
            shape=p_shp,
            hcp=p_hcp
        )

    def get_lead_candidates(self, auction):
        x_ftrs, b_ftrs = binary.get_lead_binary(auction, self.hand, self.binfo_model, self.vuln, self.ns, self.ew)
        lead_softmax = self.lead_model.model(x_ftrs, b_ftrs)
        lead_softmax = follow_suit(lead_softmax, self.hand, np.array([[0, 0, 0, 0]]))

        candidates = []
        # Make a copy of the lead_softmax array
        lead_softmax_copy = np.copy(lead_softmax)

        if self.verbose:
            print("Finding leads from neural network")
        while True:
            c = np.argmax(lead_softmax_copy[0])
            score = lead_softmax_copy[0][c]
            if score < self.lead_threshold:
                break
            lead_softmax_copy[0][c] = 0
            if self.verbose:
                print(Card.from_code(c, xcards=True))
            candidates.append(c)

        return candidates, lead_softmax

    def simulate_outcomes_opening_lead(self, auction, lead_card_indexes):
        contract = bidding.get_contract(auction)

        decl_i = bidding.get_decl_i(contract)
        lead_index = (decl_i + 1) % 4

        accepted_samples, p_hcp, p_shp = self.sample.sample_cards_auction(auction, lead_index, self.hand, self.vuln, self.bidder_model,
                                                                   self.binfo_model, self.ns, self.ew, self.sample.sample_boards_for_auction_opening_lead)

        accepted_samples = accepted_samples[:self.sample.sample_hands_opening_lead]
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

        return accepted_samples, tricks, p_hcp, p_shp


class CardPlayer:

    def __init__(self, player_models, player_i, hand_str, public_hand_str, contract, is_decl_vuln, verbose = False):
        self.player_models = player_models
        self.playermodel = player_models[player_i]
        self.player_i = player_i
        self.hand = parse_hand_f(32)(hand_str).reshape(32)
        self.hand52 = parse_hand_f(52)(hand_str).reshape(52)
        self.public52 = parse_hand_f(52)(public_hand_str).reshape(52)
        self.contract = contract
        self.is_decl_vuln = is_decl_vuln
        self.n_tricks_taken = 0
        self.verbose = verbose
        self.level = int(contract[0])
        self.strain_i = bidding.get_strain_i(contract)

        self.init_x_play(parse_hand_f(32)(public_hand_str), self.level, self.strain_i)

        self.score_by_tricks_taken = [scoring.score(self.contract, self.is_decl_vuln, n_tricks) for n_tricks in range(14)]

        from ddsolver import ddsolver
        self.dd = ddsolver.DDSolver(dds_mode=2)

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
                np.random.shuffle(pips[j])
            pip_i = [0, 0, 0, 0]

            hands[self.player_i] = deck52.deal_to_str(self.hand52)
            hands[[1,3,1,1][self.player_i]] = deck52.deal_to_str(self.public52)

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
            # We always use West as start, but hands are in BEN from LHO
            hands_pbn.append('W:' + ' '.join(hands))
            #if self.verbose:
            #    print(hands_pbn[-1])

        t_start = time.time()

        #if self.verbose:
        #    print(hands_pbn)

        dd_solved = self.dd.solve(self.strain_i, leader_i, current_trick52, hands_pbn)
        card_tricks = ddsolver.expected_tricks(dd_solved)
        card_ev = self.get_card_ev(dd_solved)

        card_result = {}
        for key in dd_solved.keys():
            card_result[key] = (card_tricks[key], card_ev[key])

        if self.verbose:
            print(f'dds took {time.time() - t_start:0.4}')
            for key, value in card_result.items():
                print(f"{deck52.decode_card(key)}: {value}")

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
        cards_softmax = self.playermodel.next_cards_softmax(self.x_play[:,:(trick_i + 1),:])
        assert cards_softmax.shape == (1, 32), f"Expected shape (1, 32), but got shape {cards_softmax.shape}"
        x = follow_suit(
            self.playermodel.next_cards_softmax(self.x_play[:,:(trick_i + 1),:]),
            binary.BinaryInput(self.x_play[:,trick_i,:]).get_player_hand(),
            binary.BinaryInput(self.x_play[:,trick_i,:]).get_this_trick_lead_suit(),
        )
        return x.reshape(-1)
    
    def next_card(self, trick_i, leader_i, current_trick, players_states, card_dd):
        t_start = time.time()
        card_softmax = self.next_card_softmax(trick_i)
        if self.verbose:
            print(f'Next card response time: {time.time() - t_start:0.4f}')

        all_cards = np.arange(32)
        ## This might be a parameter
        s_opt = card_softmax > 0.01

        card_options, card_scores = all_cards[s_opt], card_softmax[s_opt]

        card_nn = {c:s for c, s in zip(card_options, card_scores)}
        #if self.verbose:
        #    print(card_nn)

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

        # This should probably focus more on making the contract, than the score
        # Tricks should take precedence over insta_score
        candidate_cards = sorted(candidate_cards, key=lambda c: (c.expected_score, c.insta_score + np.random.random() / 10000), reverse=True)

        samples = []
        if self.verbose:
            print(f"players_states {players_states[0].shape[0]}")
        for i in range(players_states[0].shape[0]):
            samples.append('%s %s %s %s' % (
                hand_to_str(players_states[0][i,0,:32].astype(int)),
                hand_to_str(players_states[1][i,0,:32].astype(int)),
                hand_to_str(players_states[2][i,0,:32].astype(int)),
                hand_to_str(players_states[3][i,0,:32].astype(int)),
            ))

        #Remove duplicates
        samples = list(set(samples)) 
        
        card_resp = CardResp(
            card=candidate_cards[0].card,
            candidates=candidate_cards,
            samples=samples, # [:20]
            shape=-1,
            hcp=-1
        )

        #if self.verbose:
        #    pprint.pprint(card_resp.to_dict(), width=200)

        return card_resp
