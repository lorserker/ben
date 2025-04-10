import hashlib
import json
import pprint
import time
import sys
import numpy as np
import tensorflow as tf

import binary
import deck52
import calculate
import scoring

from claim import Claimer
from objects import BidResp, CandidateBid, Card, CardResp, CandidateCard
from bidding import bidding
from collections import defaultdict

import carding
from alphamju.alphamju import alphamju
from util import hand_to_str, expected_tricks_sd, p_defeat_contract, follow_suit, calculate_seed, find_vuln_text, save_for_training
from colorama import Fore, Back, Style, init

init()
class BotBid:

    def __init__(self, vuln, hand_str, models, sampler, seat, dealer, ddsolver, bba_is_controlling, verbose):
        self.vuln = vuln
        self.hand_str = hand_str
        self.hand_bidding = binary.parse_hand_f(models.n_cards_bidding)(hand_str)
        self.models = models
        # Perhaps it is an idea to store the auction (and binary version) to speed up processing
        self.min_candidate_score = models.search_threshold
        self.max_candidate_score = models.no_search_threshold
        self.seat = seat
        self.dealer = dealer
        self.sampler = sampler
        self.verbose = verbose
        self.sample_boards_for_auction = sampler.sample_boards_for_auction
        self.samples = []
        self.eval_after_bid_count = models.eval_after_bid_count
        self.sample_hands_for_review = models.sample_hands_for_review
        self.hash_integer = calculate_seed(hand_str)         
        if self.verbose:
            print(f"Setting seed (Sampling bidding info) from {hand_str}: {self.hash_integer}")
        self.rng = self.get_random_generator()
        if self.models.model_version == 0:
            self.state = models.bidder_model.zero_state
        self.ddsolver = ddsolver
        self.my_bid_no = 1
        self._bbabot_instance = None
        self.bba_is_controlling = bba_is_controlling

    @property
    def bbabot(self):
        if self._bbabot_instance is None:
            if self.models.bba_our_cc and self.models.bba_their_cc:
                from bba.BBA import BBABotBid
                # Initialize the BBABotBid instance with required parameters
                self._bbabot_instance = BBABotBid(
                    self.models.bba_our_cc,
                    self.models.bba_their_cc,
                    self.seat,
                    self.hand_str,
                    self.vuln,
                    self.dealer,
                    self.models.matchpoint,
                    self.verbose
                )
        return self._bbabot_instance
    
    def is_bba_controlling(self, bid, explanation):
        if not self.models.consult_bba or self.bbabot is None:
            return False
        
        # We let BBA answer the question
        if bid in self.bbabot.bba_controling and self.bbabot.bba_controling[bid] in explanation:
            return True
        return False
    
    def explain(self, auction):
        if not self.models.consult_bba or self.bbabot is None:
            return None, False
        return self.bbabot.explain_last_bid(auction)

    def explain_auction(self, auction):
        if not self.models.consult_bba or self.bbabot is None:
            return "", False
        return self.bbabot.explain_auction(auction)

    def bid_hand(self, auction, hand):    
        return self.bbabot.bid_hand(auction, hand)
    
    def get_bid_number_for_player_to_bid(self, auction):
        hand_i = len(auction) % 4
        i = hand_i
        while i < len(auction) and auction[i] == 'PAD_START':
            i += 4
        n_step =  1 + (len(auction) - i) // 4
        #print("get_bid_number_for_player_to_bid: ", hand_i, n_step, auction)
        return n_step

    def get_binary(self, auction, models):
        n_steps = self.get_bid_number_for_player_to_bid(auction)
        hand_ix = len(auction) % 4
        X = binary.get_auction_binary(n_steps, auction, hand_ix, self.hand_bidding, self.vuln, models)
        return X

    def get_binary_contract(self, position, vuln, hand_str, dummy_str, n_cards=32):
        X = np.zeros(2 + 2 * n_cards, dtype=np.float16)

        v_we = vuln[0] if position % 2 == 0 else vuln[1]
        v_them = vuln[1] if position % 2 == 0 else vuln[0]
        vuln = np.array([[v_we, v_them]], dtype=np.float16)
        
        hand = binary.parse_hand_f(n_cards)(hand_str).reshape(n_cards)
        dummy = binary.parse_hand_f(n_cards)(dummy_str).reshape(n_cards)
        ftrs = np.concatenate((
            vuln,
            [hand],
            [dummy],
        ), axis=1)
        X = ftrs
        return X
    
    def evaluate_rescue_bid(self, auction, passout, samples, candidate_bid, quality, my_bid_no ):
        # check configuration
        if self.verbose:
            print("Checking if we should evaluate rescue bid", self.models.check_final_contract, "Samples:",len(samples))
            print("Auction",auction, "Passout?" ,passout,"Candidate bid", candidate_bid.bid, "Sample quality: ",quality, "Expected score", candidate_bid.expected_score)
        if not self.models.check_final_contract:
            return False

        if self.models.tf_version == 1:
            sys.stderr.write("Rescue bid not supported for TF 1.x\n")
            return False

        # We did not rescue bid before our 3rd bid     
        # We should look at how the bidding continues, as we can't expect partner to bid
        # Could be a sequence like this 1D-3C-X-4C with A97.AT63.AT8652.
        if my_bid_no < 3:
            return False

        # If no samples we can't evaluate rescue bid
        if len(samples) == 0:
            return False

        # never rescue on first bid
        if binary.get_number_of_bids(auction) < 4:
            return False
        
        # we only try to avoid passing in the wrong contract
        if candidate_bid.bid != "PASS" :
            return False
                
        # If samples of bad quality we do not try to rescue
        if quality <= self.sampler.bid_accept_threshold_bidding:
            return False
        
        # RHO bid, so we will not evaluate rescue bid
        if (auction[-1] != "PASS"):
            return False
        
        # We are in the passout situation
        if passout:
            return True

        if candidate_bid.expected_score is None:
            # We will prepare data for calculating rescue bid
            return True

        # We only evaluate if the score is below a certain value, so if simulation give a score above this we do not try to rescue
        if candidate_bid.expected_score < self.models.max_estimated_score:
            return True

        return False
    
    def get_min_candidate_score(self, bid_no):
        if isinstance(self.min_candidate_score, list):
            # Get the element at index i, or the last element if i is out of bounds
            threshold_value = self.min_candidate_score[bid_no-1] if bid_no < len(self.min_candidate_score) else self.min_candidate_score[-1]
        else:
            # If it's a single float, just use the float value
            threshold_value = self.min_candidate_score

        return threshold_value

    def get_max_candidate_score(self, bid_no):
        if isinstance(self.max_candidate_score, list):
            # Get the element at index i, or the last element if i is out of bounds
            threshold_value = self.max_candidate_score[bid_no] if bid_no < len(self.max_candidate_score) else self.max_candidate_score[-1]
        else:
            # If it's a single float, just use the float value
            threshold_value = self.max_candidate_score

        return threshold_value

    def translate_hands(self, hands_np, hand_str, n_samples):
        hands_pbn = []
        for i in range(n_samples):
            # Create PBN for hand
            # deck = 'N:' + ' '.join(deck52.handxxto52str(hand,self.models.n_cards_bidding) if j != self.seat else hand_str for j, hand in enumerate(hands_np[i]))
            # We want to rotate the hands such that the hand_str comes first, and the remaining hands follow in their original order, wrapping around. 
            # This is to ensure that we get the same DD results for a rotateded deal.
            deck = ' '.join(
                hand_str if j == 0 else deck52.handxxto52str(hands_np[i][(j + self.seat) % 4], self.models.n_cards_bidding)
                for j in range(4)
            )
            deck =deck52.convert_cards(deck,0, hand_str, self.rng, self.models.n_cards_bidding)
            deck = 'N:' + deck52.reorder_hand(deck)
            # Create PBN including pips
            hands_pbn.append(deck)

        return hands_pbn

    def bid(self, auction):
        # Validate input
        if (len(auction)) % 4 != self.seat:
            error_message = f"Dealer {self.dealer}, auction {auction}, and seat {self.seat} do not match!"
            raise ValueError(error_message)

        if self.models.use_bba:
            return self.bbabot.bid(auction)
        # Reseed the rng, so that we get the same result each time in this situation
        # We should perhaps add current auction to the seed
        self.rng = self.get_random_generator()

        self.my_bid_no = self.get_bid_number_for_player_to_bid(auction)
        candidates, passout = self.get_bid_candidates(auction)
        quality = 1
        hands_np = None
        samples = []

        # If no search we will not generate any samples if switch of
        # if only 1 sample we drop sampling, but only if no rescue bidding
        generate_samples = not self.sampler.no_samples_when_no_search and self.get_min_candidate_score(self.my_bid_no) != -1
        generate_samples = generate_samples or (binary.get_number_of_bids(auction) > 4 and self.models.check_final_contract and (passout or auction[-2] != "PASS"))
        generate_samples = generate_samples or len(candidates) > 1

        if generate_samples:
            if self.verbose:
                print(f"Sampling for aution: {auction} trying to find {self.sample_boards_for_auction}")
            hands_np, sorted_score, p_hcp, p_shp, quality = self.sample_hands_for_auction(auction, self.seat)
            for i in range(hands_np.shape[0]):
                deal = '%s %s %s %s - %.5f' % (
                    hand_to_str(hands_np[i,0,:],self.models.n_cards_bidding),
                    hand_to_str(hands_np[i,1,:],self.models.n_cards_bidding),
                    hand_to_str(hands_np[i,2,:],self.models.n_cards_bidding),
                    hand_to_str(hands_np[i,3,:],self.models.n_cards_bidding),
                    sorted_score[i]
                )
                assert len(deal) == 77, f"Expected length of deal to be 77, got {len(deal)} {deal}" 
                samples.append(deal)
            sample_count = hands_np.shape[0]
        else:
            sample_count = 0

        if self.do_rollout(auction, candidates, self.get_max_candidate_score(self.my_bid_no), sample_count):
            ev_candidates = []
            ev_scores = {}
            # we would like to have the same samples including pips for all calculations
            if self.models.double_dummy_calculator:
                hands_np_as_pbn = self.translate_hands(hands_np, self.hand_str, sample_count)
            for candidate in candidates:
                if self.verbose:
                    print(f"Bid: {candidate.bid.ljust(4)} {candidate.insta_score:.3f}")
                auctions_np = self.bidding_rollout(auction, candidate.bid, hands_np, hands_np_as_pbn)

                t_start = time.time()

                # Initialize variables to None
                decl_tricks_softmax1 = None
                decl_tricks_softmax2 = None
                decl_tricks_softmax3 = None
                
                if self.models.double_dummy_calculator:
                    contracts, decl_tricks_softmax1 = self.expected_tricks_dd(hands_np_as_pbn, auctions_np, candidate.bid)
                    ev = self.expected_score(len(auction) % 4, contracts, decl_tricks_softmax1)
                    ev_scores[candidate.bid] = ev
                    decoded_tricks = np.argmax(decl_tricks_softmax1, axis=1)
                if self.models.estimator == "sde" or self.models.estimator == "both":
                    contracts, decl_tricks_softmax2 = self.expected_tricks_sd(hands_np, auctions_np)
                    ev = self.expected_score(len(auction) % 4, contracts, decl_tricks_softmax2)
                    decoded_tricks = np.argmax(decl_tricks_softmax2, axis=1)
                if self.models.estimator == "dde" or self.models.estimator == "both":
                    contracts, decl_tricks_softmax3 = self.expected_tricks_sd_no_lead(hands_np, auctions_np)
                    ev = self.expected_score(len(auction) % 4, contracts, decl_tricks_softmax3)
                    decoded_tricks = np.argmax(decl_tricks_softmax3, axis=1)

                # Filter out None values and prepare for zipping
                trick_lists = [lst for lst in [decl_tricks_softmax1, decl_tricks_softmax2, decl_tricks_softmax3] if lst is not None]
                # Iterate through the auctions and calculate the average tricks
                for idx, (auction2, contract, *trick_lists) in enumerate(zip(auctions_np, contracts, *trick_lists)):
                    auc = bidding.get_auction_as_string(auction2)
                    if contract.lower() != "pass":
                        # Calculate weighted sums and average tricks only for available trick lists
                        average_tricks = []
                        for trick in trick_lists:
                            weighted_sum = sum(i * trick[i] for i in range(len(trick)))
                            average_tricks.append(round(weighted_sum, 1))
                        
                        # Format the average tricks string
                        average_tricks_str = ", ".join(map(str, average_tricks))
                        samples[idx] += f" | {auc} ({average_tricks_str})"
                    else:
                        samples[idx] += f" | {auc}"

                if self.verbose:
                    print("tricks", np.mean(decoded_tricks))
                expected_tricks = np.mean(decoded_tricks)
    
                # We need to find a way to use how good the samples are
                # Calculate the mean of the expected score
                expected_score = np.mean(ev)
                if self.verbose:
                    print("ev",ev)

                adjust = 0

                # If we have really bad scores because we added som extra, reduce the result from those
                if candidate.insta_score < self.models.adjust_min1:
                    adjust -= self.models.adjust_min1_by
                    if self.verbose:
                        print("Adjust for bad scores", adjust)
                if candidate.insta_score < self.models.adjust_min2:
                    adjust -= self.models.adjust_min2_by
                    if self.verbose:
                        print("Adjust for really bad scores", adjust)
                
                if self.verbose:
                    print("Adjust for trust in NN", candidate.bid, candidate.insta_score, candidate.bid[0] in ["5", "6", "7"], "Samples:", hands_np.shape[0])
                if not candidate.bid[0] in ["5", "6", "7"]:
                # Adding some bonus to the bid selected by the neural network
                    if hands_np.shape[0] == self.sampler.min_sample_hands_auction:
                        # We only have the minimum number of samples, so they are often of bad quality
                        # So we add more trust to the NN
                        adjust += self.models.adjust_NN_Few_Samples * candidate.insta_score
                    else:
                        adjust += self.models.adjust_NN * candidate.insta_score

                if self.verbose:
                    print(f"Adjusted for trust in NN {candidate.bid} {adjust:0.3f}")

                if candidate.bid == "X":
                    meaning, alert = self.explain(auction + ["X"])
                    # If we do not have any controls, then adjust the score
                    # If we are void in the suit, then we should not bid X, so we adjust
                    # We also adjust for a singleton
                    if meaning and "penalty" in meaning:
                        trump = bidding.get_strain_i(bidding.get_contract(auction + ["X", "PASS", "PASS", "PASS"]))
                        if trump > 0:
                            #print(self.hand_bidding)
                            reshaped_array = self.hand_bidding.reshape(-1,int(self.models.n_cards_bidding / 4))
                            suits = reshaped_array.sum(axis=1)
                            aces = np.sum(reshaped_array[:, 0] == 1)
                            kings = np.sum(reshaped_array[:, 1] == 1)
                            controls = 2 * aces + kings
                            #print(trump, suits)
                            if suits[trump-1] == 1:
                                adjust -= 0.5 * self.models.adjust_X
                            if suits[trump-1] == 0:
                                adjust -= self.models.adjust_X
                            if controls == 0:
                                adjust -= self.models.adjust_X
                            if controls == 1:
                                adjust -= 0.5 * self.models.adjust_X

                    #print("X=",meaning, alert, auction + ["X"])

                # If we are doubling as penalty in the pass out-situation
                # These adjustments should probably be configurable
                if passout and candidate.insta_score < self.get_min_candidate_score(1):
                    # If we are bidding in the passout situation, and are going down, assume we are doubled
                    if bidding.BID2ID[candidate.bid] > 4:
                        if expected_score < 0:
                            adjust += expected_score * self.models.adjust_passout_negative
                        else:
                            adjust += self.models.adjust_passout
                        if self.verbose:
                            print("Adjust for Passout", adjust)

                # If we are doubling as penalty in the pass out-situation
                    if candidate.bid == "X":
                        #eaning, alert = self.explain(auction)
                        #print("X=",meaning, alert)
                        if self.models.adjust_X_remove > 0:

                            # Sort the dictionary by values (ascending order)
                            sorted_items = sorted(ev)
                            
                            # Determine the number of elements to remove (top X%)
                            n_elements_to_remove = int(len(sorted_items) * self.models.adjust_X_remove // 100)
                            
                            if n_elements_to_remove > 0 and n_elements_to_remove < len(sorted_items):
                                # Keep the lowest (100 - adjust_X_remove) % by slicing the sorted items
                                remaining_items = sorted_items[:-n_elements_to_remove]
                                
                                # Recreate the dictionary with the remaining items
                                ev = remaining_items
                                if self.verbose:
                                    print("Removed optimistic scores", n_elements_to_remove)

                            else:
                                # In case n_elements_to_remove is 0 or out of bounds, just keep the sorted dictionary as is
                                ev = sorted_items


                        # Don't double unless the expected score is positive with a margin
                        # if they are vulnerable
                        # We should probably try to detect if they are sacrificing
                        if self.vuln[(self.seat + 1) % 2]:
                            if expected_score < 200:
                                adjust -= 2 * self.models.adjust_X
                            else:
                                adjust -= self.models.adjust_X
                        else:
                            if expected_score < 100:
                                adjust -= 2 * self.models.adjust_X
                            else:
                                adjust -= self.models.adjust_X
                        if self.verbose:
                            print("Adjusted for double", adjust)

                    if candidate.bid == "XX":
                        # Don't redouble unless the expected score is positive with a margin
                        # if they are vulnerable
                        if self.vuln[(self.seat) % 2]:
                            adjust -= 2 * self.models.adjust_XX
                        else:
                            adjust -= self.models.adjust_XX
                        if self.verbose:
                            print("Adjusted for double", adjust)
                else:
                    # Just a general adjustment of doubles
                    # First round doubles are not included
                    no_bids  = binary.get_number_of_bids(auction) 
                    current_contract = bidding.get_contract(auction)
                    if current_contract == None:
                        current_level = 0
                    else:
                        current_level = int(current_contract[0:1])

                    if candidate.bid == "X" and candidate.insta_score < 0.5 and no_bids > 4:
                        if current_level > 5:
                            adjust -= 2 * self.models.adjust_X
                        else:
                            adjust -= self.models.adjust_X
                        if self.verbose:
                            print("Adjusted for double if insta_score to low", adjust)

                # The problem is that with a low score for X the expected bidding can be very wrong
                if candidate.bid == "X" and candidate.insta_score < 0.1:
                    adjust -= 2*self.models.adjust_X
                    if self.verbose:
                        print("Adjusted for very low score in NN", adjust)


                # Consider adding a penalty for jumping to slam
                # Another options could be to count number of times winning the slam

                if not self.models.use_adjustment:
                    adjust = 0
                    if self.verbose:
                        print("Removed all adjustments", adjust)

                # Calculate the mean of the expected score
                expected_score = np.mean(ev)

                ev_c = candidate.with_expected_score(expected_score, expected_tricks, adjust)
                if self.verbose:
                    print(ev_c)
                ev_candidates.append(ev_c)

            if self.models.use_real_imp_or_mp_bidding and self.models.double_dummy_calculator:
                ev_candidates_mp_imp = []
                #print(candidates)
                #print("ev_scores",ev_scores)
                if self.models.matchpoint:
                    expected_score = calculate.calculate_mp_score(ev_scores)
                    for bid, score in expected_score.items():
                        for candidate in ev_candidates:
                            if candidate.bid == bid:
                                adjust = candidate.adjust / self.models.factor_to_translate_to_mp
                                ev_c = candidate.with_expected_score_mp(score, adjust)
                                #print("ev_c", ev_c)
                                ev_candidates_mp_imp.append(ev_c)
                else:
                    expected_score = calculate.calculate_imp_score(ev_scores)
                    for bid, score in expected_score.items():
                        for candidate in ev_candidates:
                            if candidate.bid == bid:
                                adjust = candidate.adjust / self.models.factor_to_translate_to_imp
                                ev_c = candidate.with_expected_score_imp(score, adjust)
                                ev_candidates_mp_imp.append(ev_c)

                if self.models.matchpoint:
                    if self.verbose:
                        print(f"Sorting for MP {expected_score}")
                    candidates = sorted(ev_candidates_mp_imp, key=lambda c: (c.expected_mp + c.adjust, round(c.insta_score, 2)), reverse=True)
                else:
                    if self.verbose:
                        print(f"Sorting for IMP {expected_score}")
                    candidates = sorted(ev_candidates_mp_imp, key=lambda c: (c.expected_imp + c.adjust, round(c.insta_score, 2)), reverse=True)
                ev_candidates = ev_candidates_mp_imp
            else:
                # If the samples are bad we just trust the neural network
                if self.models.use_biddingquality  and quality < self.sampler.bidding_threshold_sampling:
                    if self.verbose:
                        print(f"Bidding quality to bad, so we select using NN {quality} - {self.sampler.bidding_threshold_sampling}")
                    candidates = sorted(ev_candidates, key=lambda c: (c.insta_score, c.expected_score + c.adjust), reverse=True)
                else:
                    candidates = sorted(ev_candidates, key=lambda c: (c.expected_score + c.adjust, round(c.insta_score, 2)), reverse=True)
            
            who = "Simulation"
            # Print candidates with their relevant information
            if self.verbose:
                for idx, candidate in enumerate(ev_candidates):
                    print(f"{idx}: {candidate}")
                print(f"Estimating took {(time.time() - t_start):0.4f} seconds")
        else:
            n_steps = binary.calculate_step_bidding_info(auction)
            p_hcp, p_shp = self.sampler.get_bidding_info(n_steps, auction, self.seat, self.hand_bidding, self.vuln, self.models)
            p_hcp = p_hcp[0]
            p_shp = p_shp[0]
            if sample_count == 0 and generate_samples:
                if self.models.consult_bba:
                    bid_resp = self.bbabot.bid(auction)
                    found = False
                    for i, candidate in enumerate(candidates):
                        if candidate.bid == bid_resp.bid:
                            if self.verbose:
                                print(f"BBA bid {bid_resp.bid} is in candidates")
                            # Move the found candidate to the first position
                            candidates.insert(0, candidates.pop(i))
                            found = True
                            break
                    if not found:
                        if self.verbose:
                            print(f"Adding BBA bid {bid_resp.bid} to candidates")
                        candidates.insert(0, CandidateBid(bid=bid_resp.bid, insta_score=-1, alert = True, who="BBA", explanation=bid_resp.explanation))
                
            who = "NN" if candidates[0].who is None else candidates[0].who
            if self.evaluate_rescue_bid(auction, passout, samples, candidates[0], quality, self.my_bid_no):    
                if self.verbose:
                    print("Updating samples with expected score")    
                # initialize auction vector
                auction_np = np.ones((len(samples), 64), dtype=np.int32) * bidding.BID2ID['PAD_END']
                for i, bid in enumerate(auction):
                    auction_np[:,i] = bidding.BID2ID[bid]

                hands_np_as_pbn = self.translate_hands(hands_np, self.hand_str, sample_count)
                contracts, decl_tricks_softmax = self.expected_tricks_dd(hands_np_as_pbn, auction_np)
                decoded_tricks = np.argmax(decl_tricks_softmax, axis=1)
                if self.verbose:
                    print("tricks", np.mean(decoded_tricks))
                expected_tricks = np.mean(decoded_tricks)
                # We need to find a way to use how good the samples are
                ev = self.expected_score(len(auction) % 4, contracts, decl_tricks_softmax)
                # Calculate the mean of the expected score
                expected_score = np.mean(ev)
                candidates[0] = candidates[0].with_expected_score(expected_score, expected_tricks, 0)


        if self.verbose:
            print(f"{Fore.LIGHTCYAN_EX}{candidates[0].bid} selected by {who} {'and sampling' if sample_count > 0 else ''}{Fore.RESET}")

        if self.evaluate_rescue_bid(auction, passout, samples, candidates[0], quality, self.my_bid_no) and not candidates[0].who == "BBA":    

            # We will avoid rescuing if we have a score of max_estimated_score or more
            t_start = time.time()
            alternatives = {}
            current_contract = bidding.get_contract(auction)
            current_contract = current_contract[0:2]
            if self.verbose:
                print("check_final_contract, current_contract:", current_contract, " Samples:", len(samples))
            # We should probably select random form the samples
            break_outer = False
            samples_to_check = self.rng.choice(samples, min(len(samples), self.models.max_samples_checked), replace=False)

            for sample in samples_to_check:
                sample = sample.split(" ")
                if self.verbose:
                    #print(samples[i].split(" ")[(self.seat + 2) % 4])
                    print(sample[(self.seat + 2) % 4], sample[5])
                if float(sample[5]) < self.models.min_bidding_trust_for_sample_when_rescue:
                    if self.verbose: 
                        print("Skipping sample due to threshold", self.models.min_bidding_trust_for_sample_when_rescue)
                    continue
                X = self.get_binary_contract(self.seat, self.vuln, self.hand_str, sample[(self.seat + 2) % 4], self.models.n_cards_bidding)
                # Perhaps we should collect all samples, and just make one call to the neural network
                contracts = self.models.contract_model.pred_fun(X)
                if tf.is_tensor(contracts):
                    contracts = contracts.numpy()
                score = 0
                result = {}
                for i in range(len(contracts[0])):
                    # We should make calculations on this, so 4H, %h or even 6H is added, if tricks are fin
                    if contracts[0][i] > 0.2:
                        y = np.zeros(5)
                        suit = bidding.ID2BID[i][1]
                        strain_i = 'NSHDC'.index(suit)
                        y[strain_i] = 1
                        Xt = [np.concatenate((X[0], y), axis=0)]
                        nn_tricks = self.models.trick_model.pred_fun(Xt)
                        nn_tricks = nn_tricks.numpy()
                        max_tricks = None
                        for j in range(14):
                            trick_score = 0
                            if nn_tricks[0][j] > 0.2:
                                if bidding.ID2BID[i] in result:
                                    # Append new data to the existing entry
                                    result[bidding.ID2BID[i]]["Tricks"].append(j)
                                    result[bidding.ID2BID[i]]["Percentage"].append(round(float(nn_tricks[0][j]), 2))
                                else:
                                    # Create a new entry
                                    result[bidding.ID2BID[i]] = {
                                        "score": round(float(contracts[0][i]), 2),
                                        "Tricks": [j],
                                        "Percentage": [round(float(nn_tricks[0][j]), 2)]
                                    }   
                                if nn_tricks[0][j] > trick_score:
                                    trick_score = nn_tricks[0][j]
                                    max_tricks = j
                        if contracts[0][i] > score and not max_tricks is None:
                            score = contracts[0][i]
                            contract_id = i
                            tricks = max_tricks
                            contract = bidding.ID2BID[contract_id] 



                if score < self.models.min_bidding_trust_for_sample_when_rescue:
                    #if self.verbose:
                    #    print(self.hand_str, [sample[(self.seat + 2) % 4]])
                    #    if score == 0:
                    #        print(f"No obvious rescue contract")
                    #    else:
                    #        print(f"Skipping sample below level: {self.models.min_bidding_trust_for_sample_when_rescue} {contract} {tricks} score {score:.3f}")
                    continue

                if self.verbose:
                    print(result)                    

                while not bidding.can_bid(contract, auction) and contract_id < 35:
                    contract_id += 5
                    contract = bidding.ID2BID[contract_id] 
                    
                # If game bid in major do not bid 5 of that major 
                if current_contract == "4H" and contract == "5H":
                    if self.verbose:
                        print("Stopping rescue, just one level higher")
                    alternatives = {}
                    break
                if current_contract == "4S" and contract == "5S":
                    if self.verbose:
                        print("Stopping rescue, just one level higher")
                    alternatives = {}
                    break
                # If 3N don't bid 4N
                if current_contract == "3N" and (contract == "4N" or contract == "5N" or contract == "6N"):
                    if self.verbose:
                        print("Stopping rescue, just one level higher")
                    alternatives = {}
                    break
                # If 6N don't bid 7N
                if current_contract == "6N" and (contract == "7N"):
                    if self.verbose:
                        print("Stopping rescue, just one level higher")
                    alternatives = {}
                    break
                if current_contract == contract:
                    if self.verbose:
                        print("Contract bid, stopping rescue")
                    alternatives = {}
                    break
                
                        
                # if the contract is in candidates we assume previous calculations are right and we stop
                for c in candidates:
                    if c.bid == contract:
                        if self.verbose:
                            print("Contract found in candidates, stopping rescue")
                        alternatives = {}
                        break_outer = True
                        break

                if break_outer:
                    break
                # Skip invalid bids
                if bidding.can_bid(contract, auction):
                    result = {"contract": contract, "tricks": tricks}
                    level = int(contract[0])
                    # If we go down we assume we are doubled
                    doubled = tricks < level + 6
                    score = scoring.score(contract + ("X" if doubled else ""), self.vuln[(self.seat + 1) % 2], tricks)
                    if self.verbose:
                        print(result, score, level, doubled, self.vuln[(self.seat + 1) % 2] )
                    if contract not in alternatives:
                        alternatives[contract] = []
                    alternatives[contract].append({"score": score, "tricks": tricks})
                    
            # Only if at least 75% of the samples suggest bidding check the score for the rescue bid
            # print(len(alternatives), min(len(samples), self.models.max_samples_checked))
            total_entries = sum(len(entries) for entries in alternatives.values())
            if total_entries > 0.75 * min(len(samples), self.models.max_samples_checked):
                # Initialize dictionaries to store counts and total scores
                contract_counts = defaultdict(int)
                contract_total_scores = defaultdict(int)
                contract_total_tricks = defaultdict(int)

                # Iterate through the alternatives dictionary to populate counts and total scores
                for contract, entries in alternatives.items():
                    for entry in entries:
                        score = entry["score"]
                        
                        contract_counts[contract] += 1
                        contract_total_scores[contract] += score
                        contract_total_tricks[contract] += entry["tricks"]

                # Calculate the average scores
                contract_average_scores = {contract: round(contract_total_scores[contract] / contract_counts[contract])
                                        for contract in contract_counts}
                contract_average_tricks = {contract: round(contract_total_tricks[contract] / contract_counts[contract],2)
                                        for contract in contract_counts}

                # Print the results
                if self.verbose:
                    print("Contract Counts:", dict(contract_counts))
                    print("Contract Average Scores:", contract_average_scores)
                    print("Contract Average tricks:", contract_average_tricks)
                # Find the contract with the highest count
                max_count_contract = max(contract_counts, key=contract_counts.get)
                # Unless we gain 300 or we expect 4 tricks more we will not override BEN

                if (contract_average_scores[max_count_contract] > candidates[0].expected_score + self.models.min_rescue_reward) or (contract_average_tricks[max_count_contract] - expected_tricks > 4):
                    # Now we have found a possible resuce bid, so we need to check the samples with that contract
                    if self.verbose:
                        print("Evaluating", max_count_contract)
                    no_hands = len(hands_np_as_pbn[:self.models.max_samples_checked])
                    new_auction = auction.copy()
                    new_auction.append(max_count_contract)
                    auction_np = np.ones((no_hands, 64), dtype=np.int32) * bidding.BID2ID['PAD_END']
                    for i, bid in enumerate(new_auction):
                        auction_np[:,i] = bidding.BID2ID[bid]

                    # Calculate score for the hands
                    contracts, decl_tricks_softmax = self.expected_tricks_dd(hands_np_as_pbn[:self.models.max_samples_checked], auction_np)
                    decoded_tricks = np.argmax(decl_tricks_softmax, axis=1)
                    if self.verbose:
                        print("tricks", np.mean(decoded_tricks))
                    expected_tricks = np.mean(decoded_tricks)
                    # We need to find a way to use how good the samples are
                    # Assume we are doubled if going down

                    ev = self.expected_score(len(auction) % 4, contracts, decl_tricks_softmax)
                    evd= self.expected_score_doubled(len(auction) % 4, contracts, decl_tricks_softmax)
                    # Calculate the mean of the expected score and expected score doubled
                    expected_score = np.mean(ev)
                    expected_score_doubled = np.mean(evd)
                    # Take the lowest score of the two
                    if expected_score_doubled < expected_score:
                        expected_score = expected_score_doubled
                    if self.verbose:
                        print("expected_score", expected_score)
                    if (expected_score > candidates[0].expected_score + self.models.min_rescue_reward) or (contract_average_tricks[max_count_contract] - expected_tricks > 4):

                        candidatebid = CandidateBid(bid=max_count_contract, insta_score=-1, 
                                                    expected_score=contract_average_scores[max_count_contract], expected_tricks=expected_tricks, adjust=0, alert = False, who="Rescue")
                        candidates.insert(0, candidatebid)
                        who = "Rescue"
                        sys.stderr.write(f"Rescuing {current_contract} {contract_counts[max_count_contract]}*{max_count_contract} {contract_average_scores[max_count_contract]:.3f} {contract_average_tricks[max_count_contract]:.2f}\n")
            else:
                if self.verbose:
                    print("No rescue, due to not enough samples, that suggest bidding: ", total_entries, len(samples), self. models.max_samples_checked)
            if self.verbose:
                print(f"Rescue bid calculation took {(time.time() - t_start):0.4f} seconds")

        else:
            if self.verbose:
                print("No rescue bid evaluated")

        # We return the bid with the highest expected score or highest adjusted score 
        return BidResp(bid=candidates[0].bid, candidates=candidates, samples=samples[:self.sample_hands_for_review], shape=p_shp, hcp=p_hcp, who=who, quality=quality, alert = bool(candidates[0].alert), explanation=candidates[0].explanation)
    
    def do_rollout(self, auction, candidates, max_candidate_score, sample_count):
        if candidates[0].insta_score > max_candidate_score:
            if self.verbose:
                print(f"A candidate above threshold {max_candidate_score}, so no need for rolling out the bidding")
            return False
        
        # Perhaps we should have another threshold for Double and Redouble as when that is suggested by NN, it is probably well defined

        # Just one candidate, so no need for rolling out the bidding
        if len(candidates) == 1:
            if self.verbose:
                print("Just one candidate, so no need for rolling out the bidding")
            return False
        
        # Do not try to simulate if opening
        if bidding.get_contract(auction) == None:
            if self.verbose:
                print("Only simulate opening bid if enabled in configuration")
            return self.models.eval_opening_bid

        if sample_count == 0:
            if self.verbose:
                print("We found no samples, so will just trust the NN or BBA")
            return False
        return True
    

    def get_bid_candidates(self, auction):
        if self.verbose:
            print("Getting bid candidates")
        explanation = None
        if self.models.consult_bba:
            explanation, alert = self.bbabot.explain_last_bid(auction[:-1])
            bba_bid_resp = self.bbabot.bid(auction)
            if self.bba_is_controlling:
                return [CandidateBid(bid=bba_bid_resp.bid, insta_score=1, alert = True, who="BBA - Keycard sequence", explanation=bba_bid_resp.explanation)], False

            if self.models.use_bba_to_count_aces and self.bbabot.is_key_card_ask(auction, explanation):
                if self.verbose:
                    print("Keycards: ", bba_bid_resp)
                self.bba_is_controlling = True
                return [CandidateBid(bid=bba_bid_resp.bid, insta_score=1, alert = True, who="BBA - Keycard response", explanation=bba_bid_resp.explanation)], False

        bid_softmax, alerts = self.next_bid_np(auction)

        #print(bid_softmax, alerts)
    
        if self.verbose:
            index_highest = np.argmax(bid_softmax)
            if self.models.alert_supported and alerts[0] > self.models.alert_threshold:
                print(f"bid {bidding.ID2BID[index_highest]} value {bid_softmax[index_highest]:.4f} is recommended by NN with alert {alerts[0]:.2f}")
            else:
                print(f"bid {bidding.ID2BID[index_highest]} value {bid_softmax[index_highest]:.4f} is recommended by NN")

        candidates = []

        # If self.min_candidate_score == -1 we will just take what the neural network suggest 
        if (self.get_min_candidate_score(self.my_bid_no) == -1):
            while True:
                # We have to loop to avoid returning an invalid bid
                bid_i = np.argmax(bid_softmax)
                if self.models.alert_supported:
                    alert = alerts[0]  > self.models.alert_threshold
                else:
                    alert = None
                #print("bid_i",bid_i)
                #print(bid_softmax)
                if bidding.can_bid(bidding.ID2BID[bid_i], auction):
                    if not self.models.suppress_warnings:
                        if (bid_softmax[bid_i] < 0.5):
                            sys.stderr.write(f"{Fore.YELLOW}Consider adding samples for {'-'.join(auction).replace('PASS', 'P').replace('PAD_START-', '')} with {self.hand_str}\n{Style.RESET_ALL}")
                    candidates.append(CandidateBid(bid=bidding.ID2BID[bid_i], insta_score=bid_softmax[bid_i], alert=alert))
                    break
                else:
                    # Only report it if above threshold
                    if not self.models.suppress_warnings:
                        if bid_softmax[bid_i] >= self.get_min_candidate_score(self.my_bid_no):
                            # Seems to be an error in the training that needs to be solved
                            sys.stderr.write(f"{Fore.GREEN}Please create samples for {'-'.join(auction).replace('PASS', 'P').replace('PAD_START-', '')}\n{Style.RESET_ALL}")
                            sys.stderr.write(f"{Fore.GREEN}Hand {self.hand_str}\n{Style.RESET_ALL}")
                            sys.stderr.write(f"Bid not valid {bidding.ID2BID[bid_i]} insta_score: {bid_softmax[bid_i]}\n")
                        
                        #assert(bid_i > 1)
                # set the score for the bid just processed to zero so it is out of the loop
                bid_softmax[bid_i] = 0
            return candidates, False
        
        no_bids  = binary.get_number_of_bids(auction) 
        passout = False
        if no_bids > 3 and auction[-2:] == ['PASS', 'PASS']:
            # this is the final pass, so we wil have a second opinion
            min_candidates = self.models.min_passout_candidates
            passout = True
            # If we are doubled trust the bidding model
            # This is not good if we are doubled in a splinter
            # If we are doubled in pass out situation, never raise the suit
            
        else:    
            # Perhaps this only should be checked if we are active in the bidding
            if (no_bids > self.eval_after_bid_count) and (self.eval_after_bid_count > 0):
                min_candidates = self.models.min_passout_candidates
            else:
                min_candidates = 1

        # After preempts the model is lacking some bidding, so we will try to get a second bid
        if no_bids < 4 and no_bids > 0:
            if bidding.BID2ID[auction[-1]] > 14:
                min_candidates = 2
                if self.verbose:
                    print("Extra candidate after opponents preempt might be needed")
                elif no_bids > 1 and bidding.BID2ID[auction[-2]] > 14:
                    if self.verbose:
                        print("Extra candidate might be needed after partners preempt/bid over preempt")

        while True:
            bid_i = np.argmax(bid_softmax)
            if bid_softmax[bid_i] < self.get_min_candidate_score(self.my_bid_no):
                if len(candidates) >= min_candidates:
                    break
                # Second candidate to low. Rescuebid should handle 
                if bid_softmax[bid_i] <= 0.005:
                    break
            if bidding.can_bid(bidding.ID2BID[bid_i], auction):
                if self.models.alert_supported:
                    alert = alerts[0]  > self.models.alert_threshold
                else:
                    alert = None
                candidates.append(CandidateBid(bid=bidding.ID2BID[bid_i], insta_score=bid_softmax[bid_i], alert = alert))
            else:
                # Seems to be an error in the training that needs to be solved
                # Only report it if above threshold
                if not self.models.suppress_warnings:
                    if bid_softmax[bid_i] >= self.get_min_candidate_score(self.my_bid_no) and self.get_min_candidate_score(self.my_bid_no) != -1:
                        sys.stderr.write(f"{Fore.GREEN}Please create samples for {'-'.join(auction).replace('PASS', 'P').replace('PAD_START-', '')}\n{Style.RESET_ALL}")
                        sys.stderr.write(f"{Fore.GREEN}Hand {self.hand_str}\n{Style.RESET_ALL}")
                        sys.stderr.write(f"Bid not valid: {bidding.ID2BID[bid_i]} insta_score: {bid_softmax[bid_i]:.3f} {self.get_min_candidate_score(self.my_bid_no)}\n")
                if len(candidates) > 0:
                    break

            # set the score for the bid just processed to zero so it is out of the loop
            bid_softmax[bid_i] = 0

        # Adding an PASS, as a possible bid, even if it is not suggested by the nn
        if self.models.eval_pass_after_bid_count > 0 and no_bids > self.models.eval_pass_after_bid_count:
            for candidate in candidates:
                if candidate.bid == 'PASS':
                    break
            else:
                candidates.append(CandidateBid(bid=bidding.ID2BID[2], insta_score=0.1, alert = None, Who = "Config"))

        if self.verbose:
            print("\n".join(str(bid) for bid in candidates))
    
        if self.models.consult_bba:
            if self.verbose:
                print("BBA suggests: ", bba_bid_resp.bid)
            # If BBA suggest Pass, then we should probably not double
            if explanation:
                if "Forcing" in explanation:
                    for candidate in candidates:
                        if candidate.bid == "PASS":
                            candidate.explanation = "We are not allowed to pass"
                            candidate.insta_score = -1

            for candidate in candidates:
                if candidate.bid == bba_bid_resp.bid:
                    candidate.alert = bba_bid_resp.alert
                    candidate.explanation = bba_bid_resp.explanation
                    break
            else:
                if self.verbose:
                    sys.stderr.write(f"{Fore.CYAN}Adding BBA bid as candidate: {bba_bid_resp.bid} Alert: { bba_bid_resp.alert} Explaination: {bba_bid_resp.explanation}{Fore.RESET}\n")
                candidates.append(CandidateBid(bid=bba_bid_resp.bid, insta_score=0.2, alert = bba_bid_resp.alert, who="BBA", explanation=bba_bid_resp.explanation))

        return candidates, passout

    def next_bid_np(self, auction):
        alerts = None
        if self.verbose:
            print("next_bid_np: Model:", self.models.name, "Version:", self.models.model_version, "NS:", self.models.ns, "Alert supported:", self.models.alert_supported)
        x = self.get_binary(auction, self.models)
        if self.models.model_version == 0:
            if self.models.ns != -1:
                print("Different models for NS and EW not supported in this version")
            x = x[:,-1,:]
            bid_np, next_state = self.models.bidder_model.pred_fun(x, self.state)
            bid_np = bid_np[0]
            self.state = next_state
        if self.models.model_version == 1 :
            bid_np = self.models.bidder_model.pred_fun_seq(x)
            if self.models.alert_supported:
                alerts = bid_np[1][-1:][0]                   
            bid_np = bid_np[0][-1:][0]
        if self.models.model_version == 2:
            bid_np = self.models.bidder_model.pred_fun_seq(x)
            if self.models.alert_supported:
                alerts = bid_np[1][-1:][0]                   
            bid_np = bid_np[0][-1:][0]
        if self.models.model_version == 3:
            bid_np, alerts = self.models.bidder_model.pred_fun_seq(x)
            if self.models.alert_supported:
                alerts = alerts[0]
                alerts = alerts[-1:][0]
            bid_np = bid_np[0]
            bid_np = bid_np[-1:][0]
        assert len(bid_np) == 40, "Wrong Result: " + str(bid_np.shape)
        return bid_np, alerts
    
    def sample_hands_for_auction(self, auction_so_far, turn_to_bid):
        # Reset randomizer
        self.rng = self.get_random_generator()

        aceking = {}
        if self.models.use_bba_to_count_aces:
            if self.bbabot is not None:
                aceking = self.bbabot.find_aces(auction_so_far)


        accepted_samples, sorted_scores, p_hcp, p_shp, quality, samplings = self.sampler.generate_samples_iterative(auction_so_far, turn_to_bid, self.sampler.sample_boards_for_auction, self.sampler.sample_hands_auction, self.rng, self.hand_str, self.vuln, self.models, [], aceking)

        # We have more samples, than we want to calculate on
        # They are sorted according to the bidding trust, but above our threshold, so we pick random
        if accepted_samples.shape[0] >= self.sampler.sample_hands_auction:
            random_indices = self.rng.permutation(accepted_samples.shape[0])
            accepted_samples = accepted_samples[random_indices[:self.sampler.sample_hands_auction], :, :]
            sorted_scores = sorted_scores[random_indices[:self.sampler.sample_hands_auction]]
        else:
            # Inform user
            if not self.models.suppress_warnings:
                if accepted_samples.shape[0] <= self.sampler.warn_to_few_samples:
                    sys.stderr.write(f"{Fore.YELLOW}Warning: Not enough samples found. Using all samples {accepted_samples.shape[0]}, Samplings={samplings}, Auction={auction_so_far}, Quality={quality:.2f}{Style.RESET_ALL}\n")

        sorted_indices = np.argsort(sorted_scores)[::-1]
        # Extract scores based on the sorted indices
        sorted_scores = sorted_scores[sorted_indices]
        accepted_samples = accepted_samples[sorted_indices]

        n_samples = accepted_samples.shape[0]
        
        hands_np = np.zeros((n_samples, 4, self.models.n_cards_bidding), dtype=np.int32)
        hands_np[:,turn_to_bid,:] = self.hand_bidding
        for i in range(1, 4):
            hands_np[:, (turn_to_bid + i) % 4, :] = accepted_samples[:,i-1,:]

        return hands_np, sorted_scores, p_hcp, p_shp, quality 

    def get_random_generator(self):
        #print(f"{Fore.BLUE}Fetching random generator for bid {self.hash_integer}{Style.RESET_ALL}")
        return np.random.default_rng(self.hash_integer)

    def bidding_rollout(self, auction_so_far, candidate_bid, hands_np, hands_np_as_pbn):
        t_start = time.time()
        n_samples = hands_np.shape[0]
        if self.verbose:
            print("bidding_rollout - n_samples: ", n_samples)
        assert n_samples > 0
        auction = [*auction_so_far, candidate_bid]
        auction_np = np.ones((n_samples, 64), dtype=np.int32) * bidding.BID2ID['PAD_END']
        for i, bid in enumerate(auction):
            auction_np[:,i] = bidding.BID2ID[bid]
        if self.models.use_bba_rollout: 
            for i in range(hands_np.shape[0]):
                bba_auction = self.bbabot.bid_hand(auction, hands_np_as_pbn[i])
                for j, bid in enumerate(bba_auction):
                    auction_np[i,j] = bidding.BID2ID[bid]

        else:
            auction_length = len(auction)
            n_steps_vals = [0, 0, 0, 0]
            for i in range(1, 5):
                n_steps_vals[(len(auction_so_far) % 4 + i) % 4] = self.get_bid_number_for_player_to_bid(auction_so_far + ['?'] * i)  
            
            # initialize auction vector

            bid_i = 0 
            turn_i = auction_length % 4

            #print(hand_to_str(hands_np[0,turn_i,:], self.models.n_cards_bidding))
            
            #X = binary.get_auction_binary_sampling(n_steps_vals[turn_i], auction_np, turn_i, hands_np[:,turn_i,:], self.vuln, self.models, self.models.n_cards_bidding)
            #print(X[0])
            #print("turn_i", turn_i)
            # Now we bid each sample to end of auction
            while not np.all(auction_np[:,auction_length -1 + bid_i] == bidding.BID2ID['PAD_END']):
                #print("bidding_rollout - n_steps_vals: ", n_steps_vals, " turn_i: ", turn_i, " bid_i: ", bid_i, " auction: ", auction)
                X = binary.get_auction_binary_sampling(n_steps_vals[turn_i], auction_np, turn_i, hands_np[:,turn_i,:], self.vuln, self.models, self.models.n_cards_bidding)
                if bid_i % 2 == 0:
                    x_bid_np, _ = self.models.opponent_model.pred_fun_seq(X)
                else:
                    x_bid_np, _ = self.models.bidder_model.pred_fun_seq(X)
                
                if self.models.model_version < 3:
                    x_bid_np = x_bid_np.reshape((n_samples, n_steps_vals[turn_i], -1))
                    
                bid_np = x_bid_np[:,-1,:]
                assert bid_np.shape[1] == 40
                # We can get invalid bids back from the neural network, so we need to filter those away first
                invalid_bids = True
                while invalid_bids:
                    invalid_bids = False
                    for i in range(n_samples):
                        auction = bidding.get_auction_as_list(auction_np[i])
                        if not bidding.auction_over(auction):
                            bid = np.argmax(bid_np[i])
                            # if Pass returned after the bidding really is over
                            # print("bid_np[i][bid]: ", bid_np[i][bid])
                            if (bid == 2 and bidding.auction_over(auction)):
                                sys.stderr.write(str(auction))
                                sys.stderr.write(f" Pass not valid as auction is over: {bidding.ID2BID[bid]} insta_score: {bid_np[i][bid]:.3f}\n")
                                bid_np[i][1] = 1
                            # Pass is always allowed
                            if (bid > 2 and not bidding.can_bid(bidding.ID2BID[bid], auction)):
                                invalid_bids = True
                                deal = ' '.join(deck52.handxxto52str(hand,self.models.n_cards_bidding) for hand in hands_np[i,:,:])
                                deal = deck52.convert_cards(deal,0, "", self.rng, self.models.n_cards_bidding)
                                deal = deck52.reorder_hand(deal)
                                deal = "NESW"[self.dealer] + " " +find_vuln_text(self.vuln) + ' ' + deal
                                if not self.models.suppress_warnings:
                                    save_for_training(deal, '-'.join(auction).replace('PASS', 'P'))
                                    sys.stderr.write(f"{Fore.RED}Sampling: {'-'.join(auction_so_far).replace('PASS', 'P').replace('PAD_START-', '').replace('PAD_START', 'Opening')} with this deal {deal} ")
                                    sys.stderr.write(f"to avoid this auction {'-'.join(auction).replace('PASS', 'P')}\n{Style.RESET_ALL}")
                                    sys.stderr.write(f"Sample: {i}, Hand {hand_to_str(hands_np[i,turn_i,:], self.models.n_cards_bidding)} Bid not valid: {bidding.ID2BID[bid]} insta_score: {bid_np[i][bid]:.3f}\n")
                                bid_np[i][bid] = 0
                                

                            assert auction_length - 1 + bid_i <= 60, f'Auction to long {bid_i} {auction} {auction_np[i]}'
                        else:
                            bid_np[i][1] = 1

                #print("Adding", bidding.ID2BID[np.argmax(bid_np, axis=1)])
                #print("Adding", np.argmax(bid_np, axis=1))
                #print(bid_np)
                auction_np[:,auction_length + bid_i] = np.argmax(bid_np, axis=1)
                bid_i += 1
                n_steps_vals[turn_i] += 1
                turn_i = (turn_i + 1) % 4
            assert len(auction_np) > 0
        
        if self.verbose:
            print(f"bidding_rollout {candidate_bid} - finished {(time.time() - t_start):0.4f}",auction_np.shape)
        
        return auction_np
    
    def transform_hand(self, hand):
        handplay = binary.parse_hand_f(self.models.n_cards_play)(deck52.handxxto52str(hand, self.models.n_cards_bidding))            
        return handplay[0]


    def expected_tricks_sd(self, hands_np, auctions_np):
        n_samples = hands_np.shape[0]

        s_all = np.arange(n_samples, dtype=np.int32)
        auctions, contracts = [], []
        declarers = np.zeros(n_samples, dtype=np.int32)
        strains = np.zeros(n_samples, dtype=np.int32)
        X_ftrs = np.zeros((n_samples, 10 +  self.models.n_cards_play))
        B_ftrs = np.zeros((n_samples, 15))
        
# Apply the function to each 1D array along the third dimension
        hands_np_play = hands_np if self.models.n_cards_bidding == self.models.n_cards_play else  np.apply_along_axis(self.transform_hand, 2, hands_np)                


        for i in range(n_samples):
            sample_auction = [bidding.ID2BID[bid_i] for bid_i in list(auctions_np[i, :]) if bid_i != 1]
            auctions.append(sample_auction)

            contract = bidding.get_contract(sample_auction)
            # All pass doesn't really fit, and is always 0 - we ignore it for now
            if contract is None:
                contracts.append("PASS")
                strains[i] = -1
                declarers[i] = -1
            else:
                contracts.append(contract)
                strains[i] = 'NSHDC'.index(contract[1])
                declarers[i] = 'NESW'.index(contract[-1])
            
                # The NN for this requires a hand from the 32 card deck
                handbidding = hands_np[i:i+1, (declarers[i] + 1) % 4, :]
                handplay = hands_np_play[i:i+1, (declarers[i] + 1) % 4, :]

                X_ftrs[i,:], B_ftrs[i,:] = binary.get_auction_binary_for_lead(sample_auction, handbidding, handplay, self.vuln, self.dealer, self.models)
        
        # We have a problem as we need to select different model based on the contract
        # if contract[1] == "N":
        #    lead_softmax = self.models.lead_nt_model.model(x_ftrs, b_ftrs)
        #else:
        #    lead_softmax = self.models.lead_suit_model.model(x_ftrs, b_ftrs)
                
        lead_softmax = self.models.lead_suit_model.pred_fun(X_ftrs, B_ftrs)
        
        lead_cards = np.argmax(lead_softmax, axis=1)
        
        X_sd = np.zeros((n_samples, 32 + 5 + 4*32))

        X_sd[s_all,32 + strains] = 1
        # lefty
        X_sd[:,(32 + 5 + 0*32):(32 + 5 + 1*32)] = hands_np_play[s_all, (declarers + 1) % 4]
        # dummy
        X_sd[:,(32 + 5 + 1*32):(32 + 5 + 2*32)] = hands_np_play[s_all, (declarers + 2) % 4]
        # righty
        X_sd[:,(32 + 5 + 2*32):(32 + 5 + 3*32)] = hands_np_play[s_all, (declarers + 3) % 4]
        # declarer
        X_sd[:,(32 + 5 + 3*32):] = hands_np_play[s_all, declarers]
        
        X_sd[s_all, lead_cards] = 1

        decl_tricks_softmax = self.models.sd_model.pred_fun(X_sd)
        return contracts, decl_tricks_softmax
    
    def expected_tricks_sd_no_lead(self, hands_np, auctions_np):
        n_samples = hands_np.shape[0]

        s_all = np.arange(n_samples, dtype=np.int32)
        declarers = np.zeros(n_samples, dtype=np.int32)
        strains = np.zeros(n_samples, dtype=np.int32)
      
        contracts = []

        # Apply the function to each 1D array along the third dimension
        hands_np_play = hands_np if self.models.n_cards_bidding == self.models.n_cards_play else np.apply_along_axis(self.transform_hand, 2, hands_np)                

        for i in range(n_samples):
            sample_auction = [bidding.ID2BID[bid_i] for bid_i in list(auctions_np[i, :]) if bid_i != 1]
            contract = bidding.get_contract(sample_auction)
            # All pass doesn't really fit, and is always 0 - we ignore it for now
            if contract is None:
                contracts.append("PASS")
                strains[i] = -1
                declarers[i] = -1
            else:
                contracts.append(contract)
                strains[i] = 'NSHDC'.index(contract[1])
                declarers[i] = 'NESW'.index(contract[-1])

        X_sd = np.zeros((n_samples, 5 + 4*32))

        X_sd[s_all,strains] = 1

        # Update the array to make the 0th element hotter only when strains is 0
        X_sd[np.arange(len(strains)), 0] = np.where(strains == 0, 10, X_sd[np.arange(len(strains)), 0])

        # lefty
        X_sd[:,(5 + 0*32):(5 + 1*32)] = hands_np_play[s_all, (declarers + 1) % 4]
        # dummy
        X_sd[:,(5 + 1*32):(5 + 2*32)] = hands_np_play[s_all, (declarers + 2) % 4]
        # righty
        X_sd[:,(5 + 2*32):(5 + 3*32)] = hands_np_play[s_all, (declarers + 3) % 4]
        # declarer
        X_sd[:,(5 + 3*32):] = hands_np_play[s_all, declarers]
        
        decl_tricks_softmax = self.models.sd_model_no_lead.pred_fun(X_sd)
        return contracts, decl_tricks_softmax

    def expected_tricks_dd(self, hands_np_as_pbn, auctions_np, bid = None):
        n_samples = auctions_np.shape[0]
        assert len(hands_np_as_pbn) == n_samples
        decl_tricks_softmax = np.zeros((n_samples, 14), dtype=np.int32)
        contracts = []
        t_start = time.time()
        
        sum = 0
        for i in range(n_samples):
            sample_auction = [bidding.ID2BID[bid_i] for bid_i in list(auctions_np[i, :]) if bid_i != 1]
            contract = bidding.get_contract(sample_auction)
            # All pass doesn't really fit, and is always 0 - we ignore it for now
            if contract is None:
                contracts.append("PASS")
                continue

            contracts.append(contract)
            strain = 'NSHDC'.index(contract[1])
            declarer = 'NESW'.index(contract[-1])
            leader = (declarer + 1) % 4

            leader = (leader + 4 - self.seat) % 4

            # Create PBN including pips
            hands_pbn = [hands_np_as_pbn[i]]

            # It will probably improve performance if all is calculated in one go
            dd_solved = self.ddsolver.solve(strain, leader, [], hands_pbn, 1)
            sum += 13 - dd_solved["max"][0]
            decl_tricks_softmax[i,13 - dd_solved["max"][0]] = 1

        if self.verbose:
            print(f'dds took: {(time.time() - t_start):0.4f}')
            print("Total tricks:", bid, sum)
        return contracts, decl_tricks_softmax
        

    def expected_score(self, turn_to_bid, contracts, decl_tricks_softmax):
        n_samples = len(contracts)
        scores_by_trick = np.zeros((n_samples, 14))
        for i, contract in enumerate(contracts):
            if contract.lower() != "pass":
                decl_i = 'NESW'.index(contract[-1])
                level = int(contract[0])
                if (turn_to_bid + decl_i) % 2 == 1:
                    tricks_needed = level + 6
                    # the other side is playing the contract
                    scores_by_trick[i] = -scoring.contract_scores_by_trick(contract, tuple(self.vuln))
                    # If going more than 3 down, we just score for three down
                    # This is just used when simulating their bidding
                    #if 'X' not in contract:
                    #    for j in range(0, tricks_needed - 2):
                    #        scores_by_trick[i,j] = scores_by_trick[i, tricks_needed - 2]
                else:
                    scores_by_trick[i] = scoring.contract_scores_by_trick(contract, tuple(self.vuln))
                    #if level >= 6:
                    #    for j in range(0, 2):  
                    #        scores_by_trick[i,j+12] -= 200
            else:
                scores_by_trick[i] = 0
        return np.sum(decl_tricks_softmax * scores_by_trick, axis=1)

    def expected_score_doubled(self, turn_to_bid, contracts, decl_tricks_softmax):
        n_samples = len(contracts)
        scores_by_trick = np.zeros((n_samples, 14))
        for i, contract in enumerate(contracts):
            if contract != "pass":
                #print("Contract",contract[:2]+"X"+contract[-1])
                scores_by_trick[i] = scoring.contract_scores_by_trick(contract[:2]+"X"+contract[-1], tuple(self.vuln))
                decl_i = 'NESW'.index(contract[-1])
                if (turn_to_bid + decl_i) % 2 == 1:
                    # the other side is playing the contract
                    scores_by_trick[i,:] *= -1
            else:
                scores_by_trick[i] = 0
        return np.sum(decl_tricks_softmax * scores_by_trick, axis=1)

class BotLead:

    def __init__(self, vuln, hand_str, models, sampler, seat, dealer, ddsolver, verbose):
        self.vuln = vuln
        self.hand_str = hand_str
        self.handbidding = binary.parse_hand_f(models.n_cards_bidding)(hand_str)
        self.handplay = binary.parse_hand_f(models.n_cards_play)(hand_str)
        self.hand52 = binary.parse_hand_f(52)(hand_str)
        self.models = models
        self.seat = seat
        self.dealer = dealer
        self.sampler = sampler
        self.verbose = verbose
        self.hash_integer  = calculate_seed(hand_str)         
        if self.verbose:
            print(f"Setting seed (Sampling bidding info) from {hand_str}: {self.hash_integer}")
        self.dd = ddsolver

    def get_random_generator(self):
        #print(f"{Fore.BLUE}Fetching random generator for lead {self.hash_integer}{Style.RESET_ALL}")
        return np.random.default_rng(self.hash_integer)

    def find_opening_lead(self, auction, aceking):
        # Validate input
        # We should check that auction match, that we are on lead
        t_start = time.time()
        lead_card_indexes, lead_softmax = self.get_opening_lead_candidates(auction)
        accepted_samples, sorted_bidding_score, tricks, p_hcp, p_shp, quality = self.simulate_outcomes_opening_lead(auction, lead_card_indexes, aceking)
        contract = bidding.get_contract(auction)
        scores_by_trick = scoring.contract_scores_by_trick(contract, tuple(self.vuln))

        candidate_cards = []
        expected_tricks_sd = None
        expected_tricks_dd = None
        expected_score_sd = None
        expected_score_dd = None
        expected_score_mp = None
        expected_score_imp = None

        # Consider adding reward to lead partner's suit
        # Consider penalty for leading from bad combionations (Singleton King, J9xx, etc)
        # https://kwbridge.com/leads.htm
        suit_adjust = [0,0,0,0]
        #print(self.seat, auction)
        partnersuit = bidding.get_partner_suit(self.seat, auction)
        if partnersuit != None and partnersuit < 4:
            suit_adjust[partnersuit] += self.models.reward_lead_partner_suit

        # penalty for leading trump with a singleton honor
        strain_i = bidding.get_strain_i(contract)
        if self.models.trump_lead_penalty:
            if strain_i > 0 and strain_i < 5:
                suit_adjust[strain_i - 1] -= self.models.trump_lead_penalty[0]
                trumps = self.hand_str.split('.')[strain_i - 1]
                if trumps == 'A' :   
                    suit_adjust[strain_i - 1] -= self.models.trump_lead_penalty[1]
                if trumps == 'K' :   
                    suit_adjust[strain_i - 1] -= self.models.trump_lead_penalty[2]
                if trumps == 'Q' :   
                    suit_adjust[strain_i - 1] -= self.models.trump_lead_penalty[3]

        level = int(contract[0])
        # Penalty for leading from Queen against slam
        # Double dummy is not good enough to handle this
        if level >= 6:
            suits = self.hand_str.split('.')
            for i, suit in enumerate(suits):
                if not suit:  # Skip empty suits
                    continue
        
                first_card = suit[0]  # Get the first card in the suit
        
                if first_card == 'Q':
                    if 'J' not in suit:
                        suit_adjust[i] -= 1
                if first_card == 'J':
                    if 'T' not in suit:
                        suit_adjust[i] -= 0.5
        # Adding reward for leading from a sequence


        if len(accepted_samples) > 0:
            if self.verbose:
                print("scores_by_trick", scores_by_trick)
            if self.models.use_real_imp_or_mp_opening_lead:
                dd_solved = {}
                for i, card_i in enumerate(lead_card_indexes):
                    dd_solved[card_i] = (13 - tricks[:,i,0]).astype(int).flatten().tolist()
                real_scores = calculate.calculate_score(dd_solved, 0, 0, scores_by_trick)
                if self.verbose:
                    print("Real scores")
                    print("\n".join(f"{Card.from_code(int(k), xcards=True)}: [{', '.join(f'{x:>5}' for x in v[:10])}..." for k, v in real_scores.items()))

                if self.models.matchpoint:
                    expected_score_mp_arr = calculate.calculate_mp_score(real_scores)
                else:
                    expected_score_imp_arr = calculate.calculate_imp_score(real_scores)

            for i, card_i in enumerate(lead_card_indexes):
                if self.models.use_real_imp_or_mp_opening_lead:
                    if self.models.matchpoint:
                        expected_score_mp = expected_score_mp_arr[card_i]
                    else:
                        expected_score_imp = round(expected_score_imp_arr[card_i],2)
                else:
                    assert(tricks[:,i,0].all() >= 0)
                    tricks_int = tricks[:,i,0].astype(int)
                    if self.models.double_dummy:
                        expected_tricks_dd=np.mean(tricks[:,i,0])
                        expected_score_dd = np.mean(scores_by_trick [tricks_int])
                    else:
                        expected_tricks_sd=np.mean(tricks[:,i,0])
                        expected_score_sd = np.mean(scores_by_trick [tricks_int])

                candidate_cards.append(CandidateCard(
                    card=Card.from_code(int(card_i), xcards=True),
                    insta_score=lead_softmax[0,card_i],
                    expected_tricks_sd=expected_tricks_sd,
                    expected_tricks_dd=expected_tricks_dd,
                    p_make_contract=np.mean(tricks[:,i,1]),
                    expected_score_sd = expected_score_sd if expected_score_sd is None else expected_score_sd + suit_adjust[int(card_i) // 8],
                    expected_score_dd = expected_score_dd if expected_score_dd is None else expected_score_dd + suit_adjust[int(card_i) // 8],
                    expected_score_mp = expected_score_mp if expected_score_mp is None else expected_score_mp + 10 * (suit_adjust[int(card_i) // 8]),
                    expected_score_imp = expected_score_imp if expected_score_imp is None else expected_score_imp + suit_adjust[int(card_i) // 8],
                    msg= f"suit adjust={suit_adjust[int(card_i) // 8]}" if suit_adjust[int(card_i) // 8] != 0 else ""
                ))

        else:
            # We do not have any samples, so we will just use the neural network        
            for i, card_i in enumerate(lead_card_indexes):
                candidate_cards.append(CandidateCard(
                    card=Card.from_code(int(card_i), xcards=True),
                    insta_score=lead_softmax[0,card_i],
                    expected_tricks_sd=expected_tricks_sd,
                    expected_tricks_dd=expected_tricks_dd,
                    p_make_contract=-1,
                    expected_score_sd = expected_score_sd,
                    expected_score_dd = expected_score_dd,
                    expected_score_mp = expected_score_mp,
                    expected_score_imp = expected_score_imp,
                    msg = ""
                ))

        candidate_cards = sorted(candidate_cards, key=lambda c: c.insta_score, reverse=True)

        # We will always take the card suggested by the neural network if it is above the threshold
        if candidate_cards[0].insta_score > self.models.lead_accept_nn:
            opening_lead = candidate_cards[0].card.code() 
            who = "NN - best"
        else:
            # If our sampling of the hands from the aution is bad.
            # We should probably try to find better samples, but for now, we just trust the neural network

            if (self.models.use_biddingquality_in_eval and  quality < 0):
                opening_lead = candidate_cards[0].card.code() 
                who = "NN - bad quality samples"
            else:
                # Now we will select the card to play
                # We have 3 factors, and they could all be right, so we remove most of the decimals
                # expected_tricks_sd is for declarer
                if self.models.use_real_imp_or_mp_opening_lead:
                    if self.models.matchpoint:
                        candidate_cards = sorted(candidate_cards, key=lambda c: (c.expected_score_mp, round(c.insta_score, 2)), reverse=True)
                        who = "Simulation (MP)"
                    else:
                        candidate_cards = sorted(candidate_cards, key=lambda c: (c.expected_score_imp, round(c.insta_score, 2)), reverse=True)
                        who = "Simulation (IMP)"
                else:
                    if self.models.double_dummy:                    
                        if self.models.matchpoint:
                            candidate_cards = sorted(candidate_cards, key=lambda c: (-round(c.expected_tricks_dd, 1), round(c.insta_score, 2)), reverse=True)
                            who = "Simulation (Tricks DD)"
                        else:
                            candidate_cards = sorted(candidate_cards, key=lambda c: (round(5*c.p_make_contract, 1), -round(c.expected_tricks_dd, 1), round(c.insta_score, 2)), reverse=True)
                            who = "Simulation (make/set DD)"
                    else:
                        if self.models.matchpoint:
                            candidate_cards = sorted(candidate_cards, key=lambda c: (-round(c.expected_tricks_sd, 1), round(c.insta_score, 2)), reverse=True)
                            who = "Simulation (Tricks SD)"
                        else:
                            candidate_cards = sorted(candidate_cards, key=lambda c: (round(5*c.p_make_contract, 1), -round(c.expected_tricks_sd, 1), round(c.insta_score, 2)), reverse=True)
                            who = "Simulation (make/set SD)"
                opening_lead = candidate_cards[0].card.code()

        if self.verbose:
            print(f"Samples quality: {quality:.3f}")
            for card in candidate_cards:
                print(card)
        if opening_lead % 8 > 5:
            contract = bidding.get_contract(auction)
            # Implement human carding here
            opening_lead52 = carding.select_right_card(self.hand52, opening_lead, self.get_random_generator(), contract, self.models, self.verbose)
        else:
            opening_lead52 = deck52.card32to52(opening_lead)

        samples1 = []
        samples2 = []
        samples = []
        if self.verbose:
            print(f"Accepted samples for opening lead: {accepted_samples.shape[0]}")
        if accepted_samples.shape[0] == 0:
            # Consider asking BBA for sample
            print("Accepted samples for opening lead: 0")
        for i in range(accepted_samples.shape[0]):
            # Extract scores for the current sample index i
            if self.models.use_real_imp_or_mp_opening_lead:
                k_values = {k: v[i] for k, v in real_scores.items()}

                # Check if all values in k_values are the same
                if len(set(k_values.values())) == 1:
                    k_str = ""  # All values are identical, so print an empty string
                    samples1.append('%s %s %s %s - %.5f | %s' % (
                        hand_to_str(self.handplay),
                        hand_to_str(accepted_samples[i,0,:], self.models.n_cards_bidding),
                        hand_to_str(accepted_samples[i,1,:], self.models.n_cards_bidding),
                        hand_to_str(accepted_samples[i,2,:], self.models.n_cards_bidding),
                        sorted_bidding_score[i],
                        k_str  # Include formatted key-value scores
                    ))
                else:
                    k_str = " ".join(f"{Card.from_code(int(k), xcards=True)}:{score}" for k, score in k_values.items())  # Format normally
                    samples2.append('%s %s %s %s - %.5f | %s' % (
                        hand_to_str(self.handplay),
                        hand_to_str(accepted_samples[i,0,:], self.models.n_cards_bidding),
                        hand_to_str(accepted_samples[i,1,:], self.models.n_cards_bidding),
                        hand_to_str(accepted_samples[i,2,:], self.models.n_cards_bidding),
                        sorted_bidding_score[i],
                        k_str  # Include formatted key-value scores
                    ))
                samples = (samples2 + samples1)[:self.models.sample_hands_for_review]
            else:
                for i in range(min(self.models.sample_hands_for_review, accepted_samples.shape[0])):
                    samples.append('%s %s %s %s %.5f' % (
                        hand_to_str(self.handplay),
                        hand_to_str(accepted_samples[i,0,:], self.models.n_cards_bidding),
                        hand_to_str(accepted_samples[i,1,:], self.models.n_cards_bidding),
                        hand_to_str(accepted_samples[i,2,:], self.models.n_cards_bidding),
                        sorted_bidding_score[i]
                    ))


        card_resp = CardResp(
            card=Card.from_code(opening_lead52),
            candidates=candidate_cards,
            samples=samples, 
            shape=p_shp,
            hcp=p_hcp,
            quality=quality,
            who=who,
            claim = -1
        )
        if self.verbose:
            print(' Opening lead found in {0:0.1f} seconds.'.format(time.time() - t_start))
            print(f"{Fore.LIGHTCYAN_EX}")
            pprint.pprint(card_resp.to_dict(), width=200)
            print(f"{Fore.RESET}")
        return card_resp

    def get_opening_lead_candidates(self, auction):
        x_ftrs, b_ftrs = binary.get_auction_binary_for_lead(auction, self.handbidding, self.handplay, self.vuln, self.dealer, self.models)
        contract = bidding.get_contract(auction)
        if contract[1] == "N":
            lead_softmax = self.models.lead_nt_model.pred_fun(x_ftrs, b_ftrs)
        else:
            lead_softmax = self.models.lead_suit_model.pred_fun(x_ftrs, b_ftrs)

        # We remove all cards suggested by NN not in hand, and rescale the softmax
        lead_softmax = follow_suit(lead_softmax, self.handplay, np.array([[0, 0, 0, 0]]))

        candidates = []
        # Make a copy of the lead_softmax array
        lead_softmax_copy = np.copy(lead_softmax)

        if self.verbose:
            print("Finding leads from neural network")
        while True:
            c = np.argmax(lead_softmax_copy[0])
            score = lead_softmax_copy[0][c]
            # Always take minimum the number from configuration
            if score < self.models.lead_threshold and len(candidates) >= self.models.min_opening_leads:
                break
            if self.verbose:
                print(f"{Card.from_code(int(c), xcards=True)} {score:.3f}")
            candidates.append(c)
            lead_softmax_copy[0][c] = 0

        return candidates, lead_softmax

    def simulate_outcomes_opening_lead(self, auction, lead_card_indexes, aceking):
        t_start = time.time()
        contract = bidding.get_contract(auction)

        decl_i = bidding.get_decl_i(contract)
        lead_index = (decl_i + 1) % 4
        
        # Reset randomizer
        self.rng = self.get_random_generator()

        accepted_samples, sorted_scores, p_hcp, p_shp, quality, samplings = self.sampler.generate_samples_iterative(auction, lead_index, self.sampler.sample_boards_for_auction_opening_lead, self.sampler.sample_hands_opening_lead, self.rng, self.hand_str, self.vuln, self.models, [], aceking)

        if self.verbose:
            print(f"Generated samples: {accepted_samples.shape[0]} in {samplings} samples. Quality {quality:.2f}")
            print(f'Now simulate on {min(accepted_samples.shape[0], self.sampler.sample_hands_opening_lead)} deals to find opening lead')
                

        # We have more samples than we want to calculate on
        # They are sorted according to the bidding trust, but above our threshold, so we pick based on scores
        if accepted_samples.shape[0] > self.sampler.sample_hands_opening_lead:
            # Normalize the scores to create a probability distribution
            scores = sorted_scores[:accepted_samples.shape[0]]
            probabilities = np.array(scores) / np.sum(scores)
            
            # Select indices based on the probability distribution
            selected_indices = self.get_random_generator().choice(
                accepted_samples.shape[0], 
                size=self.sampler.sample_hands_opening_lead, 
                replace=False, 
                p=probabilities
            )
            
            accepted_samples = accepted_samples[selected_indices, :, :]
            sorted_scores = sorted_scores[selected_indices]
    
        # We return tricks and the conversion to MP or IMP is done at a higher level
        if self.models.double_dummy:
            tricks = self.double_dummy_estimates(lead_card_indexes, contract, accepted_samples)
        else:
            tricks = self.single_dummy_estimates(lead_card_indexes, contract, accepted_samples)
        
        if self.verbose:
            print(f'simulate_outcomes_opening_lead took {(time.time() - t_start):0.4f}')

        return accepted_samples, sorted_scores, tricks, p_hcp, p_shp, quality

    def double_dummy_estimates(self, lead_card_indexes, contract, accepted_samples):
        #print("double_dummy_estimates",lead_card_indexes)
        n_accepted = accepted_samples.shape[0]
        tricks = np.zeros((n_accepted, len(lead_card_indexes), 2))
        strain_i = bidding.get_strain_i(contract)
        # When defending the target is another
        level = int(contract[0])
        tricks_needed = 13 - (level + 6) + 1

        t_start = time.time()
        for j, lead_card_i in enumerate(lead_card_indexes):
            # Subtract the opening lead from the hand
            lead_hand = self.hand52[0]
            # So now we need to figure out what the lead was if a pip
            if lead_card_i % 8 == 7:
                # it's a pip ~> choose a random one
                pips_mask = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1])
                lefty_led_pips = self.hand52.reshape((4, 13))[lead_card_i // 8] * pips_mask
                # Perhaps use human carding, but it is only for estimation
                opening_lead52 = int((lead_card_i // 8) * 13 + self.get_random_generator().choice(np.nonzero(lefty_led_pips)[0]))
            else:
                opening_lead52 = deck52.card32to52(lead_card_i)
            # Create PBN for hand
            hand_str = ""
            for k in (range(4)):
                for l in (range(13)):
                    if lead_hand[k*13 + l] == 1 and opening_lead52 != k*13 + l:
                        hand_str += 'AKQJT98765432'[l]
                if (k != 3): 
                    hand_str += '.'
            if self.verbose:
                print(f"Opening lead being examined: {Card.from_code(opening_lead52)} {n_accepted} samples. " , end="")
            t_start = time.time()
            hands_pbn = []
            for i in range(n_accepted):
                sample_pbn = 'N:' + hand_str + ' ' + ' '.join(deck52.handxxto52str(hand,self.models.n_cards_bidding) for hand in accepted_samples[i])
                hands_pbn.append(deck52.convert_cards(sample_pbn,opening_lead52, hand_str, self.get_random_generator(),self.models.n_cards_bidding))
                # lead is relative to the order in the PBN-file, so West is 0 here
            onlead = 0
                
            dd_solved = self.dd.solve(strain_i, onlead, [opening_lead52], hands_pbn, 1)

            for i in range(n_accepted):
                tricks[i, j, 0] = dd_solved["max"][i] 
                tricks[i, j, 1] = 1 if (13 - dd_solved["max"][i]) >= tricks_needed else 0

            if self.verbose:
                print(f'dds took: {(time.time() - t_start):0.4f}')
        return tricks

    def single_dummy_estimates(self, lead_card_indexes, contract, accepted_samples):
        t_start = time.time()
        n_accepted = accepted_samples.shape[0]
        X_sd = np.zeros((n_accepted, 32 + 5 + 4*32))

        strain_i = bidding.get_strain_i(contract)

        X_sd[:,32 + strain_i] = 1
        # lefty (That is us)
        X_sd[:,(32 + 5 + 0*32):(32 + 5 + 1*32)] = self.handbidding.reshape(32)
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

            tricks_softmax = self.models.sd_model.pred_fun(X_sd)
            # Get the indices of the top three probabilities
            probabilities = tricks_softmax.flatten()
            top_indices = np.argsort(tricks_softmax.flatten())[-3:]

            # Get the top three probabilities
            top_probs = probabilities[top_indices]

            # Normalize the top probabilities to sum up to 1.0 (or 100%)
            normalized_probs = top_probs / np.sum(top_probs)

            # Reconstruct the array with updated probabilities
            result_array = np.zeros_like(probabilities)
            result_array[top_indices] = normalized_probs

            # Reshape the result to match the original shape
            result_array = result_array.reshape((1, -1))

            tricks[:, j, 0:1] = expected_tricks_sd(result_array)
            tricks[:, j, 1:2] = p_defeat_contract(contract, result_array)
        if self.verbose:
            print(f'single dummy estimates took {(time.time() - t_start):0.4f}')
        return tricks

class CardPlayer:

    def __init__(self, models, player_i, hand_str, public_hand_str, contract, is_decl_vuln, sampler, pimc = None, ddsolver = None, verbose = False):
        self.models = models
        self.player_models = models.player_models
        self.strain_i = bidding.get_strain_i(contract)
        # Select playing models based on suit or NT
        if self.strain_i == 0:
            self.playermodel = models.player_models[player_i]
        else:
            self.playermodel = models.player_models[player_i + 4]
        self.player_i = player_i
        self.hand_str = hand_str
        self.public_hand_str = public_hand_str
        self.hand32 = binary.parse_hand_f(32)(hand_str).reshape(32)
        self.hand52 = binary.parse_hand_f(52)(hand_str).reshape(52)
        self.public52 = binary.parse_hand_f(52)(public_hand_str).reshape(52)
        self.dummy = self.public52.copy()
        self.contract = contract
        self.is_decl_vuln = is_decl_vuln
        self.n_tricks_taken = 0
        self.missing_cards = (13 -  np.array(binary.get_shape_array(self.hand52)) -  np.array(binary.get_shape_array(self.public52))).astype(int)
        self.missing_cards_initial = self.missing_cards
        self.verbose = verbose
        self.level = int(contract[0])
        self.init_x_play(binary.parse_hand_f(32)(public_hand_str), self.level, self.strain_i)
        self.dds = ddsolver
        self.sampler = sampler
        self.claimer = Claimer(self.verbose, self.dds)

        # If we don't get a hand, the class is just used for recording
        if hand_str != "...":
            self.sample_hands_for_review = models.sample_hands_for_review
            self.bid_accept_play_threshold = sampler.bid_accept_play_threshold
            self.score_by_tricks_taken = [scoring.score(self.contract, self.is_decl_vuln, n_tricks) for n_tricks in range(14)]
            if (player_i == 1):
                self.hash_integer  = calculate_seed(public_hand_str)         
                if self.verbose:
                    print(f"Setting seed {player_i} (Sampling bidding info) from {public_hand_str}: {self.hash_integer}")
            else:
                self.hash_integer  = calculate_seed(hand_str)         
                if self.verbose:
                    print(f"Setting seed {player_i} (Sampling bidding info) from {hand_str}: {self.hash_integer}")
            self.pimc = pimc
            # False until it kicks in
            self.pimc_declaring = False
            self.pimc_defending = False
        else: 
            self.pimc = None
    
    def get_random_generator(self):
        #print(f"{Fore.BLUE}Fetching random generator for player {self.hash_integer}{Style.RESET_ALL}")
        return np.random.default_rng(self.hash_integer)
    
    def init_x_play(self, public_hand, level, strain_i):
        self.x_play = np.zeros((1, 13, 298), dtype=np.int8)
        binary.BinaryInput(self.x_play[:,0,:]).set_player_hand(self.hand32)
        binary.BinaryInput(self.x_play[:,0,:]).set_public_hand(public_hand)
        self.x_play[:,0,292] = level
        self.x_play[:,0,293+strain_i] = 1

    def set_real_card_played(self, card52, played_by, openinglead=False):
        # Dummy has no PIMC
        if self.pimc and self.player_i != 1:
            self.pimc.set_card_played(card52, played_by, openinglead)
        # We do not count our own cards and not dummys cards
        if played_by == 1:
            return
        if self.player_i == played_by:
            return
        # Dummys is not counting declares cards
        if self.player_i == 1 and played_by == 3:
            return
        self.missing_cards[(card52 // 13)] -= 1

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

    def find_and_update_constraints(self, players_states, quality, player_i):
        # Based on player states we should be able to find min max for suits and hcps, and add that before calling PIMC
        #print("Updating constraints", player_i)
        idx1 = 2 if player_i == 0 else 0
        idx2 = 3 if player_i % 2 == 0 else 2
        h1 = []
        h3 = []
        s1 = []
        s3 = []
        for i in range(players_states[0].shape[0]):
            h1.append(binary.get_hcp(hand = np.array(players_states[idx1][i, 0, :32].astype(int)).reshape(1,32)))
            s1.append(binary.get_shape(hand = np.array(players_states[idx1][i, 0, :32].astype(int)).reshape(1,32))[0])
            h3.append(binary.get_hcp(hand = np.array(players_states[idx2][i, 0, :32].astype(int)).reshape(1,32)))
            s3.append(binary.get_shape(hand = np.array(players_states[idx2][i, 0, :32].astype(int)).reshape(1,32))[0])
            #print(hand_to_str(players_states[idx2][i,0,:32].astype(int)))
            #print(binary.get_shape(hand = np.array(players_states[idx2][i, 0, :32].astype(int)).reshape(1,32))[0])
        min_h1 = int(min(h1))
        max_h1 = int(max(h1))
        min_h3 = int(min(h3))
        max_h3 = int(max(h3))
        if self.verbose:
            print("HCP constraints:",min_h1, max_h1, min_h3, max_h3, quality)
        self.pimc.set_hcp_constraints(min_h1, max_h1, min_h3, max_h3, quality)
        min_values1 = [min(col) for col in zip(*s1)]
        max_values1 = [max(col) for col in zip(*s1)]
        min_values3 = [min(col) for col in zip(*s3)]
        max_values3 = [max(col) for col in zip(*s3)]
        if self.verbose:
            print("Shape constraints:",min_values1, max_values1, min_values3, max_values3)

        self.pimc.set_shape_constraints(min_values1, max_values1, min_values3, max_values3, quality)

    def check_pimc_constraints(self, trick_i, players_states, quality):
        # If we are declarer and PIMC enabled - use PIMC
        self.pimc_declaring = self.models.pimc_use_declaring and trick_i >= (self.models.pimc_start_trick_declarer - 1) and trick_i < (self.models.pimc_stop_trick_declarer)
        self.pimc_defending = self.models.pimc_use_defending and trick_i >= (self.models.pimc_start_trick_defender - 1) and trick_i < (self.models.pimc_stop_trick_defender)
        if not self.pimc_defending and not self.pimc_declaring:
            return
        if self.models.pimc_constraints:
            if self.models.pimc_constraints_each_trick:
                self.find_and_update_constraints(players_states, quality,self.player_i)
            else:
                if self.pimc.constraints_updated:
                    return
                if self.pimc_declaring and (self.player_i == 1 or self.player_i == 3):
                    if self.verbose:
                        print("Declaring", self.pimc_declaring, self.player_i, trick_i)
                    self.find_and_update_constraints(players_states, quality,self.player_i)
                if self.pimc_defending and (self.player_i == 0 or self.player_i == 2):
                    if self.verbose:
                        print("Defending", self.pimc_defending, self.player_i, trick_i)
                    self.find_and_update_constraints(players_states, quality,self.player_i)

    def merge_candidate_cards(self, pimc_resp, dd_resp, engine, weight, quality):
        merged_cards = {}

        if quality < self.models.pimc_bidding_quality:
            weight = 0.5

        for card52, (e_tricks, e_score, e_make, msg) in dd_resp.items():
            pimc_e_tricks, pimc_e_score, pimc_e_make, pimc_msg = pimc_resp[card52]
            new_e_tricks = round((pimc_e_tricks * weight + e_tricks * (1-weight)),2) if pimc_e_tricks is not None and e_tricks is not None else None
            new_e_score = round((pimc_e_score * weight + e_score * (1-weight)),2) if pimc_e_score is not None and e_score is not None else None
            new_e_make = round((pimc_e_make * weight + e_make * (1-weight)),2) if pimc_e_make is not None and e_make is not None else None
            new_msg = msg +"|" if msg else "" + engine + f" {weight*100:.0f}%|" + (pimc_msg or '') 
            new_msg += f"|{pimc_e_tricks:.2f} {pimc_e_score:.2f} {pimc_e_make:.2f}"
            new_msg += f"|BEN DD {(1-weight)*100:.0f}%|" 
            new_msg += f"{e_tricks:.2f} {e_score:.2f} {e_make:.2f}"
            merged_cards[card52] = (new_e_tricks, new_e_score, new_e_make, new_msg)

        return merged_cards
    
    def play_card(self, trick_i, leader_i, current_trick52, tricks52, players_states, worlds, bidding_scores, quality, probability_of_occurence, shown_out_suits, play_status, lead_scores, play_scores, logical_play_scores, discard_scores):
        t_start = time.time()
        samples = []

        for i in range(min(self.sample_hands_for_review, players_states[0].shape[0])):
            samples.append('%s %s %s %s - %.5f %.5f %.5f %.5f %.5f %.5f ' % (
                hand_to_str(players_states[0][i,0,:32].astype(int)),
                hand_to_str(players_states[1][i,0,:32].astype(int)),
                hand_to_str(players_states[2][i,0,:32].astype(int)),
                hand_to_str(players_states[3][i,0,:32].astype(int)),
                bidding_scores[i],
                probability_of_occurence[i],
                lead_scores[i],
                play_scores[i],
                logical_play_scores[i],
                discard_scores[i]
            ))
        
        if quality < 0.1 and self.verbose:
            print("Bad Samples:")
            print(samples)
        card_resp_alphamju = []
        # AlphaMju is not good at discarding yet
        if self.models.alphamju_declaring and (self.player_i == 1 or self.player_i == 3) and trick_i > 1 and play_status != "Discard":
            suit = "NSHDC"[self.strain_i]
            # if defending the target is another
            level = int(self.contract[0])
            if self.player_i % 2 == 1:
                tricks_needed = level + 6 - self.n_tricks_taken
            else:
                tricks_needed = 13 - (level + 6) - self.n_tricks_taken + 1

            tricks_left = 13 - trick_i

            print(f"Tricks needed: {tricks_needed}, Tricks left: {tricks_left}, trick_i: {trick_i}, play_status: {play_status}, leader_i: {leader_i}, current_trick52: {current_trick52}")

            # Initialize a flag to track if we adjusted tricks_needed up or down
            adjusted_up = False
            adjusted_down = False
            previous_card_resp_alphamju = None  # Store previous result of alphamju when all_100 was True

            # Start by calling alphamju inside the loop to execute at least once
            while True:
                # Call alphamju with the current value of tricks_needed
                card_resp_alphamju = alphamju(min(tricks_needed, tricks_left), suit, self.player_i, current_trick52, worlds, self.get_random_generator())

                if card_resp_alphamju != []:
                    candidate_cards = []

                # Check if all candidates have percent 100
                    all_100 = all(percent == 100 for card, percent in card_resp_alphamju)
                    # Check if all candidates have percent 0
                    all_0 = all(percent == 0 for card, percent in card_resp_alphamju)

                    # If all percent values are 100 and tricks_needed < tricks_left, increment tricks_needed
                    if all_100 and tricks_needed < tricks_left and not adjusted_down:
                        previous_card_resp_alphamju = card_resp_alphamju  # Save the current result
                        tricks_needed += 1
                        adjusted_up = True  # Mark that we've adjusted up
                        adjusted_down = False  # Reset the down adjustment flag
                        continue  # Continue the loop to try again with the updated tricks_needed

                    # If all percent values are 0 and tricks_needed > 1, decrement tricks_needed
                    if all_0 and tricks_needed > 1:
                        if adjusted_up:
                            # If we encounter all_0 after adjusting up, revert to the previous result from all_100
                            if previous_card_resp_alphamju is not None:
                                card_resp_alphamju = previous_card_resp_alphamju  # Use the previous result
                        else:
                            tricks_needed -= 1
                            adjusted_down = True  # Mark that we've adjusted down
                            continue  # Continue the loop to try again with the updated tricks_needed
      

                    # Collect candidate cards
                    for card, percent in card_resp_alphamju:
                        if percent > 0:
                            candidate = CandidateCard(Card.from_symbol(card), percent, -1, -1, -1, -1, -1, -1, -1, -1)
                            candidate_cards.append(candidate)

                    if candidate_cards != []:
                        # Sort by insta_score (descending), and in case of tie, by index (ascending)
                        candidate_cards = sorted(candidate_cards, key=lambda x: (x.insta_score, -candidate_cards.index(x)), reverse=True)

                        card_resp = CardResp(
                            card=candidate_cards[0].card,
                            candidates=candidate_cards,
                            samples=samples,
                            shape=-1,
                            hcp=-1, 
                            quality=quality,
                            who='𝛼𝞵', 
                            claim=-1
                        )
                        return card_resp

                    # If card_resp_alphamju is empty, break out of the loop (or handle as needed)
                    break

                # If card_resp_alphamju is empty, break out of the loop (or handle as needed)
                break


            
        # When play_status is discard, it might be a good idea to use PIMC even if it is not enabled
        if play_status == "discard" and not self.models.pimc_use_discard:
            dd_resp_cards, claim_cards = self.get_cards_dd_evaluation(trick_i, leader_i, tricks52, current_trick52, players_states, probability_of_occurence)
            self.update_with_alphamju(card_resp_alphamju, merged_card_resp)
            card_resp = self.pick_card_after_dd_eval(trick_i, leader_i, current_trick52, tricks52, players_states, dd_resp_cards, bidding_scores, quality, samples, play_status, self.missing_cards, claim_cards, shown_out_suits)
        else:                    
            if self.pimc_declaring and (self.player_i == 1 or self.player_i == 3):
                pimc_resp_cards = self.pimc.nextplay(self.player_i, shown_out_suits, self.missing_cards)
                if self.verbose:
                    print("PIMC result:")
                    print("\n".join(f"{Card.from_code(k)}: {v}" for k, v in pimc_resp_cards.items()))
                assert pimc_resp_cards is not None, "PIMC result is None"
                if self.models.pimc_ben_dd_declaring:
                    #print(pimc_resp_cards)
                    dd_resp_cards, claim_cards = self.get_cards_dd_evaluation(trick_i, leader_i, tricks52, current_trick52, players_states, probability_of_occurence)
                    #print(dd_resp_cards)
                    merged_card_resp = self.merge_candidate_cards(pimc_resp_cards, dd_resp_cards, "PIMC", self.models.pimc_ben_dd_declaring_weight, quality)
                else:
                    merged_card_resp = pimc_resp_cards
                self.update_with_alphamju(card_resp_alphamju, merged_card_resp)
                card_resp = self.pick_card_after_pimc_eval(trick_i, leader_i, current_trick52, tricks52, players_states, merged_card_resp, bidding_scores, quality, samples, play_status, self.missing_cards, claim_cards, shown_out_suits)            
            else:
                if self.pimc_defending and (self.player_i == 0 or self.player_i == 2):
                    pimc_resp_cards = self.pimc.nextplay(self.player_i, shown_out_suits, self.missing_cards)
                    if self.verbose:
                        print("PIMCDef result:")
                        print("\n".join(f"{Card.from_code(k)}: {v}" for k, v in pimc_resp_cards.items()))

                    assert pimc_resp_cards is not None, "PIMCDef result is None"
                    if self.models.pimc_ben_dd_defending:
                        #print(pimc_resp_cards)
                        dd_resp_cards, claim_cards = self.get_cards_dd_evaluation(trick_i, leader_i, tricks52, current_trick52, players_states, probability_of_occurence)
                        #print(dd_resp_cards)
                        merged_card_resp = self.merge_candidate_cards(pimc_resp_cards, dd_resp_cards, "PIMCDef", self.models.pimc_ben_dd_defending_weight, quality)
                    else:
                        merged_card_resp = pimc_resp_cards
                    self.update_with_alphamju(card_resp_alphamju, merged_card_resp)
                    card_resp = self.pick_card_after_pimc_eval(trick_i, leader_i, current_trick52, tricks52, players_states, merged_card_resp, bidding_scores, quality, samples, play_status, self.missing_cards, claim_cards, shown_out_suits)            
                    
                else:
                    dd_resp_cards, claim_cards = self.get_cards_dd_evaluation(trick_i, leader_i, tricks52, current_trick52, players_states, probability_of_occurence)
                    self.update_with_alphamju(card_resp_alphamju, dd_resp_cards)
                    card_resp = self.pick_card_after_dd_eval(trick_i, leader_i, current_trick52, tricks52, players_states, dd_resp_cards, bidding_scores, quality, samples, play_status, self.missing_cards, claim_cards, shown_out_suits)

        if self.verbose:
            print(f'Play card response time: {time.time() - t_start:0.4f}')
        return card_resp

    def update_with_alphamju(self, card_resp_alphamju, merged_card_resp):
        for card, percent in card_resp_alphamju:
            card52 = Card.from_symbol(card).code()
            if card52 in merged_card_resp:
                values = merged_card_resp[card52]
                updated_msg = values[-1] + f"|𝛼𝞵 {percent}%"
                merged_card_resp[card52] = (*values[:-1], updated_msg)

    def get_cards_dd_evaluation(self, trick_i, leader_i, tricks52, current_trick52, players_states, probabilities_list):
        
        n_samples = players_states[0].shape[0]
        assert n_samples > 0, "No samples for DDSolver"

        cards_played = list([card for trick in tricks52 for card in trick])
        # All previously played pips are also unavailable, so we use the original dummy and not what we can see
        # unavailable_cards = set(list(np.nonzero(self.hand52)[0]) + list(np.nonzero(self.dummy)[0]) + current_trick52)
        unavailable_cards = set(list(np.nonzero(self.hand52)[0]) + list(np.nonzero(self.public52)[0]) + current_trick52 + cards_played)

        pips = [
            [c for c in range(7, 13) if i*13+c not in unavailable_cards] for i in range(4)
        ]

        symbols = 'AKQJT98765432'

        current_trick_players = [(leader_i + i) % 4 for i in range(len(current_trick52))]
        rng = self.get_random_generator()
        hands_pbn = []
        for i in range(n_samples):
            hands = [None, None, None, None]
            for j in range(4):
                rng.shuffle(pips[j])
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
            hands_pbn.append('N:' + ' '.join(hands))


        if self.verbose:
            print('10 first samples:')
            print('\n'.join(hands_pbn[:10]))
        
        t_start = time.time()
        if self.verbose:
            print("Samples:", n_samples, " Solving:",len(hands_pbn))
        
        dd_solved = self.dds.solve(self.strain_i, leader_i, current_trick52, hands_pbn, 3)
        
        # if defending the target is another
        level = int(self.contract[0])
        if self.player_i % 2 == 1:
            tricks_needed = level + 6 - self.n_tricks_taken
        else:
            tricks_needed = 13 - (level + 6) - self.n_tricks_taken + 1

        if self.verbose:
            print("Calculating tricks. Using probability {}".format(self.models.use_probability))
        if self.models.use_probability:
            card_tricks = self.dds.expected_tricks_dds_probability(dd_solved, probabilities_list)
        else:
            card_tricks = self.dds.expected_tricks_dds(dd_solved)
        #print(card_tricks)
        making = self.dds.p_made_target(tricks_needed)(dd_solved)

        if self.models.use_real_imp_or_mp:
            if self.verbose:
                print(f"Probabilities: [{', '.join(f'{x:>6.2f}' for x in probabilities_list[:10])}...]")
                self.dds.print_dd_results(dd_solved)

            # print("Calculated scores")
            real_scores = calculate.calculate_score(dd_solved, self.n_tricks_taken, self.player_i, self.score_by_tricks_taken)
            if self.verbose:
                print("Real scores")
                print("\n".join(f"{Card.from_code(int(k))}: [{', '.join(f'{x:>5}' for x in v[:10])}..." for k, v in real_scores.items()))
            if self.models.use_probability:
                if self.models.matchpoint:
                    card_ev = calculate.calculate_mp_score_probability(dd_solved,probabilities_list)
                else:
                    card_ev = calculate.calculate_imp_score_probability(real_scores,probabilities_list)
            else:
                if self.models.matchpoint:
                    card_ev = calculate.calculate_mp_score(real_scores)
                else:
                    card_ev = calculate.calculate_imp_score(real_scores)

        else:
            if self.models.use_probability:
                if self.models.matchpoint:
                    card_ev = calculate.get_card_ev_mp_probability(dd_solved, probabilities_list)
                else:
                    card_ev = calculate.get_card_ev_probability(dd_solved, probabilities_list, self.n_tricks_taken, self.player_i, self.score_by_tricks_taken)
            else:
                if self.models.matchpoint:
                    card_ev = calculate.get_card_ev_mp(dd_solved, self.n_tricks_taken, self.player_i)
                else:
                    card_ev = calculate.get_card_ev(dd_solved, self.n_tricks_taken, self.player_i, self.score_by_tricks_taken)

        card_result = {}
        claim_cards = []
        for key in dd_solved.keys():
            card_result[key] = (card_tricks[key], card_ev[key], making[key], "")
            if card_tricks[key] == 13 - trick_i:
                claim_cards.append(key)
            if self.verbose:
                print(f'{deck52.decode_card(key)} {card_tricks[key]:0.3f} {card_ev[key]:5.2f} {making[key]:0.2f}')

        if self.verbose:
            print(f'dds took: {(time.time() - t_start):0.4f}')

        return card_result, claim_cards
    
    
    def next_card_softmax(self, trick_i):
        if self.verbose:
            print(f"next_card_softmax. Model: {self.playermodel.name} trick: {trick_i+1}")

        cards_softmax = self.playermodel.next_cards_softmax(self.x_play[:,:(trick_i + 1),:])
        assert cards_softmax.shape == (1, 32), f"Expected shape (1, 32), but got shape {cards_softmax.shape}"

        x = follow_suit(
            cards_softmax,
            binary.BinaryInput(self.x_play[:,trick_i,:]).get_player_hand(),
            binary.BinaryInput(self.x_play[:,trick_i,:]).get_this_trick_lead_suit(),
        )
        return x.reshape(-1)
    
    def calculate_sure_tricks(self, our_hands, missing_cards):
        def suit_indices(suit_index):
            """Returns the start and end indices for a suit."""
            return suit_index * 13, (suit_index + 1) * 13

        def extract_cards(hand, suit_index):
            """Extracts cards for a suit from a hand."""
            start, end = suit_indices(suit_index)
            return [12 - i for i, value in enumerate(hand[start:end]) if value == 1]

        def sure_tricks_in_suit(our_hand1, our_hand2, opponents_hand):
            """Calculates the sure tricks in a single suit."""
            max_tricks_possible = max(len(our_hand1), len(our_hand2))
            our_cards = sorted(our_hand1 + our_hand2, reverse=True)  # Sorted in descending order
            # If we can take all the tricks in ths suit we will not adjust
            if our_cards == [] or opponents_hand == []:
                return 0
            opponents_cards = sorted(opponents_hand, reverse=True)

            sure_tricks = 0
            opponent_highest = opponents_cards.pop(0)
            # Perhaps we should count possible tricks
            for our_card in our_cards:
                
                # Card 
                if opponent_highest > our_card:
                    break
                else:
                    sure_tricks += 1
                    if sure_tricks >= max_tricks_possible:
                        break

            # We do not want to create tricks for the opponents
            if len(opponents_hand)/2 > max_tricks_possible:
                return (max_tricks_possible - sure_tricks - (len(opponents_hand)/2)) / 100

            return (max_tricks_possible - sure_tricks) / 100

        results = {}

        for suit_index in range(4):
            # Extract cards for the current suit
            our_hand1 = extract_cards(our_hands[0], suit_index)
            our_hand2 = extract_cards(our_hands[1], suit_index)
            opponents_hand = extract_cards(missing_cards, suit_index)

            # Calculate score tricks for the current suit
            tricks = sure_tricks_in_suit(our_hand1, our_hand2, opponents_hand)
            results[suit_index] = tricks

        return results

    def calculate_suit_adjust_for_nt(self, leader_i, play_status, strain_i, trump_adjust, tricks52):
        # Only for dummy and declarer snd NT
        result = [0,0,0,0]
        if self.strain_i != 0:
            result[strain_i-1] = trump_adjust
            return result
        if leader_i % 2 != 1:
            return result
        # Only adjustment if we are on lead
        if play_status != "Lead": 
            return result
        s = []
        if self.models.use_suit_adjust:
            played_cards = np.zeros(52)
            remaining_cards = np.ones(52)
            for i in range(len(tricks52)):
                for J in range(4):
                    played_cards[tricks52[i][J]] = 1
            remaining_cards -= played_cards
            remaining_cards -= self.hand52
            remaining_cards -= self.public52
            result = self.calculate_sure_tricks([self.hand52, self.public52], remaining_cards)

        if self.verbose:
            print("Suit adjust", result)
        return result
        
    # Trying to help BEN to know, when to draw trump, and when not to
    def calculate_trump_adjust(self, play_status):
        trump_adjust = 0
        # Only in suit contract and if we are on lead and we are declaring
        if self.strain_i != 0 and play_status == "Lead" and (self.player_i == 1 or self.player_i == 3):
            # Any outstanding trump?
            if self.models.draw_trump_reward > 0 and self.missing_cards[self.strain_i-1] > 0:
                trump_adjust = self.models.draw_trump_reward
            # Just to be sure we won't show opps that they have no trump
            if self.models.draw_trump_penalty > 0 and self.missing_cards[self.strain_i-1] == 0:
                trump_adjust = -self.models.draw_trump_penalty
            if self.verbose:
                print("Trump adjust", trump_adjust)
            if self.models.use_real_imp_or_mp:
                if self.models.matchpoint:
                    trump_adjust = trump_adjust * 2
                else:
                    if (trump_adjust > 0):
                        trump_adjust = trump_adjust * 2
                    else:
                        trump_adjust = trump_adjust / 2
        return trump_adjust

    def pick_card_after_pimc_eval(self, trick_i, leader_i, current_trick, tricks52,  players_states, card_dd, bidding_scores, quality, samples, play_status, missing_cards, claim_cards, shown_out_suits):
        t_start = time.time()
        if claim_cards is not None and len(claim_cards) > 0:
            # DD we could claim, so let us check if one card is better
            bad_play = self.claimer.claimcheck(
                strain_i=self.strain_i,
                player_i=self.player_i,
                hands52=[self.hand52, self.public52],
                tricks52=tricks52,
                claim_cards=claim_cards,
                shown_out_suits=shown_out_suits,
                missing_cards=missing_cards,
                current_trick=current_trick,
                n_samples=50
            )
            claim_cards = [card for card in claim_cards if card not in bad_play]
        else:
            bad_play = []
        if self.verbose:
            print(f"Claim cards after check: {claim_cards}, Bad claim cards {bad_play}")

        card_scores = self.next_card_softmax(trick_i)
        if self.verbose:
            print(f'Next card response time: {time.time() - t_start:0.4f}')

        # Create a lookup dictionary to find the scores
        card_nn = {c: round(s, 3) for c, s in zip(np.arange(self.models.n_cards_play), card_scores)}
        
        trump_adjust = self.calculate_trump_adjust(play_status)
    
        suit_adjust = self.calculate_suit_adjust_for_nt(leader_i, play_status, self.strain_i, trump_adjust, tricks52)

        # Problem with playing the J from Jxxx as it might be a trick.  Seems to be in second hand as defender when card is played from dummu
        #if self.player_i == 2:
        #    if len(current_trick) == 1:
        #        print("Second hand")

        if self.verbose:
            print(f'Suit adjust: {suit_adjust}, Trump adjust: {trump_adjust}, play_status: {play_status}')
        candidate_cards = []
        
        current_suit = 0
        current_insta_score = 0
        for card52, (e_tricks, e_score, e_make, msg) in card_dd.items():
            adjust_card = suit_adjust[card52 // 13]
            card32 = deck52.card52to32(card52)
            insta_score = self.get_nn_score(card32, card52, card_nn, play_status, tricks52)
            if len(claim_cards) == 0:
                # Ignore cards not suggested by the NN
                if insta_score < self.models.pimc_trust_NN:
                    continue
                if insta_score > self.models.play_reward_threshold_NN and self.models.play_reward_threshold_NN > 0:
                    adjust_card += 0.1           
            else:
                # If we can take rest we don't adjust, then NN will decide if equal
                # Another option could be to resample the hands without restrictions
                if e_tricks >= 13 - trick_i:
                    # Calculate valid claim cards
                    # if card32 // 8 != self.strain_i - 1:
                    adjust_card = 0
                if card52 in bad_play:
                    adjust_card += -0.2            

            card = self.create_card(suit_adjust, card52, e_tricks, e_score, e_make, msg, adjust_card, insta_score)

            # For now we want lowest card first - in deck it is from A->2 so highest value is lowest card
            if (card52 > current_suit) and (insta_score == current_insta_score) and (card52 // 13 == current_suit // 13):
                candidate_cards.insert(0, card)
            else:
                candidate_cards.append(card)
            current_suit = card52
            current_insta_score = insta_score


        if self.models.use_real_imp_or_mp:
            if self.models.matchpoint:
                candidate_cards = sorted(enumerate(candidate_cards), key=lambda x: (x[1].expected_score_mp, x[1].expected_tricks_dd, x[1].insta_score, -x[0]), reverse=True)
            else:
                candidate_cards = sorted(enumerate(candidate_cards), key=lambda x: (x[1].expected_score_imp, x[1].expected_tricks_dd, x[1].insta_score, -x[0]), reverse=True)
        else:            
            if self.models.matchpoint:
                candidate_cards = sorted(enumerate(candidate_cards), key=lambda x: (x[1].expected_tricks_dd, x[1].insta_score, -x[0]), reverse=True)
            else:
                candidate_cards = sorted(enumerate(candidate_cards), key=lambda x: (round(5*x[1].p_make_contract, 1), x[1].expected_score_dd, x[1].insta_score, -x[0]), reverse=True)

        candidate_cards = [card for _, card in candidate_cards]

        if self.verbose:
            for i in range(len(candidate_cards)):
                print(candidate_cards[i].card, f"{candidate_cards[i].insta_score:.3f}", candidate_cards[i].expected_tricks_dd, round(5 * candidate_cards[i].p_make_contract, 1), int(candidate_cards[i].expected_tricks_dd * 10) / 10)

        if self.models.matchpoint:
            if self.models.pimc_ben_dd_declaring or self.models.pimc_ben_dd_defending:
                who = "PIMC-BEN-MP" 
            else:
                who = "PIMC-MP" 
        else:
            if self.models.pimc_ben_dd_declaring or self.models.pimc_ben_dd_defending:
                who = "PIMC-BEN-IMP" 
            else:
                who = "BEN-IMP" 
            
        right_card, who = carding.select_right_card_for_play(candidate_cards, self.get_random_generator(), self.contract, self.models, self.hand_str, self.public_hand_str, self.player_i, tricks52, current_trick, missing_cards, play_status, who, claim_cards,self.verbose)
        best_card_resp = CardResp(
            card=right_card,
            candidates=candidate_cards,
            samples=samples,
            shape=-1,
            hcp=-1, 
            quality=quality,
            who = who, 
            claim = -1 if not claim_cards else 13 - trick_i
        )
        return best_card_resp

    def get_nn_score(self, card32, card52, card_nn, play_status, tricks52):

        if play_status == "Lead":
            if len(tricks52) > 8:
                #print(card52)
                higher_cards = card52 % 13
                for trick in tricks52:
                    for card in trick:
                        if card // 13 == card52 // 13:
                            if card % 13 < card52 % 13:
                                higher_cards -=1
                # When playing the last 5 tricks we add priority to winners, and do not trust the neural network
                if higher_cards == 0:
                    return max(card_nn.get(card32, 0), 0.5)

        return card_nn.get(card32, 0)

    def pick_card_after_dd_eval(self, trick_i, leader_i, current_trick, tricks52, players_states, card_dd, bidding_scores, quality, samples, play_status, missing_cards, claim_cards, shown_out_suits):
        t_start = time.time()
        if claim_cards is not None and len(claim_cards) > 0:
            # DD we could claim, so let us check if one card is better
            bad_play = self.claimer.claimcheck(
                strain_i=self.strain_i,
                player_i=self.player_i,
                hands52=[self.hand52, self.public52],
                tricks52=tricks52,
                claim_cards=claim_cards,
                shown_out_suits=shown_out_suits,
                missing_cards=missing_cards,
                current_trick=current_trick,
                n_samples=50
            )
            claim_cards = [card for card in claim_cards if card not in bad_play]
        else:
            bad_play = []

        if self.verbose:
            print(f"Claim cards after check: {claim_cards}, Bad claim cards {bad_play}")

        card_scores = self.next_card_softmax(trick_i)
        if self.verbose:
            print(f'Next card response time: {time.time() - t_start:0.4f}')

        # Create a lookup dictionary to find the scores
        card_nn = {c: round(s, 3) for c, s in zip(np.arange(self.models.n_cards_play), card_scores)}

        trump_adjust = self.calculate_trump_adjust(play_status)

        suit_adjust = self.calculate_suit_adjust_for_nt(leader_i, play_status, self.strain_i, trump_adjust, tricks52)

        # Problem with playing the J from Jxxx as it might be a trick.  Seems to be in second hand as defender when card is played from dummu
        #if self.player_i == 2:
        #    if len(current_trick) == 1:
        #        print("Second hand")

        if self.verbose:
            print(f'Suit adjust: {suit_adjust}, Trump adjust: {trump_adjust}, play_status: {play_status}')
        candidate_cards = []
        
        current_card = 0
        current_insta_score = 0
        # Small cards come from DD, but if a sequence is present it is the highest card
        for card52, (e_tricks, e_score, e_make, msg) in card_dd.items():
            adjust_card = suit_adjust[card52 // 13]
            card32 = deck52.card52to32(card52)
            insta_score = self.get_nn_score(card32, card52, card_nn, play_status, tricks52)
            # Ignore cards not suggested by the NN
            if insta_score < self.models.pimc_trust_NN:
                continue
            if insta_score > self.models.play_reward_threshold_NN and self.models.play_reward_threshold_NN > 0:
                adjust_card += 0.1            
            # If we can take rest we don't adjust, then NN will decide if equal
            # Another option could be to resample the hands without restrictions
            if e_tricks == 13 - trick_i:
                # Calculate valid claim cards
                if card32 // 8 != self.strain_i - 1:
                    adjust_card = 0
            if card52 in bad_play:
                adjust_card += -0.2            
            card = self.create_card(suit_adjust, card52, e_tricks, e_score, e_make, msg, adjust_card, insta_score)
            # For now we want lowest card first - in deck it is from A->2 so highest value is lowest card
            if (card52 > current_card) and (insta_score == current_insta_score) and (card52 // 13 == current_card // 13):
                candidate_cards.insert(0, card)
            else:
                candidate_cards.append(card)

            current_card = card52
            current_insta_score = insta_score

        valid_bidding_samples = np.sum(bidding_scores >= self.sampler.bidding_threshold_sampling)
        if self.models.use_real_imp_or_mp:
            if self.models.matchpoint:
                candidate_cards = sorted(enumerate(candidate_cards), key=lambda x: (x[1].expected_score_mp, x[1].expected_tricks_dd, x[1].insta_score, -x[0]), reverse=True)
                who = "MP-calc"

            else:
                candidate_cards = sorted(enumerate(candidate_cards), key=lambda x: (x[1].expected_score_imp, x[1].expected_tricks_dd, x[1].insta_score, -x[0]), reverse=True)
                who = "IMP-calc"
            candidate_cards = [card for _, card in candidate_cards]
        else:            
            # Now we will select the card to play
            # We have 3 factors, and they could all be right, so we remove most of the decimals
            # We should probably also consider bidding_scores in this 
            # If we have bad quality of samples we should probably just use the neural network
            if valid_bidding_samples >= 0:
                if self.models.matchpoint:
                    #for i in range(len(candidate_cards)):
                    #    print(candidate_cards[i].card, candidate_cards[i].insta_score, int(candidate_cards[i].expected_tricks_dd* 10) / 10, candidate_cards[i].p_make_contract, candidate_cards[i].expected_score_dd, int(candidate_cards[i].expected_tricks_dd * 10) / 10)
                    candidate_cards = sorted(enumerate(candidate_cards), key=lambda x: (round(x[1].expected_score_dd, 0), round(5*x[1].p_make_contract, 1), round(x[1].insta_score, 3), -x[0]), reverse=True)
                    who = "NN-MP"
                    #print("Who", who)
                else:
                    candidate_cards = sorted(enumerate(candidate_cards), key=lambda x: (round(5*x[1].p_make_contract, 1), int(x[1].expected_tricks_dd * 1000) / 1000, round(x[1].expected_score_dd, 0), round(x[1].insta_score, 3), -x[0]), reverse=True)
                    who = "NN-Make"
                    #print("Who", who)
                candidate_cards = [card for _, card in candidate_cards]
                # for i in range(len(candidate_cards)):
                #     print(candidate_cards[i].card, candidate_cards[i].insta_score, int(candidate_cards[i].expected_tricks_dd* 10) / 10, round(5*candidate_cards[i].p_make_contract,1), round(candidate_cards[i].expected_score_dd,0))
            else:
                if self.models.use_biddingquality_in_eval:
                    candidate_cards = sorted(enumerate(candidate_cards), key=lambda x: ( round(x[1].insta_score, 3), round(5*x[1].p_make_contract, 1), int(x[1].expected_tricks_dd * 1900) / 1900, -x[0]), reverse=True)
                    candidate_cards = [card for _, card in candidate_cards]
                    candidate_cards2 = sorted(enumerate(candidate_cards), key=lambda x: (round(x[1].expected_score_dd, 0), round(x[1].insta_score, 3), int(x[1].expected_tricks_dd * 1090) / 1090, -x[0]), reverse=True)
                    candidate_cards2 = [card for _, card in candidate_cards2]
                    if candidate_cards[0].expected_score_dd < 0 and candidate_cards2[0].expected_score_dd:
                        candidate_cards = candidate_cards2
                    who = "DD"
                else:
                    if self.models.matchpoint:
                        candidate_cards = sorted(enumerate(candidate_cards), key=lambda x: (round(x[1].expected_score_dd, 0), round(5*x[1].p_make_contract, 1), round(x[1].insta_score, 3), -x[0]), reverse=True)
                        who = "MP-Make"
                        #print("Who", who)
                    else:
                        candidate_cards = sorted(enumerate(candidate_cards), key=lambda x: (round(5*x[1].p_make_contract, 1), round(x[1].insta_score, 3), int(x[1].expected_tricks_dd * 1009) / 1090, -x[0]), reverse=True)
                        who = "Make"
                        #print("Who", who)
                    candidate_cards = [card for _, card in candidate_cards]

        # Select the right card
        right_card, who = carding.select_right_card_for_play(candidate_cards, self.get_random_generator(), self.contract, self.models, self.hand_str, self.public_hand_str, self.player_i, tricks52, current_trick, missing_cards, play_status, who, claim_cards,  self.verbose)
        best_card_resp = CardResp(
            card=right_card,
            candidates=candidate_cards,
            samples=samples,
            shape=-1,
            hcp=-1, 
            quality=quality,
            who = who, 
            claim = -1 if not claim_cards else 13 - trick_i
        )

        return best_card_resp

    def create_card(self, suit_adjust, card52, e_tricks, e_score, e_make, msg, adjust_card, insta_score):
        card = CandidateCard(
                    card=Card.from_code(card52),
                    insta_score=insta_score,
                    expected_tricks_dd=round(e_tricks + adjust_card,3),
                    p_make_contract=e_make,
                    **({
                        "expected_score_mp": round(e_score + 20 * suit_adjust[card52 // 13],0)
                    } if self.models.matchpoint and self.models.use_real_imp_or_mp else
                    {
                        "expected_score_imp": round(e_score + adjust_card,2)
                    } if not self.models.matchpoint and self.models.use_real_imp_or_mp else
                    {
                        "expected_score_dd": e_score + adjust_card
                    }),
                    msg= (f"|adjust={adjust_card}{msg}" if adjust_card != 0 else msg)
                )
            
        return card
