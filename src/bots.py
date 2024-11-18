import math
import time
import pprint
import sys
import numpy as np
import tensorflow as tf

import binary
import deck52
import calculate
import scoring

from objects import BidResp, CandidateBid, Card, CardResp, CandidateCard
from bidding import bidding
from collections import defaultdict

import carding
from util import hand_to_str, expected_tricks_sd, p_defeat_contract, follow_suit, calculate_seed, find_vuln_text, save_for_training
from colorama import Fore, Back, Style, init

init()
class BotBid:

    def __init__(self, vuln, hand_str, models, sampler, seat, dealer, ddsolver, verbose):
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

    @property
    def bbabot(self):
        if self._bbabot_instance is None:
            from bba.BBA import BBABotBid
            # Initialize the BBABotBid instance with required parameters
            self._bbabot_instance = BBABotBid(
                self.models.bba_ns,
                self.models.bba_ew,
                self.seat,
                self.hand_str,
                self.vuln,
                self.dealer,
                self.models.matchpoint,
                self.verbose
            )
        return self._bbabot_instance
    
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

    def get_binary_contract(self, position, vuln, hand_str, dummy_str):
        X = np.zeros(2 + 2 * 32, dtype=np.float16)

        v_we = vuln[0] if position % 2 == 0 else vuln[1]
        v_them = vuln[1] if position % 2 == 0 else vuln[0]
        vuln = np.array([[v_we, v_them]], dtype=np.float16)
        
        hand = binary.parse_hand_f(32)(hand_str).reshape(32)
        dummy = binary.parse_hand_f(32)(dummy_str).reshape(32)
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
            print("Checking if we should evaluate rescue bid", self.models.check_final_contract, len(samples))
            print("Auction",auction, passout)
            print("Candidate", candidate_bid, quality)
        if not self.models.check_final_contract:
            return False

        # We did not rescue bid before our 3rd bid        
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
            # We will prepare data for calculating recue bid
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


    def bid(self, auction):
        # Validate input
        if (len(auction)) % 4 != self.seat:
            error_message = f"Dealer {self.dealer}, auction {auction}, and seat {self.seat} do not match!"
            raise ValueError(error_message)
        # A problem, that we get candidates with a threshold, and then simulates
        # When going negative, we would probably like to extend the candidates
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
                samples.append('%s %s %s %s %.5f' % (
                    hand_to_str(hands_np[i,0,:],self.models.n_cards_bidding),
                    hand_to_str(hands_np[i,1,:],self.models.n_cards_bidding),
                    hand_to_str(hands_np[i,2,:],self.models.n_cards_bidding),
                    hand_to_str(hands_np[i,3,:],self.models.n_cards_bidding),
                    sorted_score[i]
                ))
            sample_count = hands_np.shape[0]
        else:
            sample_count = 0

        if self.do_rollout(auction, candidates, self.get_max_candidate_score(self.my_bid_no), sample_count):
            ev_candidates = []
            ev_scores = {}
            for candidate in candidates:
                if self.verbose:
                    print(f" {candidate.bid.ljust(4)} {candidate.insta_score:.3f} Samples: {len(hands_np)}")
                auctions_np = self.bidding_rollout(auction, candidate.bid, hands_np)

                t_start = time.time()

                # Initialize variables to None
                decl_tricks_softmax1 = None
                decl_tricks_softmax2 = None
                decl_tricks_softmax3 = None
                
                if self.models.double_dummy_calculator:
                    contracts, decl_tricks_softmax1 = self.expected_tricks_dd(hands_np, auctions_np, self.hand_str)
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
                        samples[idx] += f" \n {auc} ({average_tricks_str}) "
                    else:
                        samples[idx] += f" \n {auc}"

                if self.verbose:
                    print("tricks", np.mean(decoded_tricks))
                expected_tricks = np.mean(decoded_tricks)
    
                # We need to find a way to use how good the samples are
                # Calculate the mean of the expected score
                expected_score = np.mean(ev)
                if self.verbose:
                    print(ev)

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
                
                # Adding some bonus to the bid selected by the neural network
                if hands_np.shape[0] == self.sampler.min_sample_hands_auction:
                    # We only have the minimum number of samples, so they are often of bad quality
                    # So we add more trust to the NN
                    adjust += self.models.adjust_NN_Few_Samples*candidate.insta_score
                else:
                    adjust += self.models.adjust_NN*candidate.insta_score

                if self.verbose:
                    print("Adjust for trust in NN", adjust)
                # These adjustments should probably be configurable
                if passout and candidate.insta_score < self.get_min_candidate_score(self.my_bid_no):
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
                    if candidate.bid == "X" and candidate.insta_score < 0.5:
                        adjust -= self.models.adjust_X
                        if self.verbose:
                            print("Adjusted for double if insta_score to low", adjust)

                # The problem is that with a low score for X the expected bidding can be very wrong
                if candidate.bid == "X" and candidate.insta_score < 0.01:
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
                    # Ajust is in points, we need to change it to some percentage
                    candidates = sorted(ev_candidates_mp_imp, key=lambda c: (c.expected_mp + c.adjust, round(c.insta_score, 2)), reverse=True)
                else:
                    # Ajust is in points, we need to change it to some imps
                    candidates = sorted(ev_candidates_mp_imp, key=lambda c: (c.expected_imp + c.adjust, round(c.insta_score, 2)), reverse=True)
                ev_candidates = ev_candidates_mp_imp
            else:
                # If the samples are bad we just trust the neural network
                if self.models.use_biddingquality  and quality < self.sampler.bidding_threshold_sampling:
                    candidates = sorted(ev_candidates, key=lambda c: (c.insta_score, c.expected_score + c.adjust), reverse=True)
                else:
                    candidates = sorted(ev_candidates, key=lambda c: (c.expected_score + c.adjust, round(c.insta_score, 2)), reverse=True)
            
            who = "Simulation"
            # Print candidates with their relevant information
            if self.verbose:
                for idx, candidate in enumerate(ev_candidates):
                    print(f"{idx}: {candidate}")
            if self.verbose:
                print(f"Estimating took {(time.time() - t_start):0.4f} seconds")
        else:
            who = "NN" if candidates[0].who is None else candidates[0].who
            n_steps = binary.calculate_step_bidding_info(auction)
            p_hcp, p_shp = self.sampler.get_bidding_info(n_steps, auction, self.seat, self.hand_bidding, self.vuln, self.models)
            p_hcp = p_hcp[0]
            p_shp = p_shp[0]
            if self.evaluate_rescue_bid(auction, passout, samples, candidates[0], quality, self.my_bid_no):    
                if self.verbose:
                    print("Updating samples with expected score")    
                # initialize auction vector
                auction_np = np.ones((len(samples), 64), dtype=np.int32) * bidding.BID2ID['PAD_END']
                for i, bid in enumerate(auction):
                    auction_np[:,i] = bidding.BID2ID[bid]

                contracts, decl_tricks_softmax = self.expected_tricks_dd(hands_np, auction_np, self.hand_str)
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
            print(candidates[0].bid, "selected")

        if self.evaluate_rescue_bid(auction, passout, samples, candidates[0], quality, self.my_bid_no):    

            # We will avoid rescuing if we have a score of max_estimated_score or more
            t_start = time.time()
            alternatives = {}
            current_contract = bidding.get_contract(auction)
            isdoubled = current_contract[-1] == "X" 
            current_contract = current_contract[0:2]
            if self.verbose:
                print("check_final_contract, current_contract", current_contract)
            break_outer = False
            for i in range(min(len(samples), self.models.max_samples_checked)):
                sample = samples[i].split(" ")
                if self.verbose:
                    #print(samples[i].split(" ")[(self.seat + 2) % 4])
                    print(sample[(self.seat + 2) % 4], sample[4])
                if float(sample[4]) < self.models.min_bidding_trust_for_sample_when_rescue:
                    if self.verbose: 
                        print("Skipping sample due to threshold", self.models.min_bidding_trust_for_sample_when_rescue)
                    continue
                X = self.get_binary_contract(self.seat, self.vuln, self.hand_str, sample[(self.seat + 2) % 4])
                # Perhaps we should collect all samples, and just make one call to the neural network
                contract_id, doubled, tricks, score = self.models.contract_model.model[0](X)
                if tf.is_tensor(score):
                    score = score.numpy()
                if tf.is_tensor(tricks):
                    tricks = tricks.numpy()

                contract_id = int(contract_id)
                doubled = bool(doubled)
                contract = bidding.ID2BID[contract_id] 
                if score < self.models.min_bidding_trust_for_sample_when_rescue:
                    if self.verbose:
                        print(self.hand_str, [sample[(self.seat + 2) % 4]])
                        print(f"Skipping sample below level{self.models.min_bidding_trust_for_sample_when_rescue} {contract}{'X' if doubled else ''}-{tricks} score {score:.3f}")
                    continue

                while not bidding.can_bid(contract, auction) and contract_id < 35:
                    contract_id += 5
                    contract = bidding.ID2BID[contract_id] 
                    
                if self.verbose: 
                    print(contract, doubled, tricks)
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
                if current_contract == "3N" and (contract == "4N" or contract == "5N"):
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
                    # Consider autodouble any contract going down
                    if tricks >= level + 6:
                        doubled = False
                    score = scoring.score(contract + ("X" if doubled else ""), self.vuln, tricks)
                    if self.verbose:
                        print(result, score)
                    if contract not in alternatives:
                        alternatives[contract] = []
                    alternatives[contract].append({"score": score, "tricks": tricks})
                    
            # Only if at least 75 of the samples suggest bidding check the score for the rescue bid
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
                    new_auction = auction.copy()
                    new_auction.append(max_count_contract)
                    auction_np = np.ones((len(samples), 64), dtype=np.int32) * bidding.BID2ID['PAD_END']
                    for i, bid in enumerate(new_auction):
                        auction_np[:,i] = bidding.BID2ID[bid]

                    # Simulate the hands
                    contracts, decl_tricks_softmax = self.expected_tricks_dd(hands_np[:min(len(samples), self.models.max_samples_checked)], auction_np, self.hand_str)
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
                                                    expected_score=contract_average_scores[max_count_contract], expected_tricks=expected_tricks, adjust=0, alert = False)
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
        return BidResp(bid=candidates[0].bid, candidates=candidates, samples=samples[:self.sample_hands_for_review], shape=p_shp, hcp=p_hcp, who=who, quality=quality, alert = bool(candidates[0].alert), explanation=None)
    
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
                print("We found no samples, so will just trust the NN")
            return False
        return True

    def get_bid_candidates(self, auction):
        if self.models.use_bba_to_count_aces:
            kc_resp = self.bbabot.is_key_card_ask(auction)
            if kc_resp != None:
                if self.verbose:
                    print("Keycards: ", kc_resp)
                return [CandidateBid(bid=kc_resp.bid, insta_score=1, alert = True)], False

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
            if bidding.BID2ID[auction[0]] > 14:
                min_candidates = 2
                if self.verbose:
                    print("Extra candidate after opponents preempt might be needed")
                elif no_bids > 1 and bidding.BID2ID[auction[1]] > 14:
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
                candidates.append(CandidateBid(bid=bidding.ID2BID[2], insta_score=0.1, alert = None))

        if self.verbose:
            print("\n".join(str(bid) for bid in candidates))

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
            bid_np, next_state = self.models.bidder_model.model(x, self.state)
            bid_np = bid_np[0]
            self.state = next_state
        if self.models.model_version == 1 :
            bid_np = self.models.bidder_model.model_seq(x)
            if self.models.alert_supported:
                alerts = bid_np[1][-1:][0]                   
            bid_np = bid_np[0][-1:][0]
        if self.models.model_version == 2:
            bid_np = self.models.bidder_model.model_seq(x)
            if self.models.alert_supported:
                alerts = bid_np[1][-1:][0]                   
            bid_np = bid_np[0][-1:][0]
        if self.models.model_version == 3:
            bid_np, alerts = self.models.bidder_model.model_seq(x)
            alerts = alerts.numpy()[0]
            bid_np = bid_np.numpy()[0]
            alerts = alerts[-1:][0]
            bid_np = bid_np[-1:][0]
        assert len(bid_np) == 40, "Wrong Result: " + str(bid_np.shape)
        return bid_np, alerts
    
    def sample_hands_for_auction(self, auction_so_far, turn_to_bid):
                # Reset randomizer
        self.rng = self.get_random_generator()

        accepted_samples, sorted_scores, p_hcp, p_shp, quality, samplings = self.sampler.generate_samples_iterative(auction_so_far, turn_to_bid, self.sampler.sample_boards_for_auction, self.sampler.sample_hands_auction, self.rng, self.hand_str, self.vuln, self.models)

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
                    sys.stderr.write(f"{Fore.YELLOW}Warning: Not enough samples found. Using all samples {accepted_samples.shape[0]}, Samplings={samplings}, Auction{auction_so_far}{Style.RESET_ALL}\n")

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

    def bidding_rollout(self, auction_so_far, candidate_bid, hands_np):
        auction = [*auction_so_far, candidate_bid]
        #print("auction: ", auction)
        n_samples = hands_np.shape[0]
        if self.verbose:
            print("bidding_rollout - n_samples: ", n_samples)
        assert n_samples > 0
        
        n_steps_vals = [0, 0, 0, 0]
        for i in range(1, 5):
            n_steps_vals[(len(auction_so_far) % 4 + i) % 4] = self.get_bid_number_for_player_to_bid(auction_so_far + ['?'] * i)  
        
        # initialize auction vector
        auction_np = np.ones((n_samples, 64), dtype=np.int32) * bidding.BID2ID['PAD_END']
        for i, bid in enumerate(auction):
            auction_np[:,i] = bidding.BID2ID[bid]

        bid_i = len(auction) - 1
        turn_i = len(auction) % 4

        # Now we bid each sample to end of auction
        while not np.all(auction_np[:,bid_i] == bidding.BID2ID['PAD_END']):
            #print("bidding_rollout - n_steps_vals: ", n_steps_vals, " turn_i: ", turn_i, " bid_i: ", bid_i, " auction: ", auction)
            X = binary.get_auction_binary_sampling(n_steps_vals[turn_i], auction_np, turn_i, hands_np[:,turn_i,:], self.vuln, self.models, self.models.n_cards_bidding)
            if turn_i % 2 == 0:
                x_bid_np, _ = self.models.bidder_model.model_seq(X)
            else:
                x_bid_np, _ = self.models.opponent_model.model_seq(X)
            
            if self.models.model_version < 3:
                x_bid_np = x_bid_np.reshape((n_samples, n_steps_vals[turn_i], -1))
            else:
                x_bid_np = x_bid_np.numpy()
                
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
                            

                        assert bid_i <= 60, f'Auction to long {bid_i} {auction} {auction_np[i]}'
                    else:
                        bid_np[i][1] = 1

            bid_i += 1
            #print("Adding", bidding.ID2BID[np.argmax(bid_np, axis=1)])
            #print("Adding", np.argmax(bid_np, axis=1))
            #print(bid_np)
            auction_np[:,bid_i] = np.argmax(bid_np, axis=1)
            n_steps_vals[turn_i] += 1
            turn_i = (turn_i + 1) % 4
        assert len(auction_np) > 0
        
        if self.verbose:
            print("bidding_rollout - finished ",auction_np.shape)
        
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
                
        lead_softmax = self.models.lead_suit_model.model(X_ftrs, B_ftrs)
        
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

        decl_tricks_softmax = self.models.sd_model.model(X_sd)
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
        
        decl_tricks_softmax = self.models.sd_model_no_lead.model(X_sd)
        return contracts, decl_tricks_softmax

    def expected_tricks_dd(self, hands_np, auctions_np, hand_str):
        n_samples = hands_np.shape[0]

        decl_tricks_softmax = np.zeros((n_samples, 14), dtype=np.int32)
        contracts = []
        t_start = time.time()

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

            # Create PBN for hand
            # deck = 'N:' + ' '.join(deck52.handxxto52str(hand,self.models.n_cards_bidding) if j != self.seat else hand_str for j, hand in enumerate(hands_np[i]))
            # We want to rotate the hands such that the hand_str comes first, and the remaining hands follow in their original order, wrapping around. 
            # This is to ensure that we get the same DD results for a rotateded deal.
            deck = 'N:' + ' '.join(
                hand_str if j == 0 else deck52.handxxto52str(hands_np[i][(j + self.seat) % 4], self.models.n_cards_bidding)
                for j in range(4)
            )
            leader = (leader + 4 - self.seat) % 4

            # Create PBN including pips
            hands_pbn = [deck52.convert_cards(deck,0, hand_str, self.rng, self.models.n_cards_bidding)]

            # It will probably improve performance if all is calculated in one go
            dd_solved = self.ddsolver.solve(strain, leader, [], hands_pbn, 1)

            decl_tricks_softmax[i,13 - dd_solved["max"][0]] = 1

        if self.verbose:
            print(f'dds took: {(time.time() - t_start):0.4f}')
        return contracts, decl_tricks_softmax
        

    def expected_score(self, turn_to_bid, contracts, decl_tricks_softmax):
        n_samples = len(contracts)
        scores_by_trick = np.zeros((n_samples, 14))
        for i, contract in enumerate(contracts):
            if contract != "pass":
                scores_by_trick[i] = scoring.contract_scores_by_trick(contract, tuple(self.vuln))
                decl_i = 'NESW'.index(contract[-1])
                if (turn_to_bid + decl_i) % 2 == 1:
                    # the other side is playing the contract
                    scores_by_trick[i,:] *= -1
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
        #print("Fetching random generator for lead", self.hash_integer)
        return np.random.default_rng(self.hash_integer)

    def find_opening_lead(self, auction):
        # Validate input
        # We should check that auction match, that we are on lead
        t_start = time.time()
        lead_card_indexes, lead_softmax = self.get_opening_lead_candidates(auction)
        accepted_samples, sorted_bidding_score, tricks, p_hcp, p_shp, quality = self.simulate_outcomes_opening_lead(auction, lead_card_indexes)
        contract = bidding.get_contract(auction)
        scores_by_trick = scoring.contract_scores_by_trick(contract, tuple(self.vuln))

        candidate_cards = []
        expected_tricks_sd = None
        expected_tricks_dd = None
        expected_score_sd = None
        expected_score_dd = None
        expected_score_mp = None
        expected_score_imp = None
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
                    print(real_scores)
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
                    expected_score_sd = expected_score_sd,
                    expected_score_dd = expected_score_dd,
                    expected_score_mp = expected_score_mp,
                    expected_score_imp = expected_score_imp
                ))

        else:        
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
                    expected_score_imp = expected_score_imp
                ))

        candidate_cards = sorted(candidate_cards, key=lambda c: c.insta_score, reverse=True)

        if candidate_cards[0].insta_score > self.models.lead_accept_nn:
            opening_lead = candidate_cards[0].card.code() 
            who = "lead_accept_nn"
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
            print("Samples quality:", quality)
            for card in candidate_cards:
                print(card)
        if opening_lead % 8 == 7:
            contract = bidding.get_contract(auction)
            # Implement human carding here
            opening_lead52 = carding.select_right_card(self.hand52, opening_lead, self.get_random_generator(), contract, self.models, self.verbose)
        else:
            opening_lead52 = deck52.card32to52(opening_lead)

        samples = []
        if self.verbose:
            print(f"Accepted samples for opening lead: {accepted_samples.shape[0]}")
        for i in range(min(self.models.sample_hands_for_review, accepted_samples.shape[0])):
            samples.append('%s %s %s %s %.5f' % (
                hand_to_str(self.handplay),
                hand_to_str(accepted_samples[i,0,:], self.models.n_cards_bidding),
                hand_to_str(accepted_samples[i,1,:], self.models.n_cards_bidding),
                hand_to_str(accepted_samples[i,2,:], self.models.n_cards_bidding),
                sorted_bidding_score[i]
            ))

        if self.verbose:
            print(' Opening lead found in {0:0.1f} seconds.'.format(time.time() - t_start))
        return CardResp(
            card=Card.from_code(opening_lead52),
            candidates=candidate_cards,
            samples=samples, 
            shape=p_shp,
            hcp=p_hcp,
            quality=quality,
            who=who
        )

    def get_opening_lead_candidates(self, auction):
        x_ftrs, b_ftrs = binary.get_auction_binary_for_lead(auction, self.handbidding, self.handplay, self.vuln, self.dealer, self.models)
        contract = bidding.get_contract(auction)
        if contract[1] == "N":
            lead_softmax = self.models.lead_nt_model.model(x_ftrs, b_ftrs)
        else:
            lead_softmax = self.models.lead_suit_model.model(x_ftrs, b_ftrs)

        if tf.is_tensor(lead_softmax):
            lead_softmax = lead_softmax.numpy()

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

    def simulate_outcomes_opening_lead(self, auction, lead_card_indexes):
        t_start = time.time()
        contract = bidding.get_contract(auction)

        decl_i = bidding.get_decl_i(contract)
        lead_index = (decl_i + 1) % 4
                # Reset randomizer
        self.rng = self.get_random_generator()

        accepted_samples, sorted_scores, p_hcp, p_shp, quality, samplings = self.sampler.generate_samples_iterative(auction, lead_index, self.sampler.sample_boards_for_auction_opening_lead, self.sampler.sample_hands_opening_lead, self.rng, self.hand_str, self.vuln, self.models)

        if self.verbose:
            print(f"Generated samples: {accepted_samples.shape[0]} in {samplings} samples. Quality {quality}")
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
                print("Opening lead being examined: ", Card.from_code(opening_lead52), n_accepted)
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

            tricks_softmax = self.models.sd_model.model(X_sd)
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
        if self.strain_i > 0 and ((player_i == 1) or (player_i == 3)):
            #print(self.strain_i)
            #print(binary.get_shape_array(self.hand52))
            self.missing_trump = 13 - binary.get_shape_array(self.hand52)[self.strain_i-1] - binary.get_shape_array(self.public52)[self.strain_i-1]

        self.verbose = verbose
        self.level = int(contract[0])
        self.init_x_play(binary.parse_hand_f(32)(public_hand_str), self.level, self.strain_i)
        self.dds = ddsolver
        self.sampler = sampler
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
        #print("Fetching random generator for player", self.hash_integer)
        return np.random.default_rng(self.hash_integer)
    
    def init_x_play(self, public_hand, level, strain_i):
        self.x_play = np.zeros((1, 13, 298))
        binary.BinaryInput(self.x_play[:,0,:]).set_player_hand(self.hand32)
        binary.BinaryInput(self.x_play[:,0,:]).set_public_hand(public_hand)
        self.x_play[:,0,292] = level
        self.x_play[:,0,293+strain_i] = 1

    def set_real_card_played(self, card52, played_by, openinglead=False):
        if (played_by == 0 or played_by == 2) and (self.player_i == 1 or self.player_i == 3):
            if (card52 // 13) + 1 == self.strain_i:
                self.missing_trump -= 1
        # Dummy has no PIMC
        if self.pimc and self.player_i != 1:
            self.pimc.set_card_played(card52, played_by, openinglead)

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
        # print("Updating constraints", player_i)
        if player_i == 1:
            idx1 = 0
            idx2 = 2
        if player_i == 0:
            idx1 = 2
            idx2 = 3
        if player_i == 2:
            idx1 = 0
            idx2 = 3
        if player_i == 3:
            idx1 = 0
            idx2 = 2
        h1 = []
        h3 = []
        s1 = []
        s3 = []
        for i in range(players_states[0].shape[0]):
            h1.append(binary.get_hcp(hand = np.array(players_states[idx1][i, 0, :32].astype(int)).reshape(1,32)))
            s1.append(binary.get_shape(hand = np.array(players_states[idx1][i, 0, :32].astype(int)).reshape(1,32))[0])
            h3.append(binary.get_hcp(hand = np.array(players_states[idx2][i, 0, :32].astype(int)).reshape(1,32)))
            s3.append(binary.get_shape(hand = np.array(players_states[idx2][i, 0, :32].astype(int)).reshape(1,32))[0])
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
        self.pimc_declaring = self.models.pimc_use_declaring and trick_i  >= (self.models.pimc_start_trick_declarer - 1)
        self.pimc_defending = self.models.pimc_use_defending and trick_i  >= (self.models.pimc_start_trick_defender - 1)
        if not self.pimc_defending and not self.pimc_declaring:
            return
        if self.models.pimc_constraints:
            if self.models.pimc_constraints_each_trick:
                self.find_and_update_constraints(players_states, quality,self.player_i)
            else:
                if self.pimc_declaring and (self.player_i == 1 or self.player_i == 3):
                    if trick_i >= (self.models.pimc_start_trick_declarer - 1) and not self.pimc.constraints_updated:
                        if self.verbose:
                            print("Declaring", self.pimc_declaring, self.player_i, trick_i)
                        self.find_and_update_constraints(players_states, quality,self.player_i)
                if self.pimc_defending and (self.player_i == 0 or self.player_i == 2):
                    if trick_i >= (self.models.pimc_start_trick_defender - 1) and not self.pimc.constraints_updated:
                        if self.verbose:
                            print("Defending", self.pimc_declaring, self.player_i, trick_i)
                        self.find_and_update_constraints(players_states, quality,self.player_i)

    def merge_candidate_cards(self, pimc_resp, dd_resp, engine, weight, quality):
        merged_cards = {}

        if quality < self.models.pimc_bidding_quality:
            weight = 1

        for card52, (e_tricks, e_score, e_make) in dd_resp.items():
            pimc_e_tricks, pimc_e_score, pimc_e_make, pimc_msg = pimc_resp[card52]
            new_e_tricks = round((pimc_e_tricks * weight + e_tricks * (1-weight)),2) if pimc_e_tricks is not None and e_tricks is not None else None
            new_e_score = round((pimc_e_score * weight + e_score * (1-weight)),2) if pimc_e_score is not None and e_score is not None else None
            new_e_make = round((pimc_e_make * weight + e_make * (1-weight)),2) if pimc_e_make is not None and e_make is not None else None
            #print(new_e_tricks, new_e_score, new_e_make)
            #print(pimc_e_tricks, pimc_e_score, pimc_e_make)
            #print(e_tricks, e_score, e_make)
            new_msg = engine + f" {weight*100:.0f}%|" + (pimc_msg or '') 
            new_msg += f"|{pimc_e_tricks:.2f} {pimc_e_score:.2f} {pimc_e_make:.2f}"
            new_msg += f"|BEN DD {(1-weight)*100:.0f}%|" 
            new_msg += f"{e_tricks:.2f} {e_score:.2f} {e_make:.2f}"
            merged_cards[card52] = (new_e_tricks, new_e_score, new_e_make, new_msg)

        return merged_cards
    
    def play_card(self, trick_i, leader_i, current_trick52, tricks52, players_states, bidding_scores, quality, probability_of_occurence, shown_out_suits, play_status, lead_scores, play_scores):
        t_start = time.time()
        current_trick = [deck52.card52to32(c) for c in current_trick52]
        samples = []

        for i in range(min(self.sample_hands_for_review, players_states[0].shape[0])):
            samples.append('%s %s %s %s | %.5f %.5f %.5f %.5f ' % (
                hand_to_str(players_states[0][i,0,:32].astype(int)),
                hand_to_str(players_states[1][i,0,:32].astype(int)),
                hand_to_str(players_states[2][i,0,:32].astype(int)),
                hand_to_str(players_states[3][i,0,:32].astype(int)),
                bidding_scores[i],
                probability_of_occurence[i],
                lead_scores[i],
                play_scores[i]
            ))
        if quality < 0.1 and self.verbose:
            print(samples)

        if self.pimc_declaring and (self.player_i == 1 or self.player_i == 3):
            pimc_resp_cards = self.pimc.nextplay(self.player_i, shown_out_suits)
            if self.verbose:
                print("PIMC result:",pimc_resp_cards)
            assert pimc_resp_cards is not None, "PIMC result is None"
            if self.models.pimc_ben_dd_declaring:
                #print(pimc_resp_cards)
                dd_resp_cards = self.get_cards_dd_evaluation(trick_i, leader_i, tricks52, current_trick52, players_states, probability_of_occurence)
                #print(dd_resp_cards)
                merged_card_resp = self.merge_candidate_cards(pimc_resp_cards, dd_resp_cards, "PIMC", self.models.pimc_ben_dd_declaring_weight, quality)
            else:
                merged_card_resp = pimc_resp_cards
            card_resp = self.pick_card_after_pimc_eval(trick_i, leader_i, current_trick, tricks52, players_states, merged_card_resp, bidding_scores, quality, samples, play_status)            
        else:
            if self.pimc_defending and (self.player_i == 0 or self.player_i == 2):
                pimc_resp_cards = self.pimc.nextplay(self.player_i, shown_out_suits)
                if self.verbose:
                    print("PIMC result:",pimc_resp_cards)
                assert pimc_resp_cards is not None, "PIMCDef result is None"
                if self.models.pimc_ben_dd_defending:
                    #print(pimc_resp_cards)
                    dd_resp_cards = self.get_cards_dd_evaluation(trick_i, leader_i, tricks52, current_trick52, players_states, probability_of_occurence)
                    #print(dd_resp_cards)
                    merged_card_resp = self.merge_candidate_cards(pimc_resp_cards, dd_resp_cards, "PIMCDef", self.models.pimc_ben_dd_defending_weight, quality)
                else:
                    merged_card_resp = pimc_resp_cards
                card_resp = self.pick_card_after_pimc_eval(trick_i, leader_i, current_trick, tricks52, players_states, merged_card_resp, bidding_scores, quality, samples, play_status)            
                
            else:
                dd_resp_cards = self.get_cards_dd_evaluation(trick_i, leader_i, tricks52, current_trick52, players_states, probability_of_occurence)
                card_resp = self.pick_card_after_dd_eval(trick_i, leader_i, current_trick, tricks52, players_states, dd_resp_cards, bidding_scores, quality, samples, play_status)

        if self.verbose:
            print(f'Play card response time: {time.time() - t_start:0.4f}')
        return card_resp

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

        hands_pbn = []
        for i in range(n_samples):
            hands = [None, None, None, None]
            for j in range(4):
                self.get_random_generator().shuffle(pips[j])
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
            print(hands_pbn[:10])

        t_start = time.time()
        if self.verbose:
            print("Samples:", n_samples, " Solving:",len(hands_pbn), self.strain_i, leader_i, current_trick52)
        
        dd_solved = self.dds.solve(self.strain_i, leader_i, current_trick52, hands_pbn, 3)

        # if defending the target is another
        level = int(self.contract[0])
        if self.player_i % 2 == 1:
            tricks_needed = level + 6 - self.n_tricks_taken
        else:
            tricks_needed = 13 - (level + 6) - self.n_tricks_taken + 1

        # print("Calculated tricks")
        if self.models.use_probability:
            card_tricks = self.dds.expected_tricks_dds_probability(dd_solved, probabilities_list)
        else:
            card_tricks = self.dds.expected_tricks_dds(dd_solved)

        making = self.dds.p_made_target(tricks_needed)(dd_solved)

        if self.models.use_real_imp_or_mp:
            if self.verbose:
                print("probabilities")
                print(probabilities_list)
                print("DD Result")
                print(dd_solved)
            # print("Calculated scores")
            real_scores = calculate.calculate_score(dd_solved, self.n_tricks_taken, self.player_i, self.score_by_tricks_taken)
            if self.verbose:
                print("Real scores")
                print(real_scores)
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
        for key in dd_solved.keys():
            card_result[key] = (card_tricks[key], card_ev[key], making[key])
            if self.verbose:
                print(f'{deck52.decode_card(key)} {card_tricks[key]:0.3f} {card_ev[key]:5.2f} {making[key]:0.2f}')

        if self.verbose:
            print(f'dds took: {(time.time() - t_start):0.4f}')

        return card_result
    
    
    def next_card_softmax(self, trick_i):
        if self.verbose:
            print("next_card_softmax", self.playermodel.name, trick_i)

        cards_softmax = self.playermodel.next_cards_softmax(self.x_play[:,:(trick_i + 1),:])
        assert cards_softmax.shape == (1, 32), f"Expected shape (1, 32), but got shape {cards_softmax.shape}"

        if tf.is_tensor(cards_softmax):
            cards_softmax = cards_softmax.numpy()
        x = follow_suit(
            cards_softmax,
            binary.BinaryInput(self.x_play[:,trick_i,:]).get_player_hand(),
            binary.BinaryInput(self.x_play[:,trick_i,:]).get_this_trick_lead_suit(),
        )
        return x.reshape(-1)
    
    def calculate_trump_adjust(self, play_status):
        trump_adjust = 0
        # Only in suit contract and if we are on lead and we are declaring
        if self.strain_i != 0 and play_status == "Lead" and (self.player_i == 1 or self.player_i == 3):
            # Any outstanding trump?
            if self.models.draw_trump_reward > 0 and self.missing_trump > 0:
                trump_adjust = self.models.draw_trump_reward
            # Just to be sure we wont to show opps that they have no trump
            if self.models.draw_trump_penalty > 0 and self.missing_trump == 0:
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

    def pick_card_after_pimc_eval(self, trick_i, leader_i, current_trick, tricks52,  players_states, card_dd, bidding_scores, quality, samples, play_status):
        t_start = time.time()
        card_softmax = self.next_card_softmax(trick_i)
        if self.verbose:
            print(f'Next card response time: {time.time() - t_start:0.4f}')

        all_cards = np.arange(32)
        ## This could be a parameter, but only used for display purposes
        s_opt = card_softmax >= self.models.pimc_trust_NN

        card_options, card_scores = all_cards[s_opt], card_softmax[s_opt]

        card_nn = {c:s for c, s in zip(card_options, card_scores)}
        
        trump_adjust = self.calculate_trump_adjust(play_status)

        candidate_cards = []
        
        for card52, (e_tricks, e_score, e_make, msg) in card_dd.items():
            card32 = deck52.card52to32(card52)
            insta_score = self.get_nn_score(card32, card52, card_nn, play_status, tricks52)
            # Ignore cards not suggested by the NN
            if insta_score < self.models.pimc_trust_NN:
                continue

            expected_score = round(e_score + (trump_adjust if (card32 // 8) + 1 == self.strain_i else 0), 0)

            candidate_cards.insert(0,CandidateCard(
                card=Card.from_code(card52),
                insta_score=round(insta_score,3),
                expected_tricks_dd=round(e_tricks + (trump_adjust if (card32 // 8) + 1 == self.strain_i else 0),3),
                p_make_contract=e_make,
                **({
                    "expected_score_mp": expected_score
                } if self.models.matchpoint and self.models.use_real_imp_or_mp else
                {
                    "expected_score_imp": round(e_score + (trump_adjust if (card32 // 8) + 1 == self.strain_i else 0),2)
                } if not self.models.matchpoint and self.models.use_real_imp_or_mp else
                {
                    "expected_score_dd": e_score + (trump_adjust if (card32 // 8) + 1 == self.strain_i else 0)
                }),
                msg=msg + (f"|trump adjust={trump_adjust}" if trump_adjust != 0 and (card32 // 8) + 1 == self.strain_i else "")
            ))


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
                print(candidate_cards[i].card, candidate_cards[i].insta_score, candidate_cards[i].expected_tricks_dd, round(5*candidate_cards[i].p_make_contract, 1), candidate_cards[i].expected_score_dd, int(candidate_cards[i].expected_tricks_dd * 10) / 10)

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
            
        right_card, who = carding.select_right_card_for_play(candidate_cards, self.get_random_generator(), self.contract, self.models, self.hand_str, self.public_hand_str, self.player_i, tricks52, current_trick, play_status, who, self.verbose)
        best_card_resp = CardResp(
            card=right_card,
            candidates=candidate_cards,
            samples=samples,
            shape=-1,
            hcp=-1, 
            quality=quality,
            who = who
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
                    return 1

        return round(card_nn.get(card32, 0),3)

    def pick_card_after_dd_eval(self, trick_i, leader_i, current_trick, tricks52, players_states, card_dd, bidding_scores, quality, samples, play_status):
        t_start = time.time()
        card_softmax = self.next_card_softmax(trick_i)
        if self.verbose:
            print(f'Next card response time: {time.time() - t_start:0.4f}')

        all_cards = np.arange(32)
        ## This could be a parameter, but only used for display purposes
        s_opt = card_softmax > 0.001

        card_options, card_scores = all_cards[s_opt], card_softmax[s_opt]

        card_nn = {c:s for c, s in zip(card_options, card_scores)}

        trump_adjust = self.calculate_trump_adjust(play_status)

        candidate_cards = []
        
        current_card = 0
        current_insta_score = 0
        # Small cards come from DD, but if a sequence is present it is the highest card
        for card52, (e_tricks, e_score, e_make) in card_dd.items():
            card32 = deck52.card52to32(card52)
            card=Card.from_code(card52)
            insta_score = self.get_nn_score(card32, card52, card_nn, play_status, tricks52)
            # For now we want lowest card first - in deck it is from A->2 so highest value is lowest card
            expected_score =round(e_score+ (trump_adjust * 20 if (card32 // 8) + 1 == self.strain_i else 0),0)
            # Ignore cards bot suggested by the NN
            if insta_score < self.models.trust_NN:
                continue
            if (card52 > current_card) and (insta_score == current_insta_score) and (card52 // 13 == current_card // 13):
                candidate_cards.insert(0, CandidateCard(
                    card=card,
                    insta_score=insta_score,
                    expected_tricks_dd=round(e_tricks + (trump_adjust if (card32 // 8) + 1 == self.strain_i else 0),3),
                    p_make_contract=e_make,
                    **({
                        "expected_score_mp": expected_score
                    } if self.models.matchpoint and self.models.use_real_imp_or_mp else
                    {
                        "expected_score_imp": round(e_score + (trump_adjust if (card32 // 8) + 1 == self.strain_i else 0),2)
                    } if not self.models.matchpoint and self.models.use_real_imp_or_mp else
                    {
                        "expected_score_dd": e_score + (trump_adjust if (card32 // 8) + 1 == self.strain_i else 0)
                    }),
                    msg= (f"trump adjust={trump_adjust}" if trump_adjust != 0 and (card32 // 8) + 1 == self.strain_i else "")
                ))
            else:
                candidate_cards.append(CandidateCard(
                    card=card,
                    insta_score=insta_score,
                    expected_tricks_dd=round(e_tricks + (trump_adjust if (card32 // 8) + 1 == self.strain_i else 0),3),
                    p_make_contract=e_make,
                    **({
                        "expected_score_mp": expected_score
                    } if self.models.matchpoint and self.models.use_real_imp_or_mp else
                    {
                        "expected_score_imp": round(e_score + (trump_adjust*2 if (card32 // 8) + 1 == self.strain_i else 0),2)
                    } if not self.models.matchpoint and self.models.use_real_imp_or_mp else
                    {
                        "expected_score_dd": e_score + (trump_adjust if (card32 // 8) + 1 == self.strain_i else 0)
                    }),
                    msg= (f"trump adjust={trump_adjust}" if trump_adjust != 0 and (card32 // 8) + 1 == self.strain_i else "")
                ))
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
                    print("Who", who)
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
        right_card, who = carding.select_right_card_for_play(candidate_cards, self.get_random_generator(), self.contract, self.models, self.hand_str, self.public_hand_str, self.player_i, tricks52, current_trick, play_status, who, self.verbose)
        best_card_resp = CardResp(
            card=right_card,
            candidates=candidate_cards,
            samples=samples,
            shape=-1,
            hcp=-1, 
            quality=quality,
            who = who
        )

        return best_card_resp
