import time
import sys
import numpy as np
import tensorflow as tf

import binary
import deck52
import calculate
import scoring

from objects import BidResp, CandidateBid
from bidding import bidding
from collections import defaultdict

from util import hand_to_str, calculate_seed, find_vuln_text, save_for_training
from colorama import Fore, Style, init

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
            return "", False, False
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
        
        # We will not try to rescue BBA
        if self.bba_is_controlling:
            return False
        
        if not self.models.check_final_contract:
            return False

        if self.models.tf_version == 1:
            sys.stderr.write("Rescue bid not supported for TF 1.x\n")
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

        # We are in the passout situation
        if passout:
            return True

        # We do not rescue bid before our 3rd bid     
        # unless there has been some preempting
        # We should look at how the bidding continues, as we can't expect partner to bid
        # Could be a sequence like this 1D-3C-X-4C with A97.AT63.AT8652.
        if my_bid_no < 2 :
            return False

        if my_bid_no < 3 :
            if bidding.BID2ID[auction[-1]] < 14:
                return False
          
        # RHO bid, so we will not evaluate rescue bid
        #if (auction[-1] != "PASS"):
        #    return False
        
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
        generate_samples = generate_samples or (binary.get_number_of_bids(auction) > 4 and self.models.check_final_contract)
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
            if self.verbose:
                print(f"No sampling for aution: {auction} trying to find {self.sample_boards_for_auction}")
                print(not self.sampler.no_samples_when_no_search and self.get_min_candidate_score(self.my_bid_no) != -1)
                print(binary.get_number_of_bids(auction) > 4 and self.models.check_final_contract)
                print(len(candidates) > 1)

            sample_count = 0

        # If quality = -1 we should probably just bid what BBA suggest, but even BBA might not have understood the bidding

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
                        if bidding.undisturbed(auction):
                            adjust += self.models.adjust_NN_undisturbed * candidate.insta_score 
                        else:
                            adjust += self.models.adjust_NN * candidate.insta_score

                if self.verbose:
                    print(f"Adjusted for trust in NN {candidate.bid} {adjust:0.3f}")

                if candidate.bid == "X":
                    meaning, alert = self.explain(auction + ["X"])
                    # If we do not have any controls, then adjust the score
                    # If we are void in the suit, then we should not bid X, so we adjust
                    # We also adjust for a singleton
                    if meaning and "penalty" in meaning.lower():
                        trump = bidding.get_strain_i(bidding.get_contract(auction + ["X", "PASS", "PASS", "PASS"]))
                        if self.verbose:
                            print(bidding.get_contract(auction + ["X", "PASS", "PASS", "PASS"]), "trump", trump)
                        if trump > 0:
                            #print(self.hand_bidding)
                            reshaped_array = self.hand_bidding.reshape(-1,int(self.models.n_cards_bidding / 4))
                            suits = reshaped_array.sum(axis=1)
                            aces = np.sum(reshaped_array[:, 0] == 1)
                            kings = np.sum(reshaped_array[:, 1] == 1)
                            controls = 2 * aces + kings
                            if self.verbose:
                                print("trump", trump, "suits", suits, "aces", aces, "kings", kings, "controls", controls)
                            if suits[trump-1] == 1:
                                adjust -= 0.5 * self.models.adjust_X
                            if suits[trump-1] == 0:
                                adjust -= self.models.adjust_X
                            if controls == 0:
                                adjust -= self.models.adjust_X
                            if controls == 1:
                                adjust -= 0.5 * self.models.adjust_X

                    #print("X=",meaning, alert, auction + ["X"], adjust)

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
            # If we found no samples, then we consult BBA if activated
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
            # Split and filter the samples based on your condition
            filtered_samples = [
                sample for sample in samples if float(sample.split(" ")[5]) >= self.models.min_bidding_trust_for_sample_when_rescue
            ]

            # Select random samples from the filtered list
            samples_to_check = self.rng.choice(filtered_samples, min(len(filtered_samples), self.models.max_samples_checked), replace=False)

            for sample in samples_to_check:
                #contract_id = 40
                sample = sample.split(" ")
                if self.verbose:
                    #print(samples[i].split(" ")[(self.seat + 2) % 4])
                    print(sample[(self.seat + 2) % 4], sample[5])
                X = self.get_binary_contract(self.seat, self.vuln, self.hand_str, sample[(self.seat + 2) % 4], self.models.n_cards_bidding)
                # Perhaps we should collect all samples, and just make one call to the neural network
                contracts = self.models.contract_model.pred_fun(X)
                if tf.is_tensor(contracts):
                    contracts = contracts.numpy()
                score = 0
                result = {}
                for i in range(len(contracts[0])):
                    # We should make calculations on this, so 4H, 5H or even 6H is added, if tricks are fine
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
                            if nn_tricks[0][j] > 0.1:
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


                if self.verbose:
                    print("result", result)                    
                if not result:
                    if self.verbose:
                        print("No valid contracts, skipping sample")
                    break

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
                    score = scoring.score(contract + ("X" if doubled else ""), self.vuln[(self.seat ) % 2], tricks)
                    if self.verbose:
                        print(result, score, level, doubled, self.vuln[(self.seat) % 2] )
                    if contract not in alternatives:
                        alternatives[contract] = []
                    alternatives[contract].append({"score": score, "tricks": tricks})
                else:
                    if self.verbose:
                        print("Skipping invalid contract", contract, auction)
                    
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
                        print("expected_score", expected_score, candidates[0].expected_score, self.models.min_rescue_reward)
                    if (expected_score > candidates[0].expected_score + self.models.min_rescue_reward) or (contract_average_tricks[max_count_contract] - expected_tricks > 4):

                        candidatebid = CandidateBid(bid=max_count_contract, insta_score=-1, 
                                                    expected_score=contract_average_scores[max_count_contract], expected_tricks=expected_tricks, adjust=0, alert = False, who="Rescue", explanation="Rescue bid")
                        candidates.insert(0, candidatebid)
                        who = "Rescue"
                        sys.stderr.write(f"Rescuing {current_contract} {contract_counts[max_count_contract]}*{max_count_contract} {contract_average_scores[max_count_contract]:.3f} {contract_average_tricks[max_count_contract]:.2f}\n")
                    else:
                        if self.verbose:
                            print("No rescue, due to low expected score: ", (expected_score, candidates[0].expected_score + self.models.min_rescue_reward))
                else:
                    if self.verbose:
                        print("No rescue, due to low reward: ", (contract_average_scores[max_count_contract], candidates[0].expected_score + self.models.min_rescue_reward, contract_average_tricks[max_count_contract] - expected_tricks))
            else:
                if self.verbose:
                    print("No rescue, due to not enough samples, that suggest bidding: ", total_entries, len(samples), self. models.max_samples_checked)
            if self.verbose:
                print(f"Rescue bid calculation took {(time.time() - t_start):0.4f} seconds")

        else:
            if self.verbose:
                print("No rescue bid evaluated", auction, passout, candidates[0], quality, self.my_bid_no, candidates[0].who)

        explain = candidates[0].explanation
        alert = bool(candidates[0].alert)
        if candidates[0].explanation is None:
            if self.models.consult_bba:
                explain, alert = self.explain(auction + [candidates[0].bid])

        #print("Final candidate", candidates[0].bid , explain)
        # We return the bid with the highest expected score or highest adjusted score 
        return BidResp(bid=candidates[0].bid, candidates=candidates, samples=samples[:self.sample_hands_for_review], shape=p_shp, hcp=p_hcp, who=who, quality=quality, alert = alert, explanation=explain)
    
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
        explanation_partner_bid = None
        explanations = False
        bba_controlled = False
        preempted = False
        if self.models.consult_bba:

            explanations, bba_controlled, preempted = self.explain_auction(auction)
            self.bba_is_controlling = bba_controlled

            explanation_partner_bid, bba_alert = self.explain(auction[:-1])
            bba_bid_resp = self.bbabot.bid(auction)
            if self.bba_is_controlling:
                return [CandidateBid(bid=bba_bid_resp.bid, insta_score=1, alert = True, who="BBA - Keycard sequence", explanation=bba_bid_resp.explanation)], False

            if self.models.use_bba_to_count_aces and self.bbabot.is_key_card_ask(auction, explanation_partner_bid):
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
        #print("no_bids",no_bids, auction[-1])
        min_bid_score = self.get_min_candidate_score(self.my_bid_no)
        passout = False
        if no_bids > 3 and auction[-1] == 'PASS':
            #print(" this is the final pass, so we wil have a second opinion")
            min_candidates = self.models.min_passout_candidates
            # Reduce the score so we search
            min_bid_score = self.get_min_candidate_score(self.my_bid_no + 1)
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
        if no_bids <= 4 and no_bids > 0:
            if preempted:
                min_candidates = 2
                # Reduce the score so we search
                min_bid_score = self.get_min_candidate_score(self.my_bid_no + 2)
                if self.verbose:
                    print("Extra candidate after opponents preempt might be needed")
                if no_bids > 1 and bidding.BID2ID[auction[-2]] > 14:
                    if self.verbose:
                        print("Extra candidate might be needed after partners preempt/bid over preempt")

        while True:
            bid_i = np.argmax(bid_softmax)
            if bid_softmax[bid_i] < min_bid_score:
                if len(candidates) >= min_candidates:
                    break
                # Second candidate to low. Rescuebid should handle 
                if bid_softmax[bid_i] <= 0.01:
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
                candidates.append(CandidateBid(bid=bidding.ID2BID[2], insta_score=0.1, alert = None, Who = "Config", explanation = "Pass after bid count"))

        if self.verbose:
            print("\n".join(str(bid) for bid in candidates))
    
        if self.models.consult_bba:
            if self.verbose:
                print(f"{Fore.CYAN}BBA suggests: {bba_bid_resp.bid} Partners bid {auction[:-1]}: {explanation_partner_bid} {Fore.RESET}")
            
            if explanation_partner_bid:
                if "Forcing" in explanation_partner_bid:
                    # If the opponents bid we are no longer in a forcing situation
                    if auction[-1] == "PASS":
                        for candidate in candidates:
                            if candidate.bid == "PASS":
                                candidate.explanation = "We are not allowed to pass"
                                candidate.insta_score = -1
                if "5NT pick a slam" in explanation_partner_bid:
                    # Do not bid 7
                    for candidate in candidates:
                        if bidding.get_level(candidate.bid) > 6:
                                candidate.explanation = "We are not allowed to bid grandslam"
                                candidate.insta_score = -1
                if "GF" in explanation_partner_bid:
                    # If the opponents bid we are no longer in a forcing situation
                    if auction[-1] == "PASS" and auction[-2] not in ('X','XX'):
                        if not bidding.is_game_bid(auction):
                            for candidate in candidates:
                                if candidate.bid == "PASS":
                                    candidate.explanation = "We are not allowed to pass"
                                    candidate.insta_score = -1
            
            for candidate in candidates:
                if candidate.bid == bba_bid_resp.bid:
                    candidate.alert = bba_bid_resp.alert
                    candidate.explanation = bba_bid_resp.explanation
                    if self.verbose:
                        sys.stderr.write(f"{Fore.CYAN}BBA bid is candidate: {bba_bid_resp.bid} Alert: { bba_bid_resp.alert} Explaination: {bba_bid_resp.explanation} NN score: {candidate.insta_score:.3f}{Fore.RESET}\n")
                    candidate.insta_score += self.models.bba_trust
                    break
            else:
                if self.verbose:
                    sys.stderr.write(f"{Fore.CYAN}Adding BBA bid as candidate: {bba_bid_resp.bid} Alert: { bba_bid_resp.alert} Explaination: {bba_bid_resp.explanation}{Fore.RESET}\n")
                candidates.append(CandidateBid(bid=bba_bid_resp.bid, insta_score=self.models.bba_trust, alert = bba_bid_resp.alert, who="BBA", explanation=bba_bid_resp.explanation))

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

