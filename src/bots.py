import time
import pprint
import sys
import numpy as np

import binary
import deck52
import scoring

from objects import BidResp, CandidateBid, Card, CardResp, CandidateCard
from bidding import bidding
from binary import parse_hand_f
from pimc.PIMC import BGADLL

from util import hand_to_str, expected_tricks_sd, p_defeat_contract, follow_suit, calculate_seed


class BotBid:

    def __init__(self, vuln, hand_str, models, sampler, seat, dealer, verbose):
        self.vuln = vuln
        self.hand_str = hand_str
        self.hand = binary.parse_hand_f(32)(hand_str)
        # Perhaps it is an idea to store the auction (and binary version) to speed up processing
        self.min_candidate_score = models.search_threshold
        self.max_candidate_score = models.no_search_threshold
        self.ns = models.ns
        self.ew = models.ew
        self.sameforboth = models.sameforboth
        self.seat = seat
        self.dealer = dealer
        self.bidder_model = models.bidder_model
        self.binfo_model = models.binfo_model
        self.lead_suit_model = models.lead_suit_model
        self.lead_nt_model = models.lead_nt_model
        self.sd_model = models.sd_model
        self.sd_model_no_lead = models.sd_model_no_lead
        self.sample = sampler
        self.verbose = verbose
        self.sample_boards_for_auction = sampler.sample_boards_for_auction
        self.lead_included = models.lead_included
        self.double_dummy_eval = models.double_dummy_eval
        self.samples = []
        self.eval_after_bid_count = models.eval_after_bid_count
        self.sample_hands_for_review = models.sample_hands_for_review
        self.use_biddingquality = models.use_biddingquality
        self.hash_integer  = calculate_seed(hand_str)         
        if self.verbose:
            print(f"Setting seed (Sampling bidding info) from {hand_str}: {self.hash_integer}")
        self.rng = np.random.default_rng(self.hash_integer)

    @staticmethod
    def get_bid_number_for_player_to_bid(auction):
        hand_i = len(auction) % 4
        i = hand_i
        while i < len(auction) and auction[i] == 'PAD_START':
            i += 4
        n_step =  1 + (len(auction) - i) // 4
        #print("get_bid_number_for_player_to_bid: ", hand_i, n_step, auction)
        return n_step

    def get_binary(self, auction):
        n_steps = BotBid.get_bid_number_for_player_to_bid(auction)
        hand_ix = len(auction) % 4
        X = binary.get_auction_binary(n_steps, auction, hand_ix, self.hand, self.vuln, self.ns, self.ew)
        return X

    def bid(self, auction):
        # A problem, that we get candidates with a threshold, and then simulates
        # When going negative, we would probably like to extend the candidates
        candidates, passout = self.get_bid_candidates(auction)
        good_quality = None

        if BotBid.do_rollout(auction, candidates, self.max_candidate_score):
            ev_candidates = []
            # To save time, this is moved to the do_rollout section, as samples are only for display if do rollout
            if self.verbose:
                print(f"Sampling for aution: {auction} trying to find {self.sample_boards_for_auction}")
            hands_np, sorted_score, p_hcp, p_shp, good_quality = self.sample_hands_for_auction(auction, self.seat)
            samples = []
            for i in range(hands_np.shape[0]):
                samples.append('%s %s %s %s %.5f' % (
                    hand_to_str(hands_np[i,0,:]),
                    hand_to_str(hands_np[i,1,:]),
                    hand_to_str(hands_np[i,2,:]),
                    hand_to_str(hands_np[i,3,:]),
                    sorted_score[i]
                ))
            for candidate in candidates:
                if self.verbose:
                    print(f" {candidate.bid.ljust(4)} {candidate.insta_score:.3f} Samples: {len(hands_np)}")
                auctions_np = self.bidding_rollout(auction, candidate.bid, hands_np, self.sameforboth)
                if self.lead_included:
                    contracts, decl_tricks_softmax = self.expected_tricks_sd(hands_np, auctions_np)
                    for idx, (auction2, (contract, trick)) in enumerate(zip(auctions_np, zip(contracts, decl_tricks_softmax))):
                        auc = bidding.get_auction_as_string(auction2)
                        if contract.lower() != "pass":
                            weighted_sum = sum(i * trick[i] for i in range(len(trick)))
                            average_tricks = round(weighted_sum, 1)
                            samples[idx] += " \n " + auc + " ("+ str(average_tricks) + ") "
                        else:
                            samples[idx] += " \n " + auc
                else:
                    _, decl_tricks_softmax2 = self.expected_tricks_sd(hands_np, auctions_np)
                    contracts, decl_tricks_softmax3 = self.expected_tricks_sd_no_lead(hands_np, auctions_np)
                    if self.double_dummy_eval:
                        contracts, decl_tricks_softmax = self.expected_tricks_dd(hands_np, auctions_np)
                        for idx, (auction2, (contract, trick1, trick2, trick3)) in enumerate(zip(auctions_np, zip(contracts, decl_tricks_softmax, decl_tricks_softmax3, decl_tricks_softmax2))):
                            auc = bidding.get_auction_as_string(auction2)
                            if contract.lower() != "pass":
                                weighted_sum1 = sum(i * trick1[i] for i in range(len(trick1)))
                                average_tricks1 = round(weighted_sum1, 1)
                                weighted_sum2 = sum(i * trick2[i] for i in range(len(trick2)))
                                average_tricks2 = round(weighted_sum2, 1)
                                weighted_sum3 = sum(i * trick3[i] for i in range(len(trick3)))
                                average_tricks3 = round(weighted_sum3, 1)
                                samples[idx] += " \n " + auc + " (" + str(average_tricks1) + ", " + str(average_tricks2) + ", " + str(average_tricks3) + ") "
                            else:
                                samples[idx] += " \n " + auc
                    else:
                        decl_tricks_softmax = decl_tricks_softmax3
                        for idx, (auction2, (contract, trick1, trick2)) in enumerate(zip(auctions_np, zip(contracts, decl_tricks_softmax3, decl_tricks_softmax2))):
                            auc = bidding.get_auction_as_string(auction2)
                            if contract.lower() != "pass":
                                weighted_sum1 = sum(i * trick1[i] for i in range(len(trick1)))
                                average_tricks1 = round(weighted_sum1, 1)
                                weighted_sum2 = sum(i * trick2[i] for i in range(len(trick2)))
                                average_tricks2 = round(weighted_sum2, 1)
                                samples[idx] += " \n " + auc + " (" + str(average_tricks1) + ", " + str(average_tricks2) + ") "
                            else:
                                samples[idx] += " \n " + auc

    
                # We need to find a way to use how good the samples are
                ev = self.expected_score(len(auction) % 4, contracts, decl_tricks_softmax)
                expected_score = np.mean(ev)
                if self.verbose:
                    print(ev)
                adjust = 0

                # The result is sorted based on the simulation. 
                # Adding some bonus to the bid selected by the neural network
                # Should probably be configurable
                if candidate.insta_score < 0.002:
                    adjust += -200
                if candidate.insta_score < 0.0002:
                    adjust += -200
                
                if hands_np.shape[0] == self.sample.min_sample_hands_auction:
                    # We only have the minimum number of samples, so they is often of bad quality
                    # So we add more trust to the NN
                    adjust += 500*candidate.insta_score
                else:
                    adjust += 50*candidate.insta_score

                # If we are doubling as penalty in the pass out-situation
                # These adjustments should probably be configurable
                if passout and candidate.insta_score < self.min_candidate_score:
                    # If we are bidding in the passout situation, and is going down, assume we are doubled
                    if bidding.BID2ID[candidate.bid] > 4:
                        if expected_score < 0:
                            adjust += expected_score * 3
                        else:
                            adjust += -100

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

            # If the samples are bad we just trust the neural network
            if self.use_biddingquality  and not good_quality:
                candidates = sorted(ev_candidates, key=lambda c: c.insta_score, reverse=True)
            else:
                candidates = sorted(ev_candidates, key=lambda c: c.expected_score + c.adjust, reverse=True)
            
            who = "Simulation"
            # Print candidates with their relevant information
            if self.verbose:
                for idx, candidate in enumerate(candidates, start=1):
                    print(f"{idx}: {candidate.bid.ljust(4)} Insta_score: {candidate.insta_score:.3f} Expected Score: {str(int(candidate.expected_score)).ljust(5)} Adjustment:{str(int(candidate.adjust)).ljust(5)}")
        else:
            who = "NN"
            # Perhaps we should sample some hands to get information about how BEN sees the bidding until now
            # For now we just pick up the bidding info
            samples = []
            n_steps = binary.calculate_step_bidding_info(auction)
            p_hcp, p_shp = self.sample.get_bidding_info(self.binfo_model, n_steps, auction, self.seat, self.hand, self.vuln, self.ns, self.ew)
            p_hcp = p_hcp[0]
            p_shp = p_shp[0]

        if self.verbose:
            print(candidates[0].bid, " selected")
        return BidResp(bid=candidates[0].bid, candidates=candidates, samples=samples[:self.sample_hands_for_review], shape=p_shp, hcp=p_hcp, who=who, quality=good_quality)
    
    @staticmethod
    def do_rollout(auction, candidates, max_candidate_score):
        if candidates[0].insta_score > max_candidate_score:
           return False
        
        # Perhaps we should have another threshold for Double and Redouble as when that is suggested by NN, it is probably well defined

        # Just one candidate, so no need for rolling out the bidding
        if len(candidates) == 1:
            #print("Just one candidate, so no need for rolling out the bidding")
            return False
        
        # Do try to simulate if first to bid
        if bidding.get_contract(auction) == None:
            return False

        return True

    def get_bid_candidates(self, auction):
        bid_softmax = self.next_bid_np(auction)[0]
        if self.verbose:
            index_highest = np.argmax(bid_softmax)
            print(f"bid {bidding.ID2BID[index_highest]} value {bid_softmax[index_highest]:.4f} is recommended by NN")

        candidates = []

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
            # This is not good if we are doubled in a splinter
            # If we are doubled in pass out situation, never raise the suit
            
            # if auction[-3:] == ['X', 'PASS', 'PASS']:
            #    min_candidates = 2
        else:    
            if no_bids > self.eval_after_bid_count and auction[-2] != "PASS":
                min_candidates = 2
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
            if bid_softmax[bid_i] < self.min_candidate_score:
                if len(candidates) >= min_candidates:
                    break
            if bidding.can_bid(bidding.ID2BID[bid_i], auction):
                candidates.append(CandidateBid(bid=bidding.ID2BID[bid_i], insta_score=bid_softmax[bid_i]))
            else:
                # Seems to be an error in the training that needs to be solved
                # Only report it if above threshold
                if bid_softmax[bid_i] >= self.min_candidate_score and self.min_candidate_score != -1:
                    sys.stderr.write(f"{auction}\n")
                    sys.stderr.write(f"Bid not valid: {bidding.ID2BID[bid_i]} insta_score: {bid_softmax[bid_i]:.3f} {self.min_candidate_score}\n")
                if len(candidates) > 0:
                    break
                #assert(bid_i > 1)

            # set the score for the bid just processed to zero so it is out of the loop
            bid_softmax[bid_i] = 0

        # Consider adding an option for PASS, as a possible bid, even if it is not suggested by the nn
        #candidates.append(CandidateBid(bid=bidding.ID2BID[2], insta_score=0.5))

        if self.verbose:
            print("\n".join(str(bid) for bid in candidates))

        return candidates, passout

    def next_bid_np(self, auction):
        x = self.get_binary(auction)
        bid_np = self.bidder_model.model_seq(x)
        bid_np = bid_np[-1:]
        return bid_np
    
    def sample_hands_for_auction(self, auction_so_far, turn_to_bid):
        # The longer the aution the more hands we might need to sample
        sample_boards_for_auction = self.sample.sample_boards_for_auction
        if len(auction_so_far) > 12:
            sample_boards_for_auction *= 2
        if len(auction_so_far) > 24:
            sample_boards_for_auction *= 4
        # Reset randomizer
        self.rng = np.random.default_rng(self.hash_integer)
        accepted_samples, sorted_scores, p_hcp, p_shp, good_quality = self.sample.sample_cards_auction(
            auction_so_far, turn_to_bid, self.hand_str, self.vuln, self.bidder_model, self.binfo_model, self.ns, self.ew, self.sameforboth,sample_boards_for_auction, self.rng)
        
        #assert good_quality, "We did not find samples for the bidding of decent quality"

        if self.verbose:
            print(f"Found {accepted_samples.shape[0]} samples for bidding")

        # We have more samples, than we want to calculate on
        # They are sorted according to the bidding trust, but above our threshold, so we pick random
        if accepted_samples.shape[0] > self.sample.sample_hands_auction:
            random_indices = self.rng.permutation(accepted_samples.shape[0])
            accepted_samples = accepted_samples[random_indices[:self.sample.sample_hands_auction], :, :]
            sorted_scores = sorted_scores[random_indices[:self.sample.sample_hands_auction]]

        n_samples = accepted_samples.shape[0]
        
        hands_np = np.zeros((n_samples, 4, 32), dtype=np.int32)
        hands_np[:,turn_to_bid,:] = self.hand
        for i in range(1, 4):
            hands_np[:, (turn_to_bid + i) % 4, :] = accepted_samples[:,i-1,:]

        return hands_np, sorted_scores, p_hcp, p_shp, good_quality 

    def bidding_rollout(self, auction_so_far, candidate_bid, hands_np, sameforboth):
        auction = [*auction_so_far, candidate_bid]
        
        n_samples = hands_np.shape[0]
        if self.verbose:
            print("bidding_rollout - n_samples: ", n_samples)
        assert n_samples > 0
        
        n_steps_vals = [0, 0, 0, 0]
        for i in range(1, 5):
            n_steps_vals[(len(auction_so_far) % 4 + i) % 4] = BotBid.get_bid_number_for_player_to_bid(auction_so_far + ['?'] * i)  
        
        # initialize auction vector
        auction_np = np.ones((n_samples, 64), dtype=np.int32) * bidding.BID2ID['PAD_END']
        for i, bid in enumerate(auction):
            auction_np[:,i] = bidding.BID2ID[bid]

        bid_i = len(auction) - 1
        turn_i = len(auction) % 4

        # Now we bid each sample to end of auction
        while not np.all(auction_np[:,bid_i] == bidding.BID2ID['PAD_END']):
            if sameforboth and self.dealer > 1:
                hand_ix = (turn_i + 2) % 4
            else:
                hand_ix = turn_i
            X = binary.get_auction_binary_sampling(n_steps_vals[turn_i], auction_np, turn_i, hands_np[:,hand_ix,:], self.vuln, self.ns, self.ew)
            y_bid_np = self.bidder_model.model_seq(X)
            x_bid_np = y_bid_np.reshape((n_samples, n_steps_vals[turn_i], -1))
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
                            sys.stderr.write(f"Pass not valid as auction is over: {bidding.ID2BID[bid]} insta_score: {bid_np[i][bid]:.3f}\n")
                            bid_np[i][1] = 1
                        # Pass is always allowed
                        if (bid > 2 and not bidding.can_bid(bidding.ID2BID[bid], auction)):
                            invalid_bids = True
                            #sys.stderr.write(str(auction))
                            #sys.stderr.write(f"Bid not valid: {bidding.ID2BID[bid]} insta_score: {bid_np[i][bid]}\n")
                            bid_np[i][bid] = 0

                        assert bid_i <= 60, f'Auction to long {bid_i} {auction} {auction_np[i]}'
                    else:
                        bid_np[i][1] = 1
                if invalid_bids: 
                    continue

            bid_i += 1
            auction_np[:,bid_i] = np.argmax(bid_np, axis=1)
            n_steps_vals[turn_i] += 1
            turn_i = (turn_i + 1) % 4
        assert len(auction_np) > 0

        if self.verbose:
            print("bidding_rollout - finished ",auction_np.shape)
        
        return auction_np
    
    def expected_tricks_sd(self, hands_np, auctions_np):
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
            # All pass doens't really fit, and is always 0 - we ignore it for now
            if contract is None:
                contracts.append("pass")
                strains[i] = -1
                declarers[i] = -1
            else:
                contracts.append(contract)
                strains[i] = 'NSHDC'.index(contract[1])
                declarers[i] = 'NESW'.index(contract[-1])
            
                hand_on_lead = hands_np[i:i+1, (declarers[i] + 1) % 4, :]
            
                X_ftrs[i,:], B_ftrs[i,:] = binary.get_auction_binary_for_lead(sample_auction, hand_on_lead, self.binfo_model, self.vuln, self.ns, self.ew)
        
        lead_softmax = self.lead_suit_model.model(X_ftrs, B_ftrs)
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
    
    def expected_tricks_sd_no_lead(self, hands_np, auctions_np):
        n_samples = hands_np.shape[0]

        s_all = np.arange(n_samples, dtype=np.int32)
        declarers = np.zeros(n_samples, dtype=np.int32)
        strains = np.zeros(n_samples, dtype=np.int32)
      
        contracts = []
        
        for i in range(n_samples):
            sample_auction = [bidding.ID2BID[bid_i] for bid_i in list(auctions_np[i, :]) if bid_i != 1]
            contract = bidding.get_contract(sample_auction)
            # All pass doens't really fit, and is always 0 - we ignore it for now
            if contract is None:
                contracts.append("pass")
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
        X_sd[:,(5 + 0*32):(5 + 1*32)] = hands_np[s_all, (declarers + 1) % 4]
        # dummy
        X_sd[:,(5 + 1*32):(5 + 2*32)] = hands_np[s_all, (declarers + 2) % 4]
        # righty
        X_sd[:,(5 + 2*32):(5 + 3*32)] = hands_np[s_all, (declarers + 3) % 4]
        # declarer
        X_sd[:,(5 + 3*32):] = hands_np[s_all, declarers]
        
        decl_tricks_softmax = self.sd_model_no_lead.model(X_sd)
        return contracts, decl_tricks_softmax

    def expected_tricks_dd(self, hands_np, auctions_np):
        from ddsolver import ddsolver
        self.dd = ddsolver.DDSolver()
        n_samples = hands_np.shape[0]

        declarers = np.zeros(n_samples, dtype=np.int32)
        strains = np.zeros(n_samples, dtype=np.int32)
        decl_tricks_softmax = np.zeros((n_samples, 14), dtype=np.int32)
        contracts = []
        t_start = time.time()
        
        for i in range(n_samples):
            sample_auction = [bidding.ID2BID[bid_i] for bid_i in list(auctions_np[i, :]) if bid_i != 1]
            contract = bidding.get_contract(sample_auction)
            # All pass doesn't really fit, and is always 0 - we ignore it for now
            if contract is None:
                contracts.append("pass")
                strains[i] = -1
                declarers[i] = -1
            else:
                contracts.append(contract)
                strains[i] = 'NSHDC'.index(contract[1])
                declarers[i] = 'NESW'.index(contract[-1])

            # Create PBN for hand
            hand_str = ""
            # We need to rotate to find the right to lead
            # Perhaps we should use our actual hand (so the pips are correct)
            hands_pbn = ['N:' + ' '.join(deck52.hand32to52str(hand) for hand in hands_np[i])]
            hands_pbn[0] = deck52.convert_cards(hands_pbn[0],0, hand_str)
            # We need to find the leader
            dd_solved = self.dd.solve(strains[i], (declarers[i] + 1) % 4, [], hands_pbn, 1)
            # Only use 1st element from the result
            first_key = next(iter(dd_solved))
            first_item = dd_solved[first_key]
            decl_tricks_softmax[i,13 - first_item[0]] = 1

        if self.verbose:
            print(f'dds took {time.time() - t_start:.3f}')
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

class BotLead:

    def __init__(self, vuln, hand_str, models, sample, seat, verbose):
        self.vuln = vuln
        self.hand_str = hand_str
        self.hand = binary.parse_hand_f(32)(hand_str)
        self.hand52 = binary.parse_hand_f(52)(hand_str)

        self.lead_suit_model = models.lead_suit_model
        self.lead_nt_model = models.lead_nt_model
        self.ns = models.ns
        self.ew = models.ew
        self.seat = seat
        self.bidder_model = models.bidder_model
        self.binfo_model = models.binfo_model
        self.sd_model = models.sd_model
        self.sd_model_no_lead = models.sd_model_no_lead
        self.double_dummy = models.double_dummy
        self.sample = sample
        self.verbose = verbose
        self.lead_threshold = models.lead_threshold
        self.lead_accept_nn = models.lead_accept_nn
        self.lead_included = models.lead_included
        self.min_opening_leads = models.min_opening_leads
        self.sample_hands_for_review = models.sample_hands_for_review
        self.use_biddingquality_in_eval = models.use_biddingquality_in_eval
        self.hash_integer  = calculate_seed(hand_str)         
        if self.verbose:
            print(f"Setting seed (Sampling bidding info) from {hand_str}: {self.hash_integer}")
        self.rng = np.random.default_rng(self.hash_integer)

    def find_opening_lead(self, auction):
        lead_card_indexes, lead_softmax = self.get_opening_lead_candidates(auction)
        accepted_samples, sorted_bidding_score, tricks, p_hcp, p_shp, good_quality = self.simulate_outcomes_opening_lead(auction, lead_card_indexes)

        candidate_cards = []
        for i, card_i in enumerate(lead_card_indexes):
            assert(tricks[:,i,0].all() >= 0)
            candidate_cards.append(CandidateCard(
                card=Card.from_code(card_i, xcards=True),
                insta_score=lead_softmax[0,card_i],
                expected_tricks_sd=np.mean(tricks[:,i,0]),
                p_make_contract=np.mean(tricks[:,i,1]),
                expected_score_sd = None
            ))
        
        candidate_cards = sorted(candidate_cards, key=lambda c: c.insta_score, reverse=True)

        if candidate_cards[0].insta_score > self.lead_accept_nn:
            opening_lead = candidate_cards[0].card.code() 
        else:
            # If our sampling of the hands from the aution is bad.
            # We should probably try to find better samples, but for now, we just trust the neural network

            if (self.use_biddingquality_in_eval and not good_quality):
                opening_lead = candidate_cards[0].card.code() 
            else:
                # Now we will select the card to play
                # We have 3 factors, and they could all be right, so we remove most of the decimals
                # expected_tricks_sd is for declarer
                candidate_cards = sorted(candidate_cards, key=lambda c: (round(5*c.p_make_contract, 1), -round(c.expected_tricks_sd, 1), round(c.insta_score, 2)), reverse=True)
                # Print each CandidateCard in the list
                opening_lead = candidate_cards[0].card.code()

        if self.verbose:
            print(good_quality)
            for card in candidate_cards:
                print(card)
        if opening_lead % 8 == 7:
            # it's a pip ~> choose a random one
            pips_mask = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1])
            lefty_led_pips = self.hand52.reshape((4, 13))[opening_lead // 8] * pips_mask
            # Implement human carding here
            opening_lead52 = (opening_lead // 8) * 13 + self.rng.choice(np.nonzero(lefty_led_pips)[0])
        else:
            opening_lead52 = deck52.card32to52(opening_lead)

        samples = []
        if self.verbose:
            print(f"Accepted samples for opening lead: {accepted_samples.shape[0]}")
        for i in range(min(self.sample_hands_for_review, accepted_samples.shape[0])):
            samples.append('%s %s %s %s %.5f' % (
                hand_to_str(self.hand),
                hand_to_str(accepted_samples[i,0,:]),
                hand_to_str(accepted_samples[i,1,:]),
                hand_to_str(accepted_samples[i,2,:]),
                sorted_bidding_score[i]
            ))

        return CardResp(
            card=Card.from_code(opening_lead52),
            candidates=candidate_cards,
            samples=samples, 
            shape=p_shp,
            hcp=p_hcp,
            quality=good_quality
        )

    def get_opening_lead_candidates(self, auction):
        x_ftrs, b_ftrs = binary.get_auction_binary_for_lead(auction, self.hand, self.binfo_model, self.vuln, self.ns, self.ew)
        contract = bidding.get_contract(auction)
        if contract[1] == "N":
            lead_softmax = self.lead_nt_model.model(x_ftrs, b_ftrs)
        else:
            lead_softmax = self.lead_suit_model.model(x_ftrs, b_ftrs)
        lead_softmax = follow_suit(lead_softmax, self.hand, np.array([[0, 0, 0, 0]]))

        candidates = []
        # Make a copy of the lead_softmax array
        lead_softmax_copy = np.copy(lead_softmax)

        if self.verbose:
            print("Finding leads from neural network")
        while True:
            c = np.argmax(lead_softmax_copy[0])
            score = lead_softmax_copy[0][c]
            # Always take minimum the number from configuration
            if score < self.lead_threshold and len(candidates) >= self.min_opening_leads:
                break
            if self.verbose:
                print(f"{Card.from_code(c, xcards=True)} {score:.3f}")
            candidates.append(c)
            lead_softmax_copy[0][c] = 0

        return candidates, lead_softmax

    def simulate_outcomes_opening_lead(self, auction, lead_card_indexes):
        contract = bidding.get_contract(auction)

        decl_i = bidding.get_decl_i(contract)
        lead_index = (decl_i + 1) % 4

        if self.verbose:
            print(f'Now generating {self.sample.sample_boards_for_auction_opening_lead} deals to find opening lead')
        # Reset randomizer
        self.rng = np.random.default_rng(self.hash_integer)
        accepted_samples, sorted_scores, p_hcp, p_shp, good_quality = self.sample.sample_cards_auction(auction, lead_index, self.hand_str, self.vuln, self.bidder_model, self.binfo_model, self.ns, self.ew, False, self.sample.sample_boards_for_auction_opening_lead, self.rng)

        if self.verbose:
            print("Generated samples:", accepted_samples.shape[0], " OK Quality", good_quality)
            print(f'Now simulate on {self.sample.sample_hands_opening_lead} deals to find opening lead')
                
        # We have more samples, then we want to calculate on
        # They are sorted according to the bidding trust, but above our threshold, so we pick random
        if accepted_samples.shape[0] > self.sample.sample_hands_opening_lead:
            random_indices = self.rng.permutation(accepted_samples.shape[0])
            accepted_samples = accepted_samples[random_indices[:self.sample.sample_hands_opening_lead], :, :]
            sorted_scores = sorted_scores[random_indices[:self.sample.sample_hands_opening_lead]]

        # For finding the opening lead we should use the opening lead as input
        if self.double_dummy:
            tricks = self.double_dummy_estimates(lead_card_indexes, contract, accepted_samples)
        else:
            tricks = self.single_dummy_estimates(lead_card_indexes, contract, accepted_samples)

        return accepted_samples, sorted_scores, tricks, p_hcp, p_shp, good_quality

    def double_dummy_estimates(self, lead_card_indexes, contract, accepted_samples):
        from ddsolver import ddsolver
        #print("double_dummy_estimates",lead_card_indexes)
        self.dd = ddsolver.DDSolver()
        n_accepted = accepted_samples.shape[0]
        tricks = np.zeros((n_accepted, len(lead_card_indexes), 2))
        strain_i = bidding.get_strain_i(contract)

        # if defending the target is another
        level = int(contract[0])
        tricks_needed = 13 - (level + 6) + 1

        for j, lead_card_i in enumerate(lead_card_indexes):
            # Subtract the opening lead from the hand
            lead_hand = self.hand52[0]
            # So now we need to figure out what the lead was if a pip
            if lead_card_i % 8 == 7:
                # it's a pip ~> choose a random one
                pips_mask = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1])
                lefty_led_pips = self.hand52.reshape((4, 13))[lead_card_i // 8] * pips_mask
                # Perhaps use human carding, but it is only for estimation
                opening_lead52 = (lead_card_i // 8) * 13 + self.rng.choice(np.nonzero(lefty_led_pips)[0])
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
                print("Opening lead being examined: ", Card.from_code(opening_lead52),n_accepted)
            t_start = time.time()
            for i in range(n_accepted):
                hands_pbn = ['N:' + hand_str + ' ' + ' '.join(deck52.hand32to52str(hand) for hand in accepted_samples[i])]
                hands_pbn[0] = deck52.convert_cards(hands_pbn[0],opening_lead52, hand_str)
                # lead is relative to the order in the PBN-file, so West is 0 here
                onlead = 0
                dd_solved = self.dd.solve(strain_i, onlead, [opening_lead52], hands_pbn, 3)
                # Only use 1st element from the result
                first_key = next(iter(dd_solved))
                first_item = dd_solved[first_key]
                tricks[i, j, 0] = first_item[0] 
                tricks[i, j, 1] = 1 if (13 - first_item[0]) >= tricks_needed else 0
            if self.verbose:
                print(f'dds took {time.time() - t_start:0.4}')
        return tricks

    def single_dummy_estimates(self, lead_card_indexes, contract, accepted_samples):
        n_accepted = accepted_samples.shape[0]
        X_sd = np.zeros((n_accepted, 32 + 5 + 4*32))

        strain_i = bidding.get_strain_i(contract)

        X_sd[:,32 + strain_i] = 1
        # lefty (That is us)
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
        return tricks

class CardPlayer:

    def __init__(self, models, player_i, hand_str, public_hand_str, contract, is_decl_vuln, sampler, pimc = None, verbose = False):
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
        self.hand = parse_hand_f(32)(hand_str).reshape(32)
        self.hand52 = parse_hand_f(52)(hand_str).reshape(52)
        self.public52 = parse_hand_f(52)(public_hand_str).reshape(52)
        self.dummy = self.public52.copy()
        self.contract = contract
        self.is_decl_vuln = is_decl_vuln
        self.n_tricks_taken = 0
        self.verbose = verbose
        self.level = int(contract[0])
        self.sample_hands_for_review = models.sample_hands_for_review
        self.init_x_play(parse_hand_f(32)(public_hand_str), self.level, self.strain_i)
        self.bid_accept_play_threshold = sampler.bid_accept_play_threshold
        self.score_by_tricks_taken = [scoring.score(self.contract, self.is_decl_vuln, n_tricks) for n_tricks in range(14)]
        self.use_biddingquality_in_eval = models.use_biddingquality_in_eval
        from ddsolver import ddsolver
        self.dd = ddsolver.DDSolver()
        if (player_i == 1):
            self.hash_integer  = calculate_seed(public_hand_str)         
            if self.verbose:
                print(f"Setting seed (Sampling bidding info) from {public_hand_str}: {self.hash_integer}")
        else:
            self.hash_integer  = calculate_seed(hand_str)         
            if self.verbose:
                print(f"Setting seed (Sampling bidding info) from {hand_str}: {self.hash_integer}")
        self.rng = np.random.default_rng(self.hash_integer)
        self.pimc = pimc

    def init_x_play(self, public_hand, level, strain_i):
        self.x_play = np.zeros((1, 13, 298))
        binary.BinaryInput(self.x_play[:,0,:]).set_player_hand(self.hand)
        binary.BinaryInput(self.x_play[:,0,:]).set_public_hand(public_hand)
        self.x_play[:,0,292] = level
        self.x_play[:,0,293+strain_i] = 1

    def set_real_card_played(self, card, playedBy, openinglead=False):
        if (self.player_i == 3) and self.models.pimc_use:
            self.pimc.set_card_played(card, playedBy, openinglead)

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

    async def play_card(self, trick_i, leader_i, current_trick52, players_states, bidding_scores, quality, probability_of_occurence, shown_out_suits):
        current_trick = [deck52.card52to32(c) for c in current_trick52]
        samples = []
        for i in range(min(self.sample_hands_for_review, players_states[0].shape[0])):
            samples.append('%s %s %s %s %.5f' % (
                hand_to_str(players_states[0][i,0,:32].astype(int)),
                hand_to_str(players_states[1][i,0,:32].astype(int)),
                hand_to_str(players_states[2][i,0,:32].astype(int)),
                hand_to_str(players_states[3][i,0,:32].astype(int)),
                bidding_scores[i]
            ))

        # If we are declarer and PIMC enabled - use PIMC
        BGADLL = (self.player_i == 1 or self.player_i == 3) and self.models.pimc_use and trick_i  >= (self.models.pimc_start_trick - 1)
        if BGADLL:
            # At trick one we generate constraints based on the samples
            if trick_i == 0 and self.models.pimc_hcp_constraints:
                h1 = []
                h3 = []
                s1 = []
                s3 = []
                for i in range(players_states[0].shape[0]):
                    # Not needed to count for declarer and dummy
                    h1.append(binary.get_hcp(hand = np.array(players_states[0][i, 0, :32].astype(int)).reshape(1,32)))
                    h3.append(binary.get_hcp(hand = np.array(players_states[2][i, 0, :32].astype(int)).reshape(1,32)))
                    s1.append(binary.get_shape(hand = np.array(players_states[0][i, 0, :32].astype(int)).reshape(1,32))[0])
                    s3.append(binary.get_shape(hand = np.array(players_states[2][i, 0, :32].astype(int)).reshape(1,32))[0])
                min_h1 = int(min(h1))
                max_h1 = int(max(h1))
                min_h3 = int(min(h3))
                max_h3 = int(max(h3))
                self.pimc.set_hcp_constraints(min_h1, max_h1, min_h3, max_h3, quality)
                min_values1 = [min(col) for col in zip(*s1)]
                max_values1 = [max(col) for col in zip(*s1)]
                min_values3 = [min(col) for col in zip(*s3)]
                max_values3 = [max(col) for col in zip(*s3)]
                print(min_values1, max_values1, min_values3, max_values3)

                self.pimc.set_shape_constraints(min_values1, max_values1, min_values3, max_values3, quality)

            # Based on player states we should be able to find min max for suits and hcps, and add that before calling PIMC
            card52_dd = await self.pimc.nextplay(self.player_i, shown_out_suits)
            card_resp = self.pick_card_after_pimc_eval(trick_i, leader_i, current_trick, players_states, card52_dd, bidding_scores, quality, samples)            
        else:
            card52_dd = self.get_cards_dd_evaluation(trick_i, leader_i, current_trick52, players_states, probability_of_occurence)
            card_resp = self.pick_card_after_dd_eval(trick_i, leader_i, current_trick, players_states, card52_dd, bidding_scores, quality, samples)

        return card_resp

    def get_cards_dd_evaluation(self, trick_i, leader_i, current_trick52, players_states, probabilities_list):
        from ddsolver import ddsolver
        
        n_samples = players_states[0].shape[0]
        assert n_samples > 0, "No samples for DDSolver"

        # All previously played pips are also unavailable, so we use the original dummy and not what we can see
        unavailable_cards = set(list(np.nonzero(self.hand52)[0]) + list(np.nonzero(self.dummy)[0]) + current_trick52)

        pips = [
            [c for c in range(7, 13) if i*13+c not in unavailable_cards] for i in range(4)
        ]

        symbols = 'AKQJT98765432'

        current_trick_players = [(leader_i + i) % 4 for i in range(len(current_trick52))]

        hands_pbn = []
        for i in range(n_samples):
            hands = [None, None, None, None]
            for j in range(4):
                self.rng.shuffle(pips[j])
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
            print("Samples: ",n_samples, " Solving: ",len(hands_pbn))
        dd_solved = self.dd.solve(self.strain_i, leader_i, current_trick52, hands_pbn, 3)

        if self.models.use_probability:
            card_tricks = ddsolver.expected_tricks_dds_probabiliy(dd_solved, probabilities_list)
        else:
            card_tricks = ddsolver.expected_tricks_dds(dd_solved)

        # if defending the target is another
        level = int(self.contract[0])
        if self.player_i % 2 == 1:
            tricks_needed = level + 6 - self.n_tricks_taken
        else:
            tricks_needed = 13 - (level + 6) - self.n_tricks_taken + 1

        making = ddsolver.p_made_target(tricks_needed)(dd_solved)

        if self.models.use_probability:
            if self.models.matchpoint:
                card_ev = self.get_card_ev_mp(dd_solved, probabilities_list)
            else:
                card_ev = self.get_card_ev_probability(dd_solved, probabilities_list)
        else:
            if self.models.matchpoint:
                card_ev = self.get_card_ev_mp(dd_solved)
            else:
                card_ev = self.get_card_ev(dd_solved)
        if self.verbose:
            print("card_ev:", card_ev)

        card_result = {}
        for key in dd_solved.keys():
            card_result[key] = (card_tricks[key], card_ev[key], making[key])
            if self.verbose:
                print(deck52.decode_card(key), card_tricks[key], card_ev[key], making[key])

        if self.verbose:
            print(f'dds took {time.time() - t_start:0.4}')

        return card_result
    
    def get_card_ev(self, dd_solved):
        card_ev = {}
        sign = 1 if self.player_i % 2 == 1 else -1
        for card, future_tricks in dd_solved.items():
            ev_sum = 0
            for ft in future_tricks:
                if ft < 0:
                    continue
                tot_tricks = self.n_tricks_taken + ft
                tot_decl_tricks = tot_tricks if self.player_i % 2 == 1 else 13 - tot_tricks
                ev_sum += sign * self.score_by_tricks_taken[tot_decl_tricks]
            card_ev[card] = ev_sum / len(future_tricks)
                
        return card_ev

    def get_card_ev_probability(self, dd_solved, probabilities_list):
        card_ev = {}
        sign = 1 if self.player_i % 2 == 1 else -1
        for card, future_tricks in dd_solved.items():
            ev_sum = 0
            for ft, proba in zip(future_tricks, probabilities_list):
                if ft < 0:
                    continue
                tot_tricks = self.n_tricks_taken + ft
                tot_decl_tricks = (
                    tot_tricks if self.player_i % 2 == 1 else 13 - tot_tricks
                )
                ev_sum += sign * self.score_by_tricks_taken[tot_decl_tricks] * proba
            card_ev[card] = ev_sum

        return card_ev
    
    def get_card_ev_mp(self, dd_solved, probabilities_list):
        card_ev = {}
        for card, future_tricks in dd_solved.items():
            ev_sum = 0
            for ft, proba in zip(future_tricks, probabilities_list):
                if ft < 0:
                    continue
                ev_sum += ft * proba
            card_ev[card] = ev_sum

        return card_ev
    
    def get_card_ev_mp(self, dd_solved):
        card_ev = {}
        sign = 1 if self.player_i % 2 == 1 else -1
        for card, future_tricks in dd_solved.items():
            ev_sum = 0
            for ft in future_tricks:
                if ft < 0:
                    continue
                tot_tricks = self.n_tricks_taken + ft
                tot_decl_tricks = tot_tricks if self.player_i % 2 == 1 else 13 - tot_tricks
                ev_sum += sign * tot_decl_tricks
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

    def pick_card_after_pimc_eval(self, trick_i, leader_i, current_trick, players_states, card_dd, bidding_scores, quality, samples):
        t_start = time.time()
        card_softmax = self.next_card_softmax(trick_i)
        if self.verbose:
            print(f'Next card response time: {time.time() - t_start:0.4f}')

        all_cards = np.arange(32)
        ## This could be a parameter, but only used for display purposes
        s_opt = card_softmax > 0.001

        card_options, card_scores = all_cards[s_opt], card_softmax[s_opt]

        card_nn = {c:s for c, s in zip(card_options, card_scores)}
        if self.verbose:
            print(card_nn)

        candidate_cards = []
        
        for card, (e_tricks, e_score, e_make, msg) in card_dd.items():
            card32 = deck52.card52to32(deck52.encode_card(str(card)))

            candidate_cards.append(CandidateCard(
                card=card,
                insta_score=card_nn.get(card32, 0),
                expected_tricks_dd=e_tricks,
                p_make_contract=e_make,
                expected_score_dd=e_score,
                msg=msg
            ))

        candidate_cards = sorted(candidate_cards, key=lambda c: (c.p_make_contract, c.expected_tricks_dd, c.expected_score_dd, c.insta_score), reverse=True)

        best_card_resp = CardResp(
            card=candidate_cards[0].card,
            candidates=candidate_cards,
            samples=samples,
            shape=-1,
            hcp=-1, 
            quality=quality

        )
        return best_card_resp


    def pick_card_after_dd_eval(self, trick_i, leader_i, current_trick, players_states, card_dd, bidding_scores, quality, samples):
        t_start = time.time()
        card_softmax = self.next_card_softmax(trick_i)
        if self.verbose:
            print(f'Next card response time: {time.time() - t_start:0.4f}')

        all_cards = np.arange(32)
        ## This could be a parameter, but only used for display purposes
        s_opt = card_softmax > 0.001

        card_options, card_scores = all_cards[s_opt], card_softmax[s_opt]

        card_nn = {c:s for c, s in zip(card_options, card_scores)}
        if self.verbose:
            print(card_nn)

        candidate_cards = []
        
        for card52, (e_tricks, e_score, e_make) in card_dd.items():
            card32 = deck52.card52to32(card52)

            candidate_cards.append(CandidateCard(
                card=Card.from_code(card52),
                insta_score=card_nn.get(card32, 0),
                expected_tricks_dd=e_tricks,
                p_make_contract=e_make,
                expected_score_dd=e_score
            ))

        valid_bidding_samples = np.sum(bidding_scores > self.bid_accept_play_threshold)
        # Now we will select the card to play
        # We have 3 factors, and they could all be right, so we remove most of the decimals
        # We should probably also consider bidding_scores in this 
        # If we have bad quality of samples we should probably just use the neural network
        if valid_bidding_samples >= 0:
            candidate_cards = sorted(candidate_cards, key=lambda c: (round(5*c.p_make_contract, 1), round(c.expected_tricks_dd, 1), round(c.insta_score, 2)), reverse=True)
        else:
            if self.use_biddingquality_in_eval:
                candidate_cards = sorted(candidate_cards, key=lambda c: (round(c.insta_score, 2), round(5*c.p_make_contract, 1), round(c.expected_tricks_dd, 1)), reverse=True)
                candidate_cards2 = sorted(candidate_cards, key=lambda c: (round(c.expected_score_dd, 1), round(c.insta_score, 2), round(c.expected_tricks_dd, 1)), reverse=True)
                if candidate_cards[0].expected_score_dd < 0 and candidate_cards2[0].expected_score_dd:
                    candidate_cards = candidate_cards2
            else:
                candidate_cards = sorted(candidate_cards, key=lambda c: (round(5*c.p_make_contract, 1), round(c.insta_score, 2), round(c.expected_tricks_dd, 1)), reverse=True)

        best_card_resp = CardResp(
            card=candidate_cards[0].card,
            candidates=candidate_cards,
            samples=samples,
            shape=-1,
            hcp=-1, 
            quality=quality

        )

        # for candidate_card in candidate_cards :
        #     print(candidate_card.to_dict())

        # Max expected score difference ?
        max_expected_score = max(
            [
                float(c.expected_score_dd)
                for c in candidate_cards
                if c.expected_score_dd is not None
            ]
        )
        card_with_max_expected_score = {c: c.expected_score_dd for c in candidate_cards if c.expected_score_dd == max_expected_score}
        if len(card_with_max_expected_score) == 1:
            return best_card_resp
        candidate_cards = [c for c in candidate_cards if c in card_with_max_expected_score.keys()]

        # Don't pick a 8 or 9 when NN difference if you have a small card in the same suit
        for c in candidate_cards:
            if c.card.rank in [5, 6]:  # 8 or 9
                for compared_candidate in candidate_cards:
                    if (
                        compared_candidate.card.rank >= 7
                        and compared_candidate.card.suit == c.card.suit
                    ):
                        c.insta_score = compared_candidate.insta_score
                        break  # Below 7, they all have the same insta_score

        # NN difference ?
        max_insta_score = max(
            [float(c.insta_score) for c in candidate_cards if c.insta_score is not None]
        )
        card_with_max_insta_score = {
            c: c.insta_score
            for c in candidate_cards
            if c.insta_score == max_insta_score
            and c.expected_score_dd == max_expected_score
        }
        if len(card_with_max_insta_score) == 1:
            return best_card_resp

        # # Play some human carding
        # hand = PlayerHand.from_pbn(deck52.hand_to_str(self.hand52))
        # valid_cards = [
        #     Card_.from_str(c.card.symbol()) for c in card_with_max_insta_score.keys()
        # ]
        # return str(
        #     play_real_card(
        #         hand,
        #         valid_cards,
        #         self.trump,
        #         self.play_record,
        #         self.player_direction,
        #         self.declarer,
        #     )
        # )

        return best_card_resp
