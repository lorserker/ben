import pprint
import time
import numpy as np
import tensorflow as tf

import binary
import deck52
import calculate
import scoring

from objects import Card, CardResp, CandidateCard
from bidding import bidding

import carding
from util import hand_to_str, expected_tricks_sd, p_defeat_contract, follow_suit, calculate_seed
from colorama import Fore, init

init()
class BotLead:

    def __init__(self, vuln, hand_str, models, sampler, seat, dealer, dds, verbose):
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
        self.dds = dds

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
                if self.verbose:
                    self.dds.print_dd_results(dd_solved, xcards=True)
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
        
        # Due to error in Keras (Using different versin, that the one used for training) we might get cards not in hand 
        # So we should probably check, and exit if we have a version conflict

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
                
            dd_solved = self.dds.solve(strain_i, onlead, [opening_lead52], hands_pbn, 1)

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

