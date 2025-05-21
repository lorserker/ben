import time
import numpy as np
import tensorflow as tf

import binary
import deck52
import calculate
import scoring

from claim import Claimer
from objects import Card, CardResp, CandidateCard
from bidding import bidding

import carding
from alphamju.alphamju import alphamju
from util import hand_to_str, follow_suit, calculate_seed, symbols
from colorama import Fore, init
init()
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
        self.missing_cards_initial = self.missing_cards.copy()
        self.verbose = verbose
        self.level = int(contract[0])
        self.init_x_play(binary.parse_hand_f(32)(public_hand_str), self.level, self.strain_i)
        self.dds = ddsolver
        self.sampler = sampler
        self.claimer = Claimer(self.verbose, self.dds)
        if self.models.use_suitc:
            from suitc.SuitC import SuitCLib
            self.suitc = SuitCLib(self.verbose)
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
    
    def alphamju_evaluation(self, trick_i, play_status, leader_i, current_trick52, quality, worlds, samples, card_scores_nn):
                # Create a lookup dictionary to find the scores
        card_nn = {c: round(s, 3) for c, s in zip(np.arange(self.models.n_cards_play), card_scores_nn)}

        card_resp_alphamju = []

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
            card_resp_alphamju = alphamju(tricks_left-2, min(tricks_needed, tricks_left), suit, current_trick52, worlds, self.get_random_generator(), self.verbose)

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
                      
                    insta_score = card_nn.get(deck52.card52to32(Card.from_symbol(card).code()), 0)
                    candidate = CandidateCard(Card.from_symbol(card), percent, -1, -1, -1, -1, -1, -1, -1, f"NN={insta_score:.2f}")
                    candidate_cards.append(candidate)

                if candidate_cards != []:
                    # Sort by insta_score (descending), and in case of tie, by index (ascending)
                    # Attach original index for stable sorting
                    candidate_cards = [c for _, c in sorted(
                        enumerate(candidate_cards),
                        key=lambda x: (x[1].insta_score, x[1].msg, -x[0]),
                        reverse=True
                    )]
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
                    return card_resp, card_resp_alphamju

                # If card_resp_alphamju is empty, break out of the loop (or handle as needed)
                break

            # If card_resp_alphamju is empty, break out of the loop (or handle as needed)
            break

        return None, card_resp_alphamju

    def play_card(self, trick_i, leader_i, current_trick52, tricks52, players_states, worlds, bidding_scores, quality, probability_of_occurence, shown_out_suits, play_status, lead_scores, play_scores, logical_play_scores, discard_scores, features):
        #print(f'{Fore.YELLOW}Play card for player {self.player_i}{Fore.RESET}')
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

        card_scores_nn = self.next_card_softmax(trick_i)
        if self.verbose:
            print(f'Next card response time: {time.time() - t_start:0.4f}')

        card_resp_alphamju = []
        if self.models.alphamju_declaring and (self.player_i == 1 or self.player_i == 3) and trick_i > self.models.alphamju_trick and play_status != "Discard":
            card_resp, card_resp_alphamju = self.alphamju_evaluation(trick_i, play_status,leader_i,current_trick52,quality,worlds, samples, card_scores_nn)
            if card_resp:
                return card_resp

        # When play_status is discard, it might be a good idea to use PIMC even if it is not enabled
        preempted = features.get("preempted", False)

        if play_status == "discard" and not self.models.pimc_use_discard:
            dd_resp_cards, claims = self.get_cards_dd_evaluation(trick_i, leader_i, tricks52, current_trick52, players_states, probability_of_occurence, quality)
            self.update_with_alphamju(card_resp_alphamju, merged_card_resp)
            card_resp = self.pick_card_after_dd_eval(trick_i, leader_i, current_trick52, tricks52, players_states, dd_resp_cards, bidding_scores, quality, samples, play_status, self.missing_cards, claims, shown_out_suits, card_scores_nn)
        else:                    
            if self.pimc_declaring and (self.player_i == 1 or self.player_i == 3):
                pimc_resp_cards = self.pimc.nextplay(self.player_i, shown_out_suits, self.missing_cards)
                if self.verbose:
                    print("PIMC result:")
                    print("\n".join(f"{Card.from_code(k)}: {v}" for k, v in pimc_resp_cards.items()))
                assert pimc_resp_cards is not None, "PIMC result is None"
                if self.models.pimc_ben_dd_declaring:
                    #print(pimc_resp_cards)
                    dd_resp_cards, claims = self.get_cards_dd_evaluation(trick_i, leader_i, tricks52, current_trick52, players_states, probability_of_occurence, quality)

                    if preempted and self.models.pimc_after_preempt:
                        weight = 1 - self.models.pimc_after_preempt_weight
                    else:
                        weight = self.models.pimc_ben_dd_declaring_weight
                    merged_card_resp = self.merge_candidate_cards(pimc_resp_cards, dd_resp_cards, "PIMC", weight, quality)
                else:
                    merged_card_resp = pimc_resp_cards
                self.update_with_alphamju(card_resp_alphamju, merged_card_resp)
                card_resp = self.pick_card_after_pimc_eval(trick_i, leader_i, current_trick52, tricks52, players_states, merged_card_resp, bidding_scores, quality, samples, play_status, self.missing_cards, claims, shown_out_suits, card_scores_nn)            
            else:
                if self.pimc_defending and (self.player_i == 0 or self.player_i == 2):
                    pimc_resp_cards = self.pimc.nextplay(self.player_i, shown_out_suits, self.missing_cards)
                    if self.verbose:
                        print("PIMCDef result:")
                        print("\n".join(f"{Card.from_code(k)}: {v}" for k, v in pimc_resp_cards.items()))

                    assert pimc_resp_cards is not None, "PIMCDef result is None"
                    if self.models.pimc_ben_dd_defending:
                        #print(pimc_resp_cards)
                        dd_resp_cards, claims = self.get_cards_dd_evaluation(trick_i, leader_i, tricks52, current_trick52, players_states, probability_of_occurence, quality)
                        #print(dd_resp_cards)
                        merged_card_resp = self.merge_candidate_cards(pimc_resp_cards, dd_resp_cards, "PIMCDef", self.models.pimc_ben_dd_defending_weight, quality)
                    else:
                        merged_card_resp = pimc_resp_cards
                    self.update_with_alphamju(card_resp_alphamju, merged_card_resp)
                    card_resp = self.pick_card_after_pimc_eval(trick_i, leader_i, current_trick52, tricks52, players_states, merged_card_resp, bidding_scores, quality, samples, play_status, self.missing_cards, claims, shown_out_suits, card_scores_nn)            
                    
                else:
                    dd_resp_cards, claims = self.get_cards_dd_evaluation(trick_i, leader_i, tricks52, current_trick52, players_states, probability_of_occurence, quality)
                    self.update_with_alphamju(card_resp_alphamju, dd_resp_cards)
                    card_resp = self.pick_card_after_dd_eval(trick_i, leader_i, current_trick52, tricks52, players_states, dd_resp_cards, bidding_scores, quality, samples, play_status, self.missing_cards, claims, shown_out_suits, card_scores_nn)

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

    def get_cards_dd_evaluation(self, trick_i, leader_i, tricks52, current_trick52, players_states, probabilities_list, bidding_quality):
        
        n_samples = players_states[0].shape[0]
        assert n_samples > 0, "No samples for DDSolver"

        use_probability = self.models.use_probability 
        cards_played = list([card for trick in tricks52 for card in trick])
        # All previously played pips are also unavailable, so we use the original dummy and not what we can see
        # unavailable_cards = set(list(np.nonzero(self.hand52)[0]) + list(np.nonzero(self.dummy)[0]) + current_trick52)
        unavailable_cards = set(list(np.nonzero(self.hand52)[0]) + list(np.nonzero(self.public52)[0]) + current_trick52 + cards_played)

        pips = [
            [c for c in range(7, 13) if i*13+c not in unavailable_cards] for i in range(4)
        ]

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
            print("Calculating tricks. Using probability {}".format(use_probability))
        if use_probability:
            card_tricks = self.dds.expected_tricks_dds_probability(dd_solved, probabilities_list)
        else:
            card_tricks = self.dds.expected_tricks_dds(dd_solved)
        #print(card_tricks)
        making = self.dds.p_made_target(tricks_needed)(dd_solved)

        if self.models.use_real_imp_or_mp:
            if self.verbose:
                print(f"Probabilities: [{', '.join(f'{x:>6.2f}' for x in probabilities_list[:20])}{' ]' if len(probabilities_list) <= 20 else '...]'}")
                self.dds.print_dd_results(dd_solved)

            # print("Calculated scores")
            real_scores = calculate.calculate_score(dd_solved, self.n_tricks_taken, self.player_i, self.score_by_tricks_taken)
            if self.verbose:
                print("Real scores")
                print("\n".join(
                    f"{Card.from_code(int(k))}: [{', '.join(f'{x:>5}' for x in v[:20])}{' ]' if len(v) <= 20 else '...'}"
                    for k, v in real_scores.items()
                ))

            if use_probability:
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
            if use_probability:
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
        max_value = round(max(card_tricks.values()),1)
        claim_cards = [k for k, v in card_tricks.items() if abs(v - max_value) <= 0.0001]
        for key in dd_solved.keys():
            card_result[key] = (card_tricks[key], card_ev[key], making[key], "")
            if self.verbose:
                print(f'{deck52.decode_card(key)} {card_tricks[key]:0.3f} {card_ev[key]:5.2f} {making[key]:0.2f}')

        if self.verbose:
            print(f'dds took: {(time.time() - t_start):0.4f}')
        return card_result, (claim_cards, max_value)
    
    
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

        def extract_symbols(hand, suit_index):
            """Extracts cards for a suit from a hand."""
            start, end = suit_indices(suit_index)
            return ''.join([symbols[i] for i, value in enumerate(hand[start:end]) if value == 1])

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
            #This need to evaluate hw many tricks we can take in the suit
            print("We do not want to create tricks for the opponents")
            print(len(opponents_hand)/2, max_tricks_possible, sure_tricks)
            # We do not want to create tricks for the opponents
            if len(opponents_hand)/2 > (max_tricks_possible - sure_tricks):
                return (max_tricks_possible - sure_tricks - (len(opponents_hand)/2)) / 100

            return (max_tricks_possible) / 100

        results = {}

        for suit_index in range(4):
            # Extract cards for the current suit
            our_hand1 = extract_cards(our_hands[0], suit_index)
            our_hand2 = extract_cards(our_hands[1], suit_index)
            opponents_hand = extract_cards(missing_cards, suit_index)
            # Calculate score tricks for the current suit
            if len(opponents_hand) == 0:
                results[suit_index] = 0
                continue

            max_tricks_possible = max(len(our_hand1), len(our_hand2))
            if max_tricks_possible < 2:
                results[suit_index] = 0
                continue

            our_hand1_symbols = extract_symbols(our_hands[0], suit_index)
            our_hand2_symbols = extract_symbols(our_hands[1], suit_index)
            opponents_hand_symbols = extract_symbols(missing_cards, suit_index)
            tricks = self.suitc.get_suit_tricks(our_hand1_symbols, our_hand2_symbols, opponents_hand_symbols)
            if max_tricks_possible == tricks:
                results[suit_index] = 0
                continue
            if len(opponents_hand) > (max_tricks_possible):
                results[suit_index] = round((max_tricks_possible - tricks - (len(opponents_hand)/2)) / 10,2)
            else:
                results[suit_index] = round((max_tricks_possible - tricks) / 10, 2)

            #tricks = sure_tricks_in_suit(our_hand1, our_hand2, opponents_hand)
            #results[suit_index] = tricks
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
    def calculate_trump_adjust(self, play_status, tricks52):
        trump_adjust = 0
        # No adjust for NT
        if self.strain_i == 0:
            return trump_adjust
        # No adjust unless we are making a lead
        if play_status != "Lead":
            return trump_adjust
         
        # Only in suit contract and if we are on lead and we are declaring
        if self.player_i % 2 ==0:
            return trump_adjust

        # Only adjust if we are controlling the suit
        # Remember adjust is helping declarer so we still play trump if DD suggest it
        if self.missing_cards_initial[self.strain_i-1] > 5:
            return trump_adjust

        used_trumps = []
        for trick in tricks52:
            for card in trick:
                if card // 13 == self.strain_i -1:
                    used_trumps.append(card % 13)
        highest_trump = -1
        for i in range(13):  # Check numbers from 0 to 12
            if i not in used_trumps:
                highest_trump = i
                break

        if self.verbose:
            print("Used trumps", used_trumps, highest_trump, self.missing_cards[self.strain_i-1], self.hand52[highest_trump + (self.strain_i-1) * 13], self.public52[highest_trump + (self.strain_i-1) * 13])
        # Only 1 trump left, and they have it
        if self.missing_cards[self.strain_i-1] == 1 and self.hand52[highest_trump + (self.strain_i-1) * 13] + self.public52[highest_trump + (self.strain_i-1) * 13] == 0:
            # If they have the highest we will not adjust the play of trump
            return trump_adjust
        
        # Any outstanding trump?
        if self.models.draw_trump_reward > 0 and self.missing_cards[self.strain_i-1] > 0:
            trump_adjust = self.models.draw_trump_reward
        # Just to be sure we won't show opps that they have no trump
        if self.models.draw_trump_penalty > 0 and self.missing_cards[self.strain_i-1] == 0:
            trump_adjust = -self.models.draw_trump_penalty
        if self.models.use_real_imp_or_mp:
            if self.models.matchpoint:
                trump_adjust = trump_adjust * 2
            else:
                if (trump_adjust > 0):
                    trump_adjust = trump_adjust * 2
                else:
                    trump_adjust = trump_adjust / 2
        if self.verbose:
            print("Trump adjust", trump_adjust)
        return trump_adjust

    def pick_card_after_pimc_eval(self, trick_i, leader_i, current_trick, tricks52,  players_states, card_dd, bidding_scores, quality, samples, play_status, missing_cards, claim, shown_out_suits, card_scores_nn):
        bad_play = []
        print("Claim", claim)
        claim_cards, claim_tricks = claim
        if claim_cards :
            if claim_tricks > 10 - trick_i:
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
                    n_samples=50,
                    tricks=claim_tricks,
                )
                claim_cards = [card for card in claim_cards if card not in bad_play]
                # If no claim card left, then we won't adjust bad_play cards
                if not claim_cards:
                    bad_play = []
            else:
                claim_cards = []

        if self.verbose:
            print(f"Claim cards after check: {claim_cards}, Bad claim cards {bad_play}. Tricks {claim_tricks}")

        # Create a lookup dictionary to find the scores
        card_nn = {c: round(s, 3) for c, s in zip(np.arange(self.models.n_cards_play), card_scores_nn)}
        
        trump_adjust = self.calculate_trump_adjust(play_status, tricks52)
    
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
                    if self.models.matchpoint:
                        adjust_card += round(insta_score * self.models.play_reward_threshold_NN_factor_MP,2)
                    else:
                        adjust_card += round(insta_score * self.models.play_reward_threshold_NN_factor_IMP,2)
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
            claim = -1 if not claim_cards else claim_tricks
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

    def pick_card_after_dd_eval(self, trick_i, leader_i, current_trick, tricks52, players_states, card_dd, bidding_scores, quality, samples, play_status, missing_cards, claim, shown_out_suits, card_scores_nn):
        bad_play = []
        print(f"Claim cards before check: {claim}")
        claim_cards, claim_tricks = claim
        if claim_cards :
            if claim_tricks > 10 - trick_i:
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
                    n_samples=50,
                    tricks=claim_tricks,
                )
                claim_cards = [card for card in claim_cards if card not in bad_play]
                # If no claim card left, then we won't adjust bad_play cards
                if not claim_cards:
                    bad_play = []
            else:
                claim_cards = []

        if self.verbose:
            print(f"Claim cards after check: {claim_cards}, Bad claim cards {bad_play}. Tricks {claim_tricks}")

        # Create a lookup dictionary to find the scores
        card_nn = {c: round(s, 3) for c, s in zip(np.arange(self.models.n_cards_play), card_scores_nn)}

        trump_adjust = self.calculate_trump_adjust(play_status, tricks52)

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
                if self.models.matchpoint:
                    adjust_card += round(insta_score * self.models.play_reward_threshold_NN_factor_MP,2)
                else:
                    adjust_card += round(insta_score * self.models.play_reward_threshold_NN_factor_IMP,2)
            # If we can take rest we don't adjust, then NN will decide if equal
            # Another option could be to resample the hands without restrictions
            if e_tricks == 13 - trick_i:
                # Calculate valid claim cards
                if card32 // 8 != self.strain_i - 1:
                    adjust_card = 0
            if card52 in bad_play:
                adjust_card = -0.05            
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
            claim = -1 if not claim_cards else claim_tricks
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
                    msg= (f"{msg}|adjust={adjust_card}" if adjust_card != 0 else msg)
                )
            
        return card
