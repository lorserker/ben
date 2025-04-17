import time
import deck52
import random
import numpy as np
from itertools import combinations

class Claimer:

    def __init__(self, verbose, ddsolver) -> None:
        self.verbose = verbose
        self.dd = ddsolver

    def claimcheck(self, strain_i, player_i, hands52, tricks52, claim_cards, shown_out_suits, missing_cards, current_trick, n_samples, tricks):
        # If any voids we will not manipulate the suit, as we can take finesses if needed
        # We remove any intermediate cards, so there is no finesse 
        hidden_cards = []
        used_cards =  [item for row in tricks52 for item in row] + current_trick

        for i in range(52):
            if i in used_cards:
                continue
            if i in current_trick:
                continue
            if hands52[0][i] == 1:
                continue
            if hands52[1][i] == 1:
                continue
            hidden_cards.append(i)
        used_cards.sort()
        index_for_dummy = 2
        if player_i == 0:
            index_for_dummy = 1  
        if player_i == 2:
            index_for_dummy = 3 
        n = 13 * 4  # Number of cards per hand
        hands = [np.zeros(n, dtype=np.int32) for _ in range(4)]
        for i in range(4):
            # all cards know for this suit
            if missing_cards[i] == 0:
                for j in range(13):
                    card52 = i * 13 + j
                    # Do not transfer cards if played in current trick
                    if card52 in current_trick:
                        continue
                    hands[0][card52] = hands52[0][card52] 
                    hands[index_for_dummy][card52] = hands52[1][card52] 
                continue
            # if we have a card lower than a missing card we swap it with the lovest hidden card in that suit
            hidden_for_suit = [c for c in hidden_cards if c >= i * 13 and c < (i + 1) * 13]
            #print("suit", i, hidden_for_suit)

            for j in range(13):
                card52 = i * 13 + j
                # Do not transfer cards if played in current trick
                if card52 in current_trick:
                    continue
                if hands52[0][card52] == 1:
                    # Remember 0 = Ace
                    if len(hidden_for_suit) == 0 or card52 < hidden_for_suit[0]: 
                        hands[0][card52] = 1
                    else:
                        # Never swap to a higher card
                        if hidden_for_suit[-1] < card52:
                            hands[0][card52] = 1
                            continue
                        hands[0][hidden_for_suit[-1]] = 1
                        for k in range(len(hidden_cards)):
                            if hidden_cards[k] == hidden_for_suit[-1]:
                                hidden_cards[k] = card52
                                break  # Stop after the first replacement
                        hidden_for_suit = hidden_for_suit[:-1]

                if hands52[1][card52] == 1:
                    # Remember 0 = Ace
                    if len(hidden_for_suit) == 0 or card52 < hidden_for_suit[0]: 
                        hands[index_for_dummy][card52] = 1
                    else:
                        # Never swap to a higher card
                        if hidden_for_suit[-1] < card52:
                            hands[index_for_dummy][card52] = 1
                            continue
                        hands[index_for_dummy][hidden_for_suit[-1]] = 1
                        for k in range(len(hidden_cards)):
                            if hidden_cards[k] == hidden_for_suit[-1]:
                                hidden_cards[k] = card52
                                break  # Stop after the first replacement
                        hidden_for_suit = hidden_for_suit[:-1]

        if self.verbose:
            hands_pbn = ['N:' + ' '.join([deck52.deal_to_str(hand) for hand in hands])]
            print("Claiming for player", player_i, hands_pbn)
        
        hands[0] = deck52.deal_to_str(hands[0])
        hands[index_for_dummy] = deck52.deal_to_str(hands[index_for_dummy])

        # With 6 or less cards, we should probably just check all combinations instead of shuffle
        sampled_hands_pbn = []
        # Ensure the number of combinations is correct based on the number of hidden cards
        n_cards = len(hidden_cards) // 2
        card_combinations = list(combinations(hidden_cards, n_cards))  # Get all possible splits

        # Ensure the number of requested samples doesn't exceed available combinations
        n_possible_samples = min(n_samples, len(card_combinations))

        # Select unique samples without replacement
        unique_combinations = random.sample(card_combinations, n_possible_samples)

        # We should check shown_out_suits
        #print("shown_out_suits", shown_out_suits)
        for chosen_combination  in unique_combinations:

            # Create the hands based on the selected combination
            remaining_cards = [card for card in hidden_cards if card not in chosen_combination]
            
            if index_for_dummy == 2:
                hands[3] = deck52.deal_to_str(_hand_from_cards(52, list(chosen_combination)))
                hands[1] = deck52.deal_to_str(_hand_from_cards(52, remaining_cards))
            if index_for_dummy == 1:
                hands[3] = deck52.deal_to_str(_hand_from_cards(52, list(chosen_combination)))
                hands[2] = deck52.deal_to_str(_hand_from_cards(52, remaining_cards))
            if index_for_dummy == 3:
                hands[2] = deck52.deal_to_str(_hand_from_cards(52, list(chosen_combination)))
                hands[1] = deck52.deal_to_str(_hand_from_cards(52, remaining_cards))
            sampled_hands_pbn.append('N:' + ' '.join(hands))

        #print('\n'.join(sampled_hands_pbn))
        #print("Trump",strain_i)
        dd_solved = self.dd.solve(strain_i, (4 - len(current_trick)) % 4, current_trick, sampled_hands_pbn, 3)
        # Filter keys where all values in the list are equal
        equal_value_keys = {key: values[0] for key, values in dd_solved.items() if all(value == values[0] for value in values)}

        if equal_value_keys:
            # Find the maximum value among keys with all equal values
            max_value = max(equal_value_keys.values())
            if self.verbose:
                print(f"Max value: {max_value}")
            if  max_value < tricks:
                # None of the cards give same result for all combinations
                # So we just ignore our claimcheck
                if self.verbose:
                    print(f"No cards yield the needed tricks {tricks} best {max_value}")
                bad_plays = claim_cards
            else:
                # Collect keys that:
                # - Either have all equal values but are NOT the max
                # - Or have non-uniform values
                non_max_keys = [
                    key
                    for key, values in dd_solved.items()
                    if (key in equal_value_keys and equal_value_keys[key] != max_value)
                    or (key not in equal_value_keys)
                ]
                bad_plays = [key for key in non_max_keys if key in claim_cards]
                # This should probably be extended as we might have moved a card to be a pip
                # and DDSolver is not aware of that, and only reports the first card from a sequence
                # will create redundant cards, but that is OK
                for card in claim_cards:
                    if card not in equal_value_keys:
                        bad_plays.append(card)
        else:
            # None of the cards give same result for all combinations
            # So we just ignore our claimcheck
            bad_plays = claim_cards


        if self.verbose:
            print(f"Play without sure claim: {bad_plays}")
        return bad_plays

    def claimapi(self, strain_i, player_i, hands52, n_samples, hidden_cards, current_trick):
        t_start = time.time()

        hands_pbn = ['N:' + ' '.join([deck52.deal_to_str(hand) for hand in hands52])]
        if self.verbose:
            print(f"Claiming for player {player_i} {hands_pbn}")
        sampled_hands_pbn = []
        seen_hand_indexes = [player_i, 3 if player_i == 1 else 1]
        hidden_hand_indexes = [i for i in range(4) if i not in seen_hand_indexes]
        hidden_cards = list(np.nonzero(hidden_cards)[0])

        hands = [None, None, None, None]
        hands[seen_hand_indexes[0]] = deck52.deal_to_str(hands52[seen_hand_indexes[0]])
        hands[seen_hand_indexes[1]] = deck52.deal_to_str(hands52[seen_hand_indexes[1]])

        for i in range(n_samples):
            np.random.shuffle(hidden_cards)
            
            n_cards = len(hidden_cards) // 2
            hands[hidden_hand_indexes[1]] = deck52.deal_to_str(_hand_from_cards(52, hidden_cards[:n_cards]))
            hands[hidden_hand_indexes[0]] = deck52.deal_to_str(_hand_from_cards(52, hidden_cards[n_cards:]))

            sampled_hands_pbn.append('N:' + ' '.join(hands))

        max_min_tricks = self._get_max_min_tricks(strain_i, player_i, sampled_hands_pbn, current_trick)
        
        
        if self.verbose:
            print(f'player {player_i} could claim {max_min_tricks} tricks.')
            print(f'claim check took {time.time() - t_start}')

        return max_min_tricks

    def claim(self, strain_i, player_i, hands52, n_samples):
        t_start = time.time()

        hands_pbn = ['N:' + ' '.join([deck52.deal_to_str(hand) for hand in hands52])]

        if self.verbose:
            print(f"Claiming for player {player_i} {hands_pbn}")
        sampled_hands_pbn = []
        seen_hand_indexes = [player_i, 3 if player_i == 1 else 1]
        hidden_hand_indexes = [i for i in range(4) if i not in seen_hand_indexes]
        hidden_cards = (
            list(np.nonzero(hands52[hidden_hand_indexes[0]])[0]) +
            list(np.nonzero(hands52[hidden_hand_indexes[1]])[0])
        )

        hands = [None, None, None, None]
        hands[seen_hand_indexes[0]] = deck52.deal_to_str(hands52[seen_hand_indexes[0]])
        hands[seen_hand_indexes[1]] = deck52.deal_to_str(hands52[seen_hand_indexes[1]])

        for i in range(n_samples):
            np.random.shuffle(hidden_cards)
            
            n_cards = len(hidden_cards) // 2
            hands[hidden_hand_indexes[0]] = deck52.deal_to_str(_hand_from_cards(52, hidden_cards[:n_cards]))
            hands[hidden_hand_indexes[1]] = deck52.deal_to_str(_hand_from_cards(52, hidden_cards[n_cards:]))

            sampled_hands_pbn.append('N:' + ' '.join(hands))

        max_min_tricks = min(
            self._get_max_min_tricks(strain_i, player_i, hands_pbn, []),
            self._get_max_min_tricks(strain_i, player_i, sampled_hands_pbn, []),
        )
        
        if self.verbose:
            print(f'player {player_i} could claim {max_min_tricks} tricks.')
            print(f'claim check took {time.time() - t_start}')

        return max_min_tricks

    def _get_max_min_tricks(self, strain_i, player_i, hands_pbn, current_trick):
        dd_solved = self.dd.solve(strain_i, (player_i-len(current_trick)) % 4, current_trick, hands_pbn, 1)
        
        max_min_tricks = 0
        for _, dd_tricks in dd_solved.items():
            max_min_tricks = max(max_min_tricks, min(dd_tricks))
        
        return max_min_tricks


def _hand_from_cards(n, cards):
    hand = np.zeros(n, dtype=np.int32)
    hand[cards] = 1
    return hand
