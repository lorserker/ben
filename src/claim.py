import time
import deck52
import random
import numpy as np


class Claimer:

    def __init__(self, verbose) -> None:
        self.verbose = verbose
        from ddsolver import ddsolver
        self.dd = ddsolver.DDSolver()

    def claim(self, strain_i, player_i, hands52, n_samples):
        t_start = time.time()

        hands_pbn = ['W:' + ' '.join([
            deck52.deal_to_str(hand) for hand in hands52
        ])]

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

            sampled_hands_pbn.append('W:' + ' '.join(hands))

        max_min_tricks = min(
            self._get_max_min_tricks(strain_i, player_i, hands_pbn),
            self._get_max_min_tricks(strain_i, player_i, sampled_hands_pbn)
        )
        
        if self.verbose:
            print(f'player {player_i} could claim {max_min_tricks} tricks.')
            print(f'claim check took {time.time() - t_start}')

        return max_min_tricks
    
    def _get_max_min_tricks(self, strain_i, player_i, hands_pbn):
        dd_solved = self.dd.solve(strain_i, player_i, [], hands_pbn)
        
        max_min_tricks = 0
        for _, dd_tricks in dd_solved.items():
            max_min_tricks = max(max_min_tricks, min(dd_tricks))
        
        return max_min_tricks


def _hand_from_cards(n, cards):
    hand = np.zeros(n, dtype=np.int32)
    hand[cards] = 1
    return hand
