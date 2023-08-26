import functools
import numpy as np

from bidding import bidding

TRICK_VAL = {'C': 20, 'D': 20, 'H': 30, 'S': 30, 'N': 30}

def score(contract, is_vulnerable, n_tricks):
    if contract == "Pass":
        return 0
    level = int(contract[0])
    strain = contract[1]
    doubled = 'X' in contract
    redoubled = 'XX' in contract

    target = 6 + level

    final_score = 0
    if n_tricks >= target:
        # contract made
        base_score = level * TRICK_VAL[strain]
        if strain == 'N':
            base_score += 10
        bonus = 0
        
        # doubles and redoubles
        if redoubled:
            base_score *= 4
            bonus += 100
        elif doubled:
            base_score *= 2
            bonus += 50
        
        # game bonus
        if base_score < 100:
            bonus += 50
        else:
            bonus += 500 if is_vulnerable else 300
        # slam bonus
        if level == 6:
            bonus += 750 if is_vulnerable else 500
        elif level == 7:
            bonus += 1500 if is_vulnerable else 1000

        n_overtricks = n_tricks - target
        overtrick_score = 0
        if redoubled:
            overtrick_score = n_overtricks * (400 if is_vulnerable else 200)
        elif doubled:
            overtrick_score = n_overtricks * (200 if is_vulnerable else 100)
        else:
            overtrick_score = n_overtricks * TRICK_VAL[strain]

        final_score = base_score + overtrick_score + bonus
    else:
        # contract failed
        n_undertricks = target - n_tricks
        undertrick_values = []
        if is_vulnerable:
            undertrick_values = [100] * 13
            if redoubled:
                undertrick_values = [400] + [600] * 12
            elif doubled:
                undertrick_values = [200] + [300] * 12
        else:
            undertrick_values = [50] * 13
            if redoubled:
                undertrick_values = [200, 400, 400] + [600] * 10
            elif doubled:
                undertrick_values = [100, 200, 200] + [300] * 10
        
        final_score = -sum(undertrick_values[:n_undertricks])
    
    return final_score

@functools.lru_cache()
def contract_scores_by_trick(contract, vuln):
    scores = np.zeros(14)
    is_vuln = [vuln[0], vuln[1], vuln[0], vuln[1]]['NESW'.index(contract[-1])]
    for i in range(14):
        scores[i] = score(contract, is_vuln, i)
    return scores
