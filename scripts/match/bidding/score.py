
import sys
sys.path.append('../../../src')

import json

import deck52
import scoring

from ddsolver.ddsolver import DDSolver
from bidding import bidding
from objects import Card
from bidding.binary import parse_hand_f
from auction import VULN


DD = DDSolver(dds_mode=1)

def get_dd_score(obj):
    if obj['contract'] is None:
        if 'lead' not in obj:
            obj['lead'] = None
        return 0, 0

    hands = [obj['north'], obj['east'], obj['south'], obj['west']]
    current_trick = []
    if 'lead' in obj:
        current_trick = [Card.from_symbol(obj['lead']).code()]
    strain_i = bidding.get_strain_i(obj['contract'])
    decl_i = bidding.get_decl_i(obj['contract'])
    leader_i = (decl_i+1) % 4
    if 'lead' in obj:
        hands[leader_i] = hand_remove_card(hands[leader_i], obj['lead'])

    hands_pbn = 'W:' + ' '.join(hands[leader_i:] + hands[:leader_i])

    sol = DD.solve(strain_i, 0, current_trick, [hands_pbn])
    dd_tricks = max(vals[0] for vals in sol.values())
    if 'lead' not in obj:
        card_index = [v[0] for v in sol.values()].index(dd_tricks)
        c = list(sol.keys())[card_index]
        obj['lead'] = 'SHDC'[c//13] + 'AKQJT98765432'[c%13]
        dd_tricks = 13 - dd_tricks
    dd_score = scoring.score(obj['contract'], is_vulnerable=VULN[obj['vuln']][decl_i % 2], n_tricks=dd_tricks)
    dd_score = dd_score if decl_i % 2 == 0 else -dd_score

    return dd_tricks, dd_score

def hand_remove_card(hand_str, card):
    hand = parse_hand_f(52)(hand_str).reshape(-1)
    hand[Card.from_symbol(card).code()] -= 1
    return deck52.hand_to_str(hand)


if __name__ == '__main__':
    for line in sys.stdin:
        obj = json.loads(line)

        dd_tricks, dd_score = get_dd_score(obj)

        obj['dd_tricks'] = dd_tricks
        obj['dd_score'] = dd_score

        print(json.dumps(obj))
