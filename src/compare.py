import sys
import math
import bisect
import json


IMP = [10, 40, 80, 120, 160, 210, 260, 310, 360, 420, 490, 590, 740, 890, 1090, 1290, 1490, 1740, 1990, 2240, 2490, 3490, 3990]


def get_imps(score1, score2):
    score_diff = score1 - score2
    imp = bisect.bisect_left(IMP, int(math.fabs(score_diff)))
    if score_diff >= 0:
        return imp
    else:
        return -imp


def results_it(fin):
    for line in fin:
        yield json.loads(line)


if __name__ == '__main__':
    results_fnm_1 = sys.argv[1]
    results_fnm_2 = sys.argv[2]

    for res_obj_1, res_obj_2 in zip(results_it(open(results_fnm_1)), results_it(open(results_fnm_2))):
        keys_same = ['dealer', 'vuln', 'north', 'east', 'south', 'west']
        assert all([res_obj_1[key] == res_obj_2[key] for key in keys_same])
        cmp_obj = {
            key:res_obj_1[key] for key in keys_same
        }
        keys_compare = ['auction', 'contract', 'lead', 'dd_tricks', 'dd_score']
        for key in keys_compare:
            cmp_obj[key] = [res_obj_1[key], res_obj_2[key]]

        if tuple(res_obj_1['auction']) == tuple(res_obj_2['auction']):
            cmp_obj['imp'] = 0
        else:
            cmp_obj['imp'] = get_imps(*cmp_obj['dd_score'])

        print(json.dumps(cmp_obj))
