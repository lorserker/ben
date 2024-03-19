import numpy as np

n_samples = 100
PBN = ""
decl_tricks_softmax = np.zeros((n_samples, 14), dtype=np.int32)

# Shuffle to two unknown hands
hands_pbn = []
from ddsolver import ddsolver
dd = ddsolver.DDSolver()
# 4 suits + NT
for i in range (5):
    # Lead from 2 sides
    for j in range(2):
        dd_solved = dd.solve(i, j, [], hands_pbn, 1)
        # Only use 1st element from the result
        first_key = next(iter(dd_solved))
        first_item = dd_solved[first_key]
        decl_tricks_softmax[i,13 - first_item[0]] = 1


# Find the suit, where we take the most tricks
# Find the bid with the best score
# scores_by_trick[i] = scoring.contract_scores_by_trick(contract, tuple(self.vuln))