"""Quick DDS verification for sample 1 from Board 18 analysis.

Tests whether CQ lead gives 6 tricks in this position:
  N(=real East): .5.Q64.952  (H5, DQ64, C952)
  E(=real South): ..T.QJT743  (DT, CQJT743)
  S(=real West): .7642.32.8  (H7642, D32, C8)
  W(=real North): J85..KJ98. (SJ85, DKJ98)

Trump = Diamonds, East leads.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ddsolver import dds
import ctypes

dds.SetMaxThreads(0)

bo = dds.boardsPBN()
solved = dds.solvedBoards()

# Test 1: Sample 1, 7-card position
pbn = b"N:.5.Q64.952 ..T.QJT743 .7642.32.8 J85..KJ98."

# Test 3: Actual deal, 7-card position (when CQ/DT decision was made)
# Reconstructed from play history and 5-card ending:
#   North(=declarer): SJ85, DKJ98 (3+4=7)
#   South(=dummy): DT, CQJT743 (1+6=7)  [CQ,CJ,CT,C7,C4,C3 = 6 clubs]
#   East: H4, DQ62, C985 (1+3+3=7)
#   West: H7652, D43, C2 (4+2+1=7)
# BEN mapping: N:East South West North
pbn3 = b"N:.4.Q62.985 ..T.QJT743 .7652.43.2 J85..KJ98."

bo.noOfBoards = 3
bo.deals[0].trump = 2  # Diamonds
bo.deals[0].first = 1  # East leads (= real South/dummy)

# Test 2: Actual deal, 5-card position after overruff
# North(=declarer) to play: SJ8, DKJ9
# South(=dummy): DT, CT743
# East: H4, DQ62, C9
# West: H7652, D4
# BEN mapping: N:East South West North
pbn2 = b"N:.4.Q62.9 ..T.T743 .7652.4. J8..KJ9."
bo.deals[1].trump = 2  # Diamonds
bo.deals[1].first = 3  # West leads (= real North/declarer)

# Test 3: Actual deal, 7-card position
# East(=South/dummy) leads, same as test 1
bo.deals[2].trump = 2  # Diamonds
bo.deals[2].first = 1  # East leads (= real South/dummy)

for h in range(3):
    for i in range(3):
        bo.deals[h].currentTrickSuit[i] = 0
        bo.deals[h].currentTrickRank[i] = 0
    bo.target[h] = -1
    bo.solutions[h] = 3
    bo.mode[h] = 1

bo.deals[0].remainCards = pbn
bo.deals[1].remainCards = pbn2
bo.deals[2].remainCards = pbn3

res = dds.SolveAllBoards(ctypes.pointer(bo), ctypes.pointer(solved))
if res != 1:
    error_message = dds.get_error_message(res)
    print(f"Error: {res} - {error_message}")
    sys.exit(1)

card_rank_names = ['A','K','Q','J','T','9','8','7','6','5','4','3','2']
suit_names = ['S','H','D','C']

labels = [
    f"Sample 1 (7-card): {pbn.decode()}\n  Trump=D, East(=South/dummy) leads",
    f"Actual deal (5-card): {pbn2.decode()}\n  Trump=D, West(=North/declarer) leads",
    f"Actual deal (7-card): {pbn3.decode()}\n  Trump=D, East(=South/dummy) leads"
]

for h in range(3):
    fut = ctypes.pointer(solved.solvedBoards[h])
    print(labels[h])
    print(f"  Cards returned: {fut.contents.cards}")
    for i in range(fut.contents.cards):
        suit_i = fut.contents.suit[i]
        rank = fut.contents.rank[i]
        score = fut.contents.score[i]
        card_name = suit_names[suit_i] + card_rank_names[14 - rank]
        print(f"    {card_name}: {score} tricks")
    print()
