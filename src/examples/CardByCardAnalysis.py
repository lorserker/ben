import sys
sys.path.append('src')

# from nn.models import Models
# from analysis import CardByCard
# from util import parse_lin, display_lin

# models = Models.load('models')   # loading neural networks

# # we specify all the information about a board
# # (it's quite tedious to enter every single thing by hand here,
# # later we'll have an example of how you can give it a board played on BBO)

# dealer = 'S'
# vuln = [True, True]  # fist element is NS, second element is EW

# hands = [
#     'AJ87632.J96.753.',
#     'K9.Q8542.T6.AJ74',
#     'QT4.A.KJ94.KQ986',
#     '5.KT73.AQ82.T532'
# ]

# auction = ['1N', 'PASS', '4H', 'PASS', '4S', 'PASS', 'PASS', 'PASS']

# play = ['C2', 'D3', 'CA', 'C6', 'D6', 'DJ', 'DQ', 'D5', 'DA', 'D7', 'DT', 'D4', 'D8', 'H6', 'H2', 'D9', 'SQ', 'S5', 'S2', 'SK', 'H4', 'HA', 'H7', 'H9', 'S4', 'C3', 'SA', 'S9', 'S3', 'C4', 'ST', 'H3', 'CK', 'C5', 'HJ', 'C7', 'C8', 'CT', 'S6', 'CJ', 'S7', 'H8', 'C9', 'D2', 'S8', 'H5', 'CQ', 'HT', 'SJ', 'HQ', 'DK', 'HK']

# card_by_card = CardByCard(dealer, vuln, hands, auction, play, models)

# calling this starts the analysis
# it will go bid-by-bid and card-by-card, and will take a few moments
# possible mistakes will be annotated with ? or with ?? (if it's a bigger mistake)
# (possible mistake means that the engine does not agree with the bid/play. the engine could be wrong too :))

#card_by_card.analyze()

# the engine does not agree with the 1N opening.
# indeed, it's a little offbeat with a singleton
# let's see what the engine is thinking (what would it bid instead)

#card_by_card.bid_responses[0].to_dict()  # the 0 index is the first bid in the auction

# what about the opening lead? let's see...

#card_by_card.cards['C2'].to_dict()
# QJ6.K652.J85.T98:873.J97.AT764.Q4:5.T83.KQ93.76532:AKT942.AQ4.2.AKJ
# {<Seat.N: 0>: 'QJ6 K652 J85 T98', <Seat.E: 1>: '873 J97 AT764 Q4', <Seat.S: 2>: '5 T83 KQ93 76532', <Seat.W: 3>: 'AKT942 AQ4 2 AKJ'}
# result: N S D 5 1
# result: N H D 5 4
# result: N D D 5 4
# result: N C D 5 6
# result: N N D J 4
# result: E S S 3 12
# result: E H S 3 9
# result: E D S 3 9
# result: E C S 3 8
# result: E N S 3 10
# result: S S H 3 2
# result: S H H 3 4
# result: S D H 3 4
# result: S C C 3 6
# result: S N D 3 4
# result: W S S A 12
# result: W H S A 9
# result: W D S A 9
# result: W C S A 8
# result: W N S A 10


# strain_i 'NSHDC' 0-4
# decl_i 'NESW' 0-3

from ddsolver import ddsolver
import deck52
dd = ddsolver.DDSolver()
#               QJ6.K652.J85.T98:873.J97.AT764.Q4:K5.T83.KQ9.A7652:AT942.AQ4.32.KJ3
#             W:95.QT75.A82.J43 AJ87632.J96.753. K.K8432.QT6.AT75 QT4.A.KJ94.KQ986
hands_pbn = ['N:QJ6.K652.J85.T98 873.J97.AT764.Q4 5.T83.KQ93.76532 AKT942.AQ4.2.AKJ']
current_trick52 = []

for leader_i in range(4):
  for strain_i in range(5):
    #print(Strain(strain), 'SHCDN'[strain], 'NESW'[leader], Seat(leader))
    # strain is CDHSN, dds is SHCDN
    dd_solved = dd.solve(strain_i, leader_i, current_trick52, hands_pbn)
    # card_tricks = ddsolver.expected_tricks(dd_solved)
    #print ('result:', 'NESW'[leader_i], 'NSHDC'[strain_i], 'SHDC'[r.suit[0]], Rank(r.rank[0]), r.score[0])
    res = {}
    for k in dd_solved:
        res[deck52.decode_card(k)] = dd_solved[k]
    res = next(iter(dd_solved.items()))[1][0]
    print('WNES'[leader_i], 'NSHDC'[strain_i], res)


#print('hands_pbn', hands_pbn[0])
#print(strain_i, leader_i, current_trick52)

#card_tricks = ddsolver.expected_tricks(dd_solved)
#card_ev = self.get_card_ev(dd_solved)
#print(dd_solved, card_tricks)
