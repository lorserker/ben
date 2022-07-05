from cv2 import bitwise_and
from sklearn.metrics import auc
import streamlit as st

import sys
sys.path.append('src')

from game import Models, conf
from bots import BotBid


def st_print(hand):
    lines = []
    suits = hand.split('.')
    for i in range(4):
        s = 'SHDC'[i]
        lines.append(f'{s} {suits[i]}')
    return lines

models = Models.from_conf(conf.load('default.conf'))

# East deals, EW vulnerable.
vuln_ns, vuln_ew = False, True

# you sit West and hold:
hands = 'A8.Q752.T54.JT63 K54.T643.A92.A72 JT932.K9.Q86.854 Q76.AJ8.KJ73.KQ9'

input_text = st.text_input('card bids play', f'{hands} _ p')

# the auction goes:
# (a few words about 'PAD_START':
# the auction is padded to dealer North
# if North is not dealer, than we have to put in a 'PAD_START' for every seat that was skipped
# if East deals we have one pad (because North is skipped)
# if South deals we have two pads (because North and East are skipped)
# etc.)
inputs = input_text.split()
hands = inputs[:4]
bidder_bots = [BotBid([vuln_ns, vuln_ew], hand, models) for hand in hands]
auction = []
for bid in inputs[4:]:
    if bid == '_':
        bid = 'PAD_START'
    if bid.lower() == 'p':
        bid = 'PASS'
    else:
        bid = bid.upper()
    auction.append(bid)

turn = len(auction) % 4
lines = [f'Vul: NS {vuln_ns} EW {vuln_ew}']
layout = [(0, 1), (1, 2), (2, 1), (1, 0)]
row = [[[]]*3, [[]]*3, [[]]*3]

for i in range(4):
    r,col = layout[i]
    suits = st_print(hands[i])
    row[r][col] = suits

# nice ascii graph to display cards and bidding
for r in range(3):
    for subr in range(4):
        line = ''
        for col in range(3):
            if len(row[r][col]) == 0:
                if col == 0:
                    line += ' '*7
                elif col == 1:
                    line += ' '*(15-len(line))
            else:
                line += row[r][col][subr]
        lines.append(line)    
line = 'North East  South West'
lines.append(line)
line = ''
for i in range(len(auction)):
    bid = auction[i]
    if bid == 'PAD_START':
        bid = ' '*6
    if len(bid) < 6:
        bid += ' '*(6-len(bid))
    if line != '' and i % 4 == 0:
        lines.append(line)
        line = ''
    line += bid
i = len(auction)
if line != '' and i % 4 == 0:
    lines.append(line)
    line = ''
bid = bidder_bots[turn].bid(auction).bid
line += bid
lines.append(line)

st.text('\n'.join(lines))




