from sklearn.metrics import auc
import streamlit as st

import asyncio
import sys
sys.path.append('src')

from game import Models, conf
from bots import BotBid

st.text('start')

models = Models.from_conf(conf.load('default.conf'))

# East deals, EW vulnerable.
vuln_ns, vuln_ew = False, True

# you sit West and hold:
hand = '73.KJ83.AT2.T962'

input_text = st.text_input('card bids play', f'{hand} _ 1D 1S')

# the auction goes:
# (a few words about 'PAD_START':
# the auction is padded to dealer North
# if North is not dealer, than we have to put in a 'PAD_START' for every seat that was skipped
# if East deals we have one pad (because North is skipped)
# if South deals we have two pads (because North and East are skipped)
# etc.)
inputs = input_text.split()
hand = inputs[0]
auction = []
for bid in inputs[1:]:
    if bid == '_':
        bid = 'PAD_START'
    if bid.lower() == 'p':
        bid = 'PASS'
    else:
        bid = bid.upper()
    auction.append(bid)

bot_bid = BotBid([vuln_ns, vuln_ew], hand, models)

st.text(bot_bid.bid(auction).bid)



st.text('end')



