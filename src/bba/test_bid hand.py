import sys
import json
sys.path.append('../../src')

from bba.BBA import BBABotBid

our_system_file = "../../BBA/CC/GIB-BBO.bbsa"
their_system_file = "../../BBA/CC/GIB-BBO.bbsa"
position = 2
hand = "AT762.K74.Q8.654"
vuln = [False, True]
dealer = 0
scoring_matchpoint = False
verbose = False

bot = BBABotBid(our_system_file, their_system_file, position, hand, vuln, dealer, scoring_matchpoint, verbose)
bid = bot.bid_hand(["1H"], "N:5.AQ8652.AK.AQT7 AT762.K74.Q8.654 QJ43.T9.J9732.92 K98.J3.T654.KJ83")
print(bid)

