import sys
import json
sys.path.append('../../src')

from bba.BBA import BBABotBid

our_system_file = "../../BBA/CC/GIB-BBO.bbsa"
their_system_file = "../../BBA/CC/GIB-BBO.bbsa"
position = 1
hand = "AK943.JT96.T.J92"
vuln = [True, True]
dealer = 0
scoring_matchpoint = False
verbose = False

bot = BBABotBid(our_system_file, their_system_file, position, hand, vuln, dealer, scoring_matchpoint, verbose)
bid = bot.bid(['PASS', 'PASS', '1D', 'PASS', 'PASS', 'X', 'XX', '2C', 'PASS', 'PASS', 'X', '2D', 'PASS', '2S', 'PASS', '3C', 'PASS', 'PASS','X', 'PASS', 'PASS'])
print(bid)

