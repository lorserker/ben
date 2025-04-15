import sys
import json
sys.path.append('../../src')
import platform
print(platform.architecture())

from bba.BBA import BBABotBid

our_system_file = "../../BBA/CC/GIB-BBO.bbsa"
their_system_file = "../../BBA/CC/GIB-BBO.bbsa"
position = 2
hand = "AT762.K74.Q8.654"
vuln = [True, True]
dealer = 0
scoring_matchpoint = False
verbose = False

bot = BBABotBid(our_system_file, their_system_file, position, hand, vuln, dealer, scoring_matchpoint, verbose)
bid = bot.bid(['PASS', '1H'])
print(bid)

