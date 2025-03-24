import sys
import json
sys.path.append('../../src')

from bba.BBA import BBABotBid

our_system_file = "../../BBA/CC/GIB-BBO.bbsa"
their_system_file = "../../BBA/CC/GIB-BBO.bbsa"
position = 1
hand = "AT762.K74.Q8.654"
vuln = [True, True]
dealer = 0
scoring_matchpoint = False
verbose = True

bot = BBABotBid(our_system_file, their_system_file, position, hand, vuln, dealer, scoring_matchpoint, verbose)
result = bot.list_bids(["1D","PASS","2H","PASS","3D","PASS"])
print(result)

