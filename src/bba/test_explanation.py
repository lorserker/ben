import sys
import json
sys.path.append('../../src')

from bba.BBA import BBABotBid

our_system_file = "../../BBA/CC/BEN-21GF.bbsa"
their_system_file = "../../BBA/CC/BEN-21GF.bbsa"
position = 1
hand = "QJT85.Q6.KT764.2"
vuln = [False, False]
dealer = 2
scoring_matchpoint = False
verbose = True

bot = BBABotBid(our_system_file, their_system_file, position, hand, vuln, dealer, scoring_matchpoint, verbose)
explanation = bot.explain_last_bid(['PASS'])
print(explanation)
explanation = bot.explain_last_bid(['PASS','2S'])
print(explanation)
explanation = bot.explain_last_bid(['PASS','2S', 'PASS'])
print(explanation)
explanation = bot.explain_last_bid(['PASS','2S', 'PASS','2N'])
print(explanation)

