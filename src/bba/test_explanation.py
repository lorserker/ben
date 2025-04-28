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
dealer = 3
bot = BBABotBid(our_system_file, their_system_file, position, hand, vuln, dealer, scoring_matchpoint, verbose)
explanation, bba_controlled, preempted = bot.explain_auction(['PASS', '2S', 'X', 'PASS', '3C', 'PASS', '3N', 'PASS', 'PASS', 'PASS'])
print(explanation, bba_controlled, preempted)

explanation, bba_controlled, preempted = bot.explain_auction(['PASS', '2S', 'X', 'PASS', '3C', 'PASS', '3N', 'PASS', 'PASS', 'PASS'])
print(explanation, bba_controlled, preempted)

bot = BBABotBid(our_system_file, their_system_file, position, hand, vuln, dealer, scoring_matchpoint, verbose)
explanation, bba_controlled, preempted = bot.explain_auction(['PASS', '2S', 'X', 'PASS', '3C', 'PASS', '3N', 'PASS', 'PASS', 'PASS'])
print(explanation, bba_controlled, preempted)

explanation, bba_controlled, preempted = bot.explain_auction(['PASS', '2S', 'X', 'PASS', '3C', 'PASS', '3N', 'PASS', 'PASS', 'PASS'])
print(explanation, bba_controlled, preempted)

explanation, bba_controlled, preempted = bot.explain_auction(['1H','PASS', '2N', 'PASS', '3H', 'PASS', '3N', 'PASS', '4C', 'PASS', '4D', 'PASS', '4N', 'PASS', '5H', 'PASS', 'PASS', 'PASS'])
print(explanation, bba_controlled, preempted)

explanation = bot.explain_last_bid(['2C','PASS', '2D', 'PASS', '2H', 'PASS', '2S', 'PASS', '2N'])
print(explanation)
