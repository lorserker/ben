import sys
import json
sys.path.append('../../src')

from bba.BBA import BBABotBid

our_system_file = "../../BBA/CC/GIB-BBO.bbsa"
their_system_file = "../../BBA/CC/GIB-BBO.bbsa"
# North = 0
position = 3
hand = "K87652.J9872.Q.Q"
vuln = [True, True]
dealer = 0
scoring_matchpoint = False
verbose = False

bot = BBABotBid(our_system_file, their_system_file, position, hand, vuln, dealer, scoring_matchpoint, verbose)
bot.get_sample( ['1D', 'PASS', '2N', 'PASS', '3N', 'PASS', 'PASS', 'PASS'] )

# North = 0
position = 2
hand = "985.KQ832.T.AK53"
vuln = [False, False]
dealer = 0
scoring_matchpoint = False
verbose = False

bot = BBABotBid(our_system_file, their_system_file, position, hand, vuln, dealer, scoring_matchpoint, verbose)
bot.get_sample( ['PASS', '1C', '1H', 'PASS', 'PASS', 'X', 'PASS', 'PASS', 'PASS'] )

# North = 0
position = 3
hand = "63.T7.K7652.K952"
vuln = [True, True]
dealer = 1
scoring_matchpoint = False
verbose = False

bot = BBABotBid(our_system_file, their_system_file, position, hand, vuln, dealer, scoring_matchpoint, verbose)
bot.get_sample( ['1H', '1N', 'PASS', '4H', 'PASS', '4S', 'PASS', 'PASS', 'PASS'] )
