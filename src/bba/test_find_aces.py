import sys
import json
sys.path.append('../../src')

from bba.BBA import BBABotBid

our_system_file = "../../BBA/CC/BEN-21GF.bbsa"
their_system_file = "../../BBA/CC/BEN-21GF.bbsa"
position = 2
hand = "T942.AKT972.Q4.7"
vuln = [True, True]
dealer = 0
scoring_matchpoint = False
verbose = False

bot = BBABotBid(our_system_file, their_system_file, position, hand, vuln, dealer, scoring_matchpoint, verbose)
# hand = "AQT8.AK.AK6.A986"
#print(bot.find_info(['PASS','2C','PASS','2H','PASS','2N','PASS','3D','PASS','3S','PASS','3N','PASS','4D','PASS','4N','PASS','5D','PASS']))
info = bot.find_aces(['2N', 'PASS', '4D', 'PASS', '4H', 'PASS', '4N', 'PASS', '5C', 'PASS'])
print(json.dumps(info))

