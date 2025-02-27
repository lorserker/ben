import sys
import json
sys.path.append('../../src')

from bba.BBA import BBABotBid

our_system_file = "../../BBA/CC/BEN-21GF.bbsa"
their_system_file = "../../BBA/CC/BEN-21GF.bbsa"
position = 2
hand = "2.T542.K8542.KJT"
vuln = [True, True]
dealer = 2
scoring_matchpoint = False
verbose = False

bot = BBABotBid(our_system_file, their_system_file, position, hand, vuln, dealer, scoring_matchpoint, verbose)
# hand = "AQT8.AK.AK6.A986"
#print(bot.find_info(['PASS','2C','PASS','2H','PASS','2N','PASS','3D','PASS','3S','PASS','3N','PASS','4D','PASS','4N','PASS','5D','PASS']))
info = bot.find_aces(['PASS', 'PASS', '2C', 'PASS', '3D', 'PASS', '4N', 'PASS'])
print(json.dumps(info))

