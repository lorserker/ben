
import clr
import ctypes
import sys
import os

sys.path.append("..")
from src.objects import BidResp
from bidding import bidding

BEN_HOME = os.getenv('BEN_HOME') or '..'
BIN_FOLDER = os.path.join(BEN_HOME, 'bin')
if sys.platform == 'win32':
    EPBot_LIB = 'EPBot86'
elif sys.platform == 'darwin':
    EPBot_LIB = 'N/A'
else:
    EPBot_LIB = 'N/A'

EPBot_PATH = os.path.join(BIN_FOLDER, EPBot_LIB)

#The error you're encountering suggests that the info_meaning property is not directly accessible as an attribute in the EPBot class. Instead, you should use the get_info_meaning and set_info_meaning methods that are automatically generated for properties with a getter and setter in .NET.


class BBABotBid: 

    # Define constants for system types and conventions (replace these with actual values)
    C_NS = 0
    C_WE = 1
    C_INTERPRETED = 13

    SCORING_MATCH_POINTS = 0
    SCORING_IMP = 1

    # Define your conventions in an array
    conventions_list = {
        "1D opening with 4 cards": False,
        "1D opening with 5 cards": False,
        "1m opening allows 5M": True,
        "1M-3M blocking": False,
        "1M-3M inviting": True,
        "5431 convention after 1NT": True,
        "5NT pick a slam": True,
        "Bergen": False,
        "Blackwood 0123": False,
        "Blackwood 0314": True,
        "Blackwood 1430": False,
        "BROMAD": False,
        "Canape style": False,
        "Cappelletti": False,
        "Checkback": True,
        "Crosswood 0123": False,
        "Crosswood 0314": False,
        "Crosswood 1430": False,
        "Cue bid": True,
        "Drury": True,
        "Exclusion": False,
        "Extended acceptance after NT": False,
        "Fit showing jumps": True,
        "Forcing 1NT": False,
        "Fourth suit": True,
        "French 2D": False,
        "Gambling": False,
        "Gazzilli": False,
        "Gerber": True,
        "Ghestem": False,
        "Inverted minors": True,
        "Jacoby 2NT": True,
        "Jordan Truscott 2NT": True,
        "Kickback 0123": False,
        "Kickback 0314": False,
        "Kickback 1430": False,
        "Leaping Michaels": False,
        "Lebensohl after 1NT": True,
        "Lebensohl after 1m": False,
        "Lebensohl after double": True,
        "Maximal Doubles": True,
        "Michaels Cuebid": True,
        "Mini Splinter": False,
        "Minor Suit Stayman after 1NT": True,
        "Minor Suit Stayman after 2NT": True,
        "Minor Suit Transfers after 1NT": False,
        "Minor Suit Transfers after 2NT": False,
        "Mixed raise": True,
        "Multi": False,
        "Multi-Landy": True,
        "Namyats": False,
        "Ogust": False,
        "Polish two suiters": False,
        "Quantitative 4NT": True,
        "Raptor 1NT": False,
        "Responsive double": True,
        "Reverse Bergen": False,
        "Reverse drury": False,
        "Reverse style": False,
        "ROPI DOPI": True,
        "Rubensohl after 1NT": False,
        "Rubensohl after 1m": False,
        "Rubensohl after double": False,
        "Semi forcing 1NT": False,
        "SMOLEN": True,
        "Snapdragon Double": False,
        "Soloway Jump Shifts": False,
        "Soloway Jump Shifts Extended": False,
        "Splinter": True,
        "Support 1NT": True,
        "Support double redouble": True,
        "Two suit takeout double": True,
        "Two way game tries": True,
        "Two Way New Minor Forcing": False,
        "Unusual 1NT": True,
        "Unusual 2NT": True,
        "Unusual 4NT": True,
        "Weak Jump Shifts": False,
        "Western cue bid": False,
        "Weak natural 2D": True,
        "Weak natural 2M": True,
        "Wilkosz": False,
        "Not defined": False,
        "1D opening to 18 HCP": False
    }   

    def __init__(self, ns_system, ew_system, position, hand, vuln, dealer):
        try:
           # Load the .NET assembly and import the types and classes from the assembly
            clr.AddReference(EPBot_PATH)
            from EPBot86 import EPBot
        except Exception as ex:
            # Provide a message to the user if the assembly is not found
            print("Error: Unable to load EPBot86.dll. Make sure the DLL is in the ./bin directory")
            print("Make sure the dll is not blocked by OS (Select properties and click unblock)")
            print('Error:', ex)
        self.ns_system = ns_system
        self.ew_system = ew_system
        self.vuln = vuln
        self.hand_str = hand.split('.')
        self.hand_str.reverse()
        self.player = EPBot()
        #print(f"Version: {self.player.version()}")
        self.dealer = dealer
        self.position = position
        # Set system types for NS and EW
        self.player.set_system_type(self.C_NS,ns_system)
        self.player.set_system_type(self.C_WE,ew_system)

        # This is what we play
        print(self.player.system_name(0))
        print(self.player.system_name(1))

         # Iterate through the conventions array and set conventions for a player at a specific position
        for convention, selected in self.conventions_list.items():
            if selected:
                self.player.set_conventions(self.C_NS, convention, True)
                self.player.set_conventions(self.C_WE, convention, True)

        # Set scoring type
        self.player.scoring = self.SCORING_IMP

        if vuln[0] and vuln[1]:
            self.vuln = 3
        if vuln[0] and not vuln[1]:
            self.vuln = 2
        if not vuln[0] and vuln[1]:
            self.vuln = 1
        if not vuln[0] and not vuln[1]:
            self.vuln = 0

        self.vuln = 0


    #system_type:
    #0 - 21GF;
    #1 - SAYC:
    #2 - WJ;
    #3 - PC,

    async def async_bid(self, auction):
        return self.bid(auction)

    # Define a Python function to find a bid
    def bid(self, auction):
        # Define constants for scoring
        # Send all bids to the bot
        self.player.new_hand(self.position, self.hand_str, self.dealer, self.vuln)

        for k in range(len(auction)):
            bidid = bidding.BID2ID[auction[k]]
            if bidid < 2:
                continue
            if bidid < 5:
                bidid = bidid - 2
            self.player.set_bid(k % 4, bidid)


        new_bid = self.player.get_bid()

        # Interpret the potential bid
        self.player.interpret_bid(new_bid)
        if new_bid < 5:
            new_bid += 2
        # Get information from Player(position) about the interpreted player
        meaning = self.player.get_info_meaning(self.C_INTERPRETED)
        if not meaning: meaning = ""
        info = self.player.get_info_feature(self.C_INTERPRETED)
        minhcp = info[102]
        maxhcp = info[103]
        if minhcp > 0:
            if maxhcp < 37:
                meaning += f" {minhcp}-{maxhcp} hcp"
            else:
                meaning += f" {minhcp}+ hcp"
        elif maxhcp < 37:
            meaning += f" {maxhcp}- hcp"

        print(f"Bid: {bidding.ID2BID[new_bid]}={meaning}")
        return BidResp(bid=bidding.ID2BID[new_bid], candidates=[], samples=[], shape=-1, hcp=-1, who = "BBA")

