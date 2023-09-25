
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

# Load the .NET assembly
clr.AddReference(EPBot_PATH)

# Import the types and classes from the assembly
from EPBot86 import EPBot

#The error you're encountering suggests that the info_meaning property is not directly accessible as an attribute in the EPBot class. Instead, you should use the get_info_meaning and set_info_meaning methods that are automatically generated for properties with a getter and setter in .NET.


class BBABotBid: 

    # Define constants for system types and conventions (replace these with actual values)
    C_NS = 0
    C_WE = 1
    C_INTERPRETED = 13

    SCORING_MATCH_POINTS = 0
    SCORING_IMP = 1

    # Define your conventions in an array
    conventions_list = [
        (4, "1M-3M inviting"),
        (6, "5NT pick a slam"),
        (9, "Ask for aces with kings"),
        (14, "Chekback"),
        (15, "Cue bid"),
        (16, "Drury"),
        (18, "Fit showing jumps"),
        (20, "Fourth suit"),
        (25, "Inverted minors"),
        (26, "Jacoby 2NT"),
        (29, "Lebensohl after 1NT"),
        (31, "Lebensohl after double"),
        (32, "Maximal Doubles"),
        (33, "Michaels Cuebid"),
        (34, "Mini Splinter"),
        (35, "Minor Suit Stayman after 1NT"),
        (36, "Minor Suit Stayman after 2NT"),
        (39, "Mixed raise"),
        (41, "Multi-Landy"),
        (44, "Quantitative 4NT"),
        (46, "Responsive double"),
        (48, "Reverse style"),
        (50, "RKC Gerber"),
        (51, "RKC Blackwood"),
        (59, "SMOLEN"),
        (61, "Splinter"),
        (62, "Support 1NT"),
        (63, "Support 2NT"),
        (64, "Support double redouble"),
        (65, "Two suit takeout double"),
        (66, "Two way game tries"),
        (67, "Two Way New Minor Forcing"),
        (68, "Unusual 1NT"),
        (69, "Unusual 2NT"),
        (70, "Unusual 4NT"),
        (72, "Weak natural 2D"),
        (73, "Weak natural 2M")
    ]

    def __init__(self, ns_system, ew_system, position, hand, vuln, dealer):
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
        #for conv_number, conv_desc in self.conventions_list:
        #    self.player.set_conventions(self.C_NS, conv_desc, True)
        #    self.player.set_conventions(self.C_WE, conv_desc, True)

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

        #print("Vul: ", self.vuln)
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
        #print(meaning)
        #print(minhcp)
        #print(maxhcp)
        if minhcp > 0:
            if maxhcp < 37:
                meaning += f" {minhcp}-{maxhcp} hcp"
            else:
                meaning += f" {minhcp}+ hcp"
        elif maxhcp < 37:
            meaning += f" {maxhcp}- hcp"

        print(f"Bid: {bidding.ID2BID[new_bid]}={meaning}")
        return BidResp(bid=bidding.ID2BID[new_bid], candidates=[], samples=[])

