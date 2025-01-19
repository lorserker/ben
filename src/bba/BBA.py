
import sys
import os
from util import calculate_seed, load_dotnet_framework_assembly
from threading import Lock
import deck52
import numpy as np


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Calculate the parent directory
parent_dir = os.path.join(script_dir, "../..")
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from src.objects import BidResp
from bidding import bidding
from colorama import Fore, Back, Style, init

init()

if "src" in script_dir and "bba" in script_dir: 
    # We are running inside the src/pimc directory
    BIN_FOLDER = parent_dir + os.path.sep + 'bin'
else:

    BEN_HOME = os.getenv('BEN_HOME')
    if BEN_HOME == '.':
        BIN_FOLDER = 'bin'
    else:
        BIN_FOLDER = os.path.join(BEN_HOME, 'bin')

if sys.platform == 'win32':
    EPBot_LIB = 'EPBot86'
elif sys.platform == 'darwin':
    EPBot_LIB = 'N/A'
else:
    EPBot_LIB = 'N/A'

EPBot_PATH = os.path.join(BIN_FOLDER, EPBot_LIB)


class BBABotBid: 

    _dll_loaded = None  # Class-level attribute to store the DLL singleton
    _lock = Lock()      # Lock to ensure thread-safe initialization

    @classmethod
    def get_dll(cls, verbose = False):
        """Access the loaded DLL classes."""
        if cls._dll_loaded is None:
            with cls._lock:  # Ensure only one thread can enter this block at a time
                if cls._dll_loaded is None:  # Double-checked locking
                    try:
                        load_dotnet_framework_assembly(EPBot_PATH, verbose)
                        from EPBot86 import EPBot
                        # Load the .NET assembly and import the types and classes from the assembly
                        if verbose:
                            print(f"EPBot Version (DLL): {EPBot().version()}")
                        cls._dll_loaded = {
                            "EPBot": EPBot,
                        }
                    except Exception as ex:
                        # Provide a message to the user if the assembly is not found
                        print(f"{Fore.RED}Error: Unable to load EPBot86.dll. Make sure the DLL is in the ./bin directory")
                        print("Make sure the dll is not blocked by OS (Select properties and click unblock)")
                        print(f"Make sure the dll is not writeprotected{Fore.RESET}")
                        print('Error:', ex)
                        sys.exit(1)
        return cls._dll_loaded
    

    # Define constants for system types and conventions 
    C_NS = 0
    C_WE = 1
    C_INTERPRETED = 13

    SCORING_MATCH_POINTS = 0
    SCORING_IMP = 1


    def __init__(self, our_system_file, their_system_file, position, hand, vuln, dealer, scoring_matchpoint, verbose):


        dll = BBABotBid.get_dll(verbose)  # Retrieve the loaded DLL classes through the singleton
        EPBot = dll["EPBot"]
        self.verbose = verbose
        # We just needed to Load the .NET assembly
        if position == None:  
            return
        assert our_system_file is not None, "Our system file is not set"
        assert their_system_file is not None, "Their system file is not set"
        self.our_system_file = our_system_file
        self.their_system_file = their_system_file
        self.our_system = -1
        self.their_system = -1
        self.vuln_nsew = vuln
        self.hand_str = hand.split('.')
        self.hand_str.reverse()
        self.hash_integer = calculate_seed(hand)         
        self.rng = self.get_random_generator()

        # Initialize 4 players
        self.players = [EPBot() for _ in range(4)]
        if self.verbose:
            print(f"BBA Version (DLL): {self.players[0].version()}")
        self.dealer = dealer
        self.position = position

        self.our_conventions, self.their_conventions = self.load_ccs()
        # Position is always N=0, E=1, S=2, W=3
        # Set system types for each player
        for position in range(4):
            if position % 2 == 0:  # N (0) and S (2)
                we = self.C_NS
                they = self.C_WE
                self.vuln_wethey = self.vuln_nsew
            else:  # E (1) and W (3)
                we = self.C_WE
                they = self.C_NS
                self.vuln_wethey = [self.vuln_nsew[1], self.vuln_nsew[0]]

            # Set the system type for each player
            self.players[position].set_system_type(we, int(self.our_system))
            self.players[position].set_system_type(they, int(self.their_system))
            if self.verbose:
                # This is what we play
                print(f"Our system: {self.our_system_file}")
                print(f"Their system: {self.their_system_file}")
                print("Our System   :", self.players[position].system_name(we))
                print("Their System :", self.players[position].system_name(they))

            # Iterate through the conventions array and set conventions for a player at a specific position
            for convention, selected in self.our_conventions.items():
                self.players[position].set_conventions(we, convention, selected)

            # Iterate through the conventions array and set conventions for a player at a specific position
            for convention, selected in self.their_conventions.items():
                self.players[position].set_conventions(they, convention, selected)

            # Set scoring type
            if scoring_matchpoint == True:
                self.players[position].scoring = self.SCORING_MATCH_POINTS
            else:
                self.players[position].scoring = self.SCORING_IMP

    def bba_vul(self, vuln):
        return vuln[1] * 2 + vuln[0]

    def get_random_generator(self):
        #print(f"{Fore.BLUE}Fetching random generator for bid {self.hash_integer}{Style.RESET_ALL}")
        return np.random.default_rng(self.hash_integer)


    async def async_bid(self, auction, alert=None):
        return self.bid(auction)

    def load_ccs(self):
        # Initialize the dictionary to store the conventions
        their_conventions = {}

        try:
            # Open the file and process each line
            with open(self.their_system_file, 'r') as file:
                for i, line in enumerate(file):
                    # Split the line into key and value
                    key, value = line.strip().split(' = ')
                    # Special case for the first line (System type)
                    if i == 0 and key == "System type":
                        cc = int(value)  # Store the value as an integer
                        self.their_system = cc
                    else:
                        # Convert other values to boolean (1 -> True, 0 -> False)
                        their_conventions[key] = bool(int(value))
        except:
            print(f"{Fore.RED}Error: Unable to load {self.their_system_file}{Style.RESET_ALL}")
            sys.exit(1)

        our_conventions = {}

        try:
            # Open the file and process each line
            with open(self.our_system_file, 'r') as file:
                for i, line in enumerate(file):
                    # Split the line into key and value
                    key, value = line.strip().split(' = ')
                    
                    # Special case for the first line (System type)
                    if i == 0 and key == "System type":
                        cc = int(value)  # Store the value as an integer
                        self.our_system = cc
                    else:
                        # Convert other values to boolean (1 -> True, 0 -> False)
                        our_conventions[key] = bool(int(value))
        except:
            print(f"{Fore.RED}Error: Unable to load {self.their_system_file}{Style.RESET_ALL}")
            sys.exit(1)

        return our_conventions, their_conventions


    def is_key_card_ask(self, auction):
        # Did partner ask for keycards
        if len(auction) > 1:
            if auction[-2] == "4N":
                explanation, alert = self.explain(auction[:-1])
                if self.verbose:
                    print(explanation, alert)
                if "Blackwood" in explanation:
                    return self.bid(auction)
            if auction[-2] == "5N":
                explanation, alert = self.explain(auction[:-1])
                if self.verbose:
                    print(explanation, alert)
                if "King ask" in explanation:
                    return self.bid(auction)
            if auction[-2] == "4C":
                explanation, alert = self.explain(auction[:-1])
                if self.verbose:
                    print(explanation, alert)
                if "Gerber" in explanation:
                    return self.bid(auction)
            if auction[-2] == "5C":
                explanation, alert = self.explain(auction[:-1])
                if self.verbose:
                    print(explanation, alert)
                if "King ask" in explanation:
                    return self.bid(auction)
        return None
        
    def explain(self, auction):
        our_seat = (len(auction) - 1) % 4
        self.players[0].new_hand(our_seat, self.hand_str, self.dealer, self.bba_vul(self.vuln_nsew))

        # We bid up to the bid we want explained
        position = self.dealer
        for k in range(len(auction)-1):
            bidid = bidding.BID2ID[auction[k]]
            if bidid < 2:
                continue
            if bidid < 5:
                bidid = bidid - 2
            self.players[0].set_bid((position) % 4, bidid)
            position += 1

        lastbid = bidding.BID2ID[auction[-1]]
        if lastbid < 5:
            lastbid = lastbid - 2

        # Now ask for the bid we want explained
        self.players[0].interpret_bid(lastbid)
        # Get information from Player(position) about the interpreted bid
        meaning = self.players[0].get_info_meaning(self.C_INTERPRETED)
        if meaning is None: meaning = ""
        if meaning.strip() == "calculated bid": meaning = ""
        if meaning.strip() == "bidable suit": meaning = ""
        bba_alert = self.players[0].get_info_alerting(self.C_INTERPRETED)

        info = self.players[0].get_info_feature(self.C_INTERPRETED)
        if not bba_alert and meaning != "":   
            minhcp = info[402]
            maxhcp = info[403]
            if minhcp > 0:
                if maxhcp < 37:
                    meaning += f" ({minhcp}-{maxhcp} hcp)"
                else:
                    meaning += f" ({minhcp}+ hcp)"
            elif maxhcp < 37:
                meaning += f" ({maxhcp}- hcp)"

        return f"{meaning}", bba_alert

    # Define a Python function to find a bid
    def bid(self, auction):
        # Send all bids to the bot
        # We are to make a bid, so we can use the position
        if self.verbose:
            print(auction)
            print("new_hand", self.position, self.hand_str, self.dealer, self.vuln_nsew)
        self.players[self.position].new_hand(self.position, self.hand_str, self.dealer, self.bba_vul(self.vuln_nsew))

        position = self.dealer
        for k in range(len(auction)):
            bidid = bidding.BID2ID[auction[k]]
            if bidid < 2:
                continue
            if bidid < 5:
                bidid = bidid - 2
            self.players[self.position].set_bid((position) % 4, bidid)
            position += 1


        #print("get_bid()")
        new_bid = self.players[self.position].get_bid()

        # Interpret the potential bid
        #print("interpret_bid",new_bid)
        self.players[self.position].interpret_bid(new_bid)
        if new_bid < 5:
            new_bid += 2

        # Get information from Player(position) about the interpreted player
        meaning = self.players[self.position].get_info_meaning(self.C_INTERPRETED)
        if meaning is None: meaning = ""
        if meaning.strip() == "calculated bid": meaning = ""
        if meaning.strip() == "bidable suit": meaning = ""
        bba_alert = self.players[self.position].get_info_alerting(self.C_INTERPRETED)

        info = self.players[self.position].get_info_feature(self.C_INTERPRETED)

        if not bba_alert and meaning != "":   
            minhcp = info[402]
            maxhcp = info[403]
            if minhcp > 0:
                if maxhcp < 37:
                    meaning += f" ({minhcp}-{maxhcp} hcp)"
                else:
                    meaning += f" ({minhcp}+ hcp)"
            elif maxhcp < 37:
                meaning += f" ({maxhcp}- hcp)"

        if self.verbose:
            print(f"BBABid: {bidding.ID2BID[new_bid]}={meaning}")

        return BidResp(bid=bidding.ID2BID[new_bid], candidates=[], samples=[], shape=-1, hcp=-1, who = "BBA", quality=None, alert = bba_alert, explanation=meaning)

    def bid_hand(self, auction, deal):
        #print(auction, deal)
        # To get deterministic result the hand is always North
        position = 0
        dealer = ((self.dealer - self.position) + 4) % 4 

        hands = deal.split(":")[1].split(' ') 
        bba_auction = auction.copy()
        #print("dealer", self.dealer)
        for i in range(4):
            hand_str = hands[i].split('.')
            hand_str.reverse()
            # The deal we get is always our hand first
            self.players[i].new_hand(i, hand_str, dealer, self.bba_vul(self.vuln_wethey))

        # Update bidding until now
        passes = 0
        position = dealer
        for k in range(len(auction)):
            bidid = bidding.BID2ID[auction[k]]
            if bidid < 2:
                continue
            if bidid < 5:
                bidid = bidid - 2

            for i in range(4):
                self.players[(i) % 4].set_bid(position % 4, bidid)
            if bidid == 0:
                passes += 1
            else:
                passes = 0
            position += 1
        
        # Now bid the hand to the end
        # Always LHO"
        position = 1
        while passes < 3:

            new_bid = self.players[position].get_bid()
            for i in range(4):
                self.players[i].set_bid(position, new_bid)
            if new_bid == 0:
                passes += 1
            else:
                passes = 0
            if new_bid < 5:
                new_bid += 2
            bba_auction.append(bidding.ID2BID[new_bid])

            position = (position + 1) % 4
        
        return bba_auction

