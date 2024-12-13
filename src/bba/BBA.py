
import sys
import os
import util
from threading import Lock


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
                        util.load_dotnet_framework_assembly(EPBot_PATH, verbose)
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

    def __init__(self, ns_system, ew_system, position, hand, vuln, dealer, scoring_matchpoint, verbose):


        dll = BBABotBid.get_dll(verbose)  # Retrieve the loaded DLL classes through the singleton
        EPBot = dll["EPBot"]
        self.verbose = verbose
        # Load the .NET assembly
        if ew_system == None and ns_system == None:  
            return
        if ew_system == '-1' or ns_system == '-1':  
            print(f"{Fore.RED}Error: No CC defined for BBA{Fore.RESET}")
            sys.exit(1)
        self.ns_system = ns_system
        self.ew_system = ew_system
        self.ns = -1
        self.ew = -1
        self.vuln = vuln
        self.hand_str = hand.split('.')
        self.hand_str.reverse()
        self.player = EPBot()
        if self.verbose:
            print(f"BBA Version (DLL): {self.player.version()}")
        self.dealer = dealer
        self.position = position
        self.conventions_ns, self.conventions_ew = self.load_ccs()
        # Set system types for NS and EW
        self.player.set_system_type(self.C_NS,int(self.ns))
        self.player.set_system_type(self.C_WE,int(self.ew))
        if self.verbose:
            # This is what we play
            print("System NS:", self.player.system_name(0))
            print("System EW:", self.player.system_name(1))

         # Iterate through the conventions array and set conventions for a player at a specific position
        for convention, selected in self.conventions_ns.items():
            if selected:
                self.player.set_conventions(self.C_NS, convention, True)

         # Iterate through the conventions array and set conventions for a player at a specific position
        for convention, selected in self.conventions_ew.items():
            if selected:
                self.player.set_conventions(self.C_WE, convention, True)

        # Set scoring type
        if scoring_matchpoint == True:
            self.player.scoring = self.SCORING_MATCH_POINTS
        else:
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

    async def async_bid(self, auction, alert=None):
        return self.bid(auction)

    def load_ccs(self):
        # Initialize the dictionary to store the conventions
        conventions_ew = {}

        # Open the file and process each line
        with open(self.ew_system, 'r') as file:
            for i, line in enumerate(file):
                # Split the line into key and value
                key, value = line.strip().split(' = ')
                # Special case for the first line (System type)
                if i == 0 and key == "System type":
                    cc = int(value)  # Store the value as an integer
                    self.ew = cc
                else:
                    # Convert other values to boolean (1 -> True, 0 -> False)
                    conventions_ew[key] = bool(int(value))

        conventions_ns = {}

        # Open the file and process each line
        with open(self.ns_system, 'r') as file:
            for i, line in enumerate(file):
                # Split the line into key and value
                key, value = line.strip().split(' = ')
                
                # Special case for the first line (System type)
                if i == 0 and key == "System type":
                    cc = int(value)  # Store the value as an integer
                    self.ns = cc
                else:
                    # Convert other values to boolean (1 -> True, 0 -> False)
                    conventions_ns[key] = bool(int(value))
        return conventions_ns, conventions_ew


    def is_key_card_ask(self, auction):
        # Did partner ask for keycards
        if len(auction) > 1:
            if auction[-2] == "4N":
                explanation = self.explain(auction[:-1])
                if self.verbose:
                    print(explanation)
                if "Blackwood" in explanation:
                    return self.bid(auction)
        return None
        
    def explain(self, auction):
        #print("new_hand", self.position, self.hand_str, self.dealer, self.vuln)
        self.player.new_hand(self.position, self.hand_str, self.dealer, self.vuln)

        for k in range(len(auction)-1):
            bidid = bidding.BID2ID[auction[k]]
            if bidid < 2:
                continue
            if bidid < 5:
                bidid = bidid - 2
            #print("set_bid",((k) % 4, bidid))
            self.player.set_bid((k) % 4, bidid)

        #new_bid = self.player.get_bid()
        #print("get_bid",new_bid)

        #self.player.interpret_bid(new_bid)
        # Get information from Player(position) about the interpreted player
        #meaning = self.player.get_info_meaning(self.C_INTERPRETED)
        #print("interpret_bid",meaning)
        lastbid = bidding.BID2ID[auction[-1]]
        if lastbid < 5:
            lastbid = lastbid - 2
        #print("interpret_bid",lastbid)
        # Interpret the potential bid
        self.player.interpret_bid(lastbid)
        # Get information from Player(position) about the interpreted player
        meaning = self.player.get_info_meaning(self.C_INTERPRETED)
        if meaning is None: meaning = ""
        info = self.player.get_info_feature(self.C_INTERPRETED)
        minhcp = info[102]
        maxhcp = info[103]
        forcing = info[112]
        alert = info[144] == 1
        if minhcp > 0:
            if maxhcp < 37:
                meaning += f" ({minhcp}-{maxhcp} hcp)"
            else:
                meaning += f" ({minhcp}+ hcp)"
        elif maxhcp < 37:
            meaning += f" ({maxhcp}- hcp)"
        #print(f"Bid: {auction[-1]}={meaning} {'*' if alert else ''}")
        return f"{meaning}"

    # Define a Python function to find a bid
    def bid(self, auction):
        # Send all bids to the bot
        if self.verbose:
            print("new_hand", self.position, self.hand_str, self.dealer, self.vuln)
        self.player.new_hand(self.position, self.hand_str, self.dealer, self.vuln)

        for k in range(len(auction)):
            bidid = bidding.BID2ID[auction[k]]
            if bidid < 2:
                continue
            if bidid < 5:
                bidid = bidid - 2
            #print("set_bid",(k % 4, bidid))
            self.player.set_bid(k % 4, bidid)


        #print("get_bid()")
        new_bid = self.player.get_bid()

        # Interpret the potential bid
        #print("interpret_bid",new_bid)
        self.player.interpret_bid(new_bid)
        if new_bid < 5:
            new_bid += 2
        # Get information from Player(position) about the interpreted player
        meaning = self.player.get_info_meaning(self.C_INTERPRETED)
        info = self.player.get_info_feature(self.C_INTERPRETED)
        alert = info[144] == 1
        if  meaning is not None: 
            minhcp = info[102]
            maxhcp = info[103]
            if minhcp > 0:
                if maxhcp < 37:
                    meaning += f" ({minhcp}-{maxhcp} hcp)"
                else:
                    meaning += f" ({minhcp}+ hcp)"
            elif maxhcp < 37:
                meaning += f" ({maxhcp}- hcp)"

        if self.verbose:
            print(f"Bid: {bidding.ID2BID[new_bid]}={meaning}")

        return BidResp(bid=bidding.ID2BID[new_bid], candidates=[], samples=[], shape=-1, hcp=-1, who = "BBA", quality=None, alert = alert, explanation=meaning)

