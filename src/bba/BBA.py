
import sys
import os
from util import calculate_seed
from threading import Lock
import numpy as np
import platform
import ctypes
from ctypes import c_void_p, c_int, c_char_p, c_bool


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

BBA_BIN = os.getenv('BBA_BIN')
if BBA_BIN:
    BIN_FOLDER = BBA_BIN
elif "src" in script_dir and "bba" in script_dir:
    # We are running inside the src/bba directory
    BIN_FOLDER = parent_dir + os.path.sep + 'bin'
else:

    BEN_HOME = os.getenv('BEN_HOME') or '.'
    if BEN_HOME == '.':
        BIN_FOLDER = 'bin'
    else:
        BIN_FOLDER = os.path.join(BEN_HOME, 'bin')

# Determine the native EPBot library path based on platform and architecture.
# BEN always uses the native (NativeAOT) EPBot build via ctypes — there is no
# longer a .NET fallback (EPBot64/EPBot86/coreclr have been removed).
machine = platform.machine().lower()
if sys.platform == 'win32':
    _native_subdir = 'windows'
    _native_arch = 'arm64' if machine in ('arm64', 'aarch64') else 'x64'
    _native_lib_name = 'EPBot.dll'
elif sys.platform == 'darwin':
    _native_subdir = 'macos'
    _native_arch = 'arm64'
    _native_lib_name = 'libEPBot.dylib'
else:
    _native_subdir = 'linux'
    _native_arch = 'arm64' if machine in ('arm64', 'aarch64') else 'x64'
    _native_lib_name = 'libEPBot.so'

EPBot_NATIVE_PATH = os.path.join(BIN_FOLDER, 'BBA', _native_subdir, _native_arch, _native_lib_name)


class EPBotNative:
    """Wrapper around the native EPBot C library (ctypes), exposing the same
    interface as the .NET EPBot class.

    The native library is a NativeAOT build of EPBotNET and uses a
    buffer-out-param ABI (recovered from the EPBotFFI shim that ships with the
    wasm build):
      * every entry point returns an int status: 0 = OK, -1 = null handle,
        -2 = exception (call epbot_get_last_error for the message),
        -3 = supplied buffer too small;
      * value getters (version, get_bid, get_info_alerting, ...) return the
        value directly (always >= 0 in practice);
      * string results are written NUL-terminated UTF-8 into a caller buffer;
      * int-array results are written into a caller int32 buffer plus a
        count out-param;
      * string-array params and results are passed as a single
        "\n"-joined UTF-8 string (plus a count out-param when returned).
    """

    OK = 0
    ERR_BUFFER_TOO_SMALL = -3
    _DEFAULT_STR_BUF = 1 << 14

    _dll = None
    _dll_lock = Lock()

    @classmethod
    def _load_dll(cls):
        if cls._dll is None:
            with cls._dll_lock:
                if cls._dll is None:
                    dll = ctypes.CDLL(EPBot_NATIVE_PATH)
                    p_int = ctypes.POINTER(c_int)

                    def _sig(name, restype, argtypes):
                        fn = getattr(dll, name)
                        fn.restype = restype
                        fn.argtypes = argtypes

                    _sig('epbot_create', c_void_p, [])
                    _sig('epbot_destroy', None, [c_void_p])
                    _sig('epbot_get_last_error', c_char_p, [])
                    _sig('epbot_version', c_int, [c_void_p])
                    # new_hand(instance, position, hand("\n"-joined suits), dealer, vuln, repeating, b_playing)
                    _sig('epbot_new_hand', c_int, [c_void_p, c_int, c_char_p, c_int, c_int, c_int, c_int])
                    _sig('epbot_set_system_type', c_int, [c_void_p, c_int, c_int])
                    _sig('epbot_set_conventions', c_int, [c_void_p, c_int, c_char_p, c_int])
                    _sig('epbot_system_name', c_int, [c_void_p, c_int, c_char_p, c_int])
                    _sig('epbot_get_scoring', c_int, [c_void_p])
                    _sig('epbot_set_scoring', c_int, [c_void_p, c_int])
                    # set_bid(instance, position, bidid, alert_text)
                    _sig('epbot_set_bid', c_int, [c_void_p, c_int, c_int, c_char_p])
                    _sig('epbot_get_bid', c_int, [c_void_p])
                    _sig('epbot_interpret_bid', c_int, [c_void_p, c_int])
                    _sig('epbot_get_info_meaning', c_int, [c_void_p, c_int, c_char_p, c_int])
                    _sig('epbot_get_info_alerting', c_int, [c_void_p, c_int])
                    for _name in ('feature', 'min_length', 'max_length', 'probable_length',
                                  'strength', 'stoppers', 'honors', 'suit_power'):
                        _sig('epbot_get_info_' + _name, c_int, [c_void_p, c_int, c_void_p, c_int, p_int])
                    # set_arr_bids(instance, bids("\n"-joined))
                    _sig('epbot_set_arr_bids', c_int, [c_void_p, c_char_p])
                    # get_arr_suits(instance, current_longers, buffer, bufsize, countOut)
                    _sig('epbot_get_arr_suits', c_int, [c_void_p, c_int, c_char_p, c_int, p_int])
                    _sig('epbot_get_str_bidding', c_int, [c_void_p, c_char_p, c_int])

                    cls._dll = dll
        return cls._dll

    # -- low-level helpers ---------------------------------------------------

    @classmethod
    def _last_error(cls):
        try:
            msg = cls._dll.epbot_get_last_error()
        except Exception:
            return ""
        return msg.decode('utf-8', 'replace') if msg else ""

    def _check(self, ret, what):
        if ret is not None and ret < 0:
            raise RuntimeError("EPBot %s failed (%d): %s" % (what, ret, self._last_error()))
        return ret

    def _str_call(self, what, fn, *args):
        size = self._DEFAULT_STR_BUF
        while True:
            buf = ctypes.create_string_buffer(size)
            ret = fn(self._handle, *args, buf, size)
            if ret == self.ERR_BUFFER_TOO_SMALL:
                size *= 4
                continue
            self._check(ret, what)
            return buf.value.decode('utf-8', 'replace')

    def _int_array_call(self, what, fn, *args, count_hint):
        size = max(count_hint, 8)
        while True:
            arr = (c_int * size)()
            count = c_int(0)
            ret = fn(self._handle, *args, arr, size * 4, ctypes.byref(count))
            if ret == self.ERR_BUFFER_TOO_SMALL:
                size *= 4
                continue
            self._check(ret, what)
            return [arr[i] for i in range(count.value)]

    def _str_array_call(self, what, fn, *args, count_hint):
        size = self._DEFAULT_STR_BUF
        while True:
            buf = ctypes.create_string_buffer(size)
            count = c_int(0)
            ret = fn(self._handle, *args, buf, size, ctypes.byref(count))
            if ret == self.ERR_BUFFER_TOO_SMALL:
                size *= 4
                continue
            self._check(ret, what)
            n = count.value
            if n == 0:
                return []
            parts = buf.value.decode('utf-8', 'replace').split('\n')
            if len(parts) < n:
                parts += [''] * (n - len(parts))
            return parts[:n]

    @staticmethod
    def _enc(value):
        if value is None:
            return None
        return value.encode('utf-8') if isinstance(value, str) else value

    @staticmethod
    def _join(seq):
        return '\n'.join(s if isinstance(s, str) else s.decode('utf-8') for s in seq).encode('utf-8')

    def __init__(self):
        dll = EPBotNative._load_dll()
        self._dll = dll
        handle = dll.epbot_create()
        if not handle:
            raise RuntimeError("epbot_create failed: %s" % EPBotNative._last_error())
        self._handle = handle

    def __del__(self):
        if getattr(self, '_handle', None) and getattr(self, '_dll', None):
            try:
                self._dll.epbot_destroy(self._handle)
            except Exception:
                pass
            self._handle = None

    def version(self):
        return self._check(self._dll.epbot_version(self._handle), 'version')

    def new_hand(self, position, hand_str, dealer, vuln):
        self._check(self._dll.epbot_new_hand(self._handle, position, self._join(hand_str), dealer, vuln, 0, 0), 'new_hand')

    def set_system_type(self, we_they, system_id):
        self._check(self._dll.epbot_set_system_type(self._handle, we_they, int(system_id)), 'set_system_type')

    def set_conventions(self, we_they, convention_name, selected):
        self._check(self._dll.epbot_set_conventions(self._handle, we_they, self._enc(convention_name), 1 if selected else 0), 'set_conventions')

    def system_name(self, we_they):
        return self._str_call('system_name', self._dll.epbot_system_name, we_they)

    @property
    def scoring(self):
        return self._check(self._dll.epbot_get_scoring(self._handle), 'get_scoring')

    @scoring.setter
    def scoring(self, value):
        self._check(self._dll.epbot_set_scoring(self._handle, value), 'set_scoring')

    def set_bid(self, position, bidid):
        self._check(self._dll.epbot_set_bid(self._handle, position, bidid, b''), 'set_bid')

    def get_bid(self):
        return self._check(self._dll.epbot_get_bid(self._handle), 'get_bid')

    def interpret_bid(self, bid):
        self._check(self._dll.epbot_interpret_bid(self._handle, bid), 'interpret_bid')

    def get_info_meaning(self, position):
        meaning = self._str_call('get_info_meaning', self._dll.epbot_get_info_meaning, position)
        return meaning if meaning else None

    def get_info_alerting(self, position):
        # The .NET EPBot exposes this as a bool; mirror that (the FFI returns 0/1).
        return bool(self._check(self._dll.epbot_get_info_alerting(self._handle, position), 'get_info_alerting'))

    def get_info_feature(self, position):
        feature = self._int_array_call('get_info_feature', self._dll.epbot_get_info_feature, position, count_hint=512)
        if len(feature) < 512:
            feature += [0] * (512 - len(feature))
        return feature

    def get_info_min_length(self, position):
        return self._int_array_call('get_info_min_length', self._dll.epbot_get_info_min_length, position, count_hint=4)

    def get_info_max_length(self, position):
        return self._int_array_call('get_info_max_length', self._dll.epbot_get_info_max_length, position, count_hint=4)

    def get_info_probable_length(self, position):
        return self._int_array_call('get_info_probable_length', self._dll.epbot_get_info_probable_length, position, count_hint=4)

    def get_info_strength(self, position):
        return self._int_array_call('get_info_strength', self._dll.epbot_get_info_strength, position, count_hint=4)

    def get_info_stoppers(self, position):
        return self._int_array_call('get_info_stoppers', self._dll.epbot_get_info_stoppers, position, count_hint=4)

    def get_info_honors(self, position):
        return self._int_array_call('get_info_honors', self._dll.epbot_get_info_honors, position, count_hint=4)

    def get_info_suit_power(self, position):
        return self._int_array_call('get_info_suit_power', self._dll.epbot_get_info_suit_power, position, count_hint=4)

    def set_arr_bids(self, arr_bids):
        self._check(self._dll.epbot_set_arr_bids(self._handle, self._join(arr_bids)), 'set_arr_bids')

    def get_arr_suits(self):
        return self._str_array_call('get_arr_suits', self._dll.epbot_get_arr_suits, 0, count_hint=16)

    def get_str_bidding(self):
        return self._str_call('get_str_bidding', self._dll.epbot_get_str_bidding)


def _str_array(lst):
    """Thin alias kept for call-site readability — the native EPBot API takes plain Python lists."""
    return lst


class BBABotBid: 

    _dll_loaded = None  # Class-level attribute to store the DLL singleton
    _lock = Lock()      # Lock to ensure thread-safe initialization

    @classmethod
    def get_dll(cls, verbose = False):
        """Load the native EPBot library (ctypes) once and cache it."""
        if cls._dll_loaded is None:
            with cls._lock:  # Ensure only one thread can enter this block at a time
                if cls._dll_loaded is None:  # Double-checked locking
                    try:
                        EPBot = EPBotNative
                        if verbose:
                            print(f"Loading native EPBot from {EPBot_NATIVE_PATH}")
                            print(f"EPBot Version (DLL): {EPBot().version()}")
                        cls._dll_loaded = {
                            "EPBot": EPBot,
                        }
                    except Exception as ex:
                        print(f"{Fore.RED}Error: Unable to load EPBot from {EPBot_NATIVE_PATH}")
                        print("Make sure the library exists and is not blocked by the OS (Properties > Unblock)")
                        print(f"Make sure the library is not write-protected{Fore.RESET}")
                        print('Error:', ex)
                        sys.exit(1)
        return cls._dll_loaded
    

    # Define constants for system types and conventions 
    C_NS = 0
    C_WE = 1
    C_INTERPRETED = 13

    SCORING_MATCH_POINTS = 0
    SCORING_IMP = 1
    suitsymbols = ["!C", "!D", "!H", "!S"]   

    # After getting this explanation we leave to BBA to continue the bidding
    bba_controling = {
            "4N": "Blackwood",
            "5N": "King ask",
            "4C": "Gerber",
            "5C": "King ask",
            "4S": "Exclusion",
            "5C": "Exclusion",
            "5D": "Exclusion",            
            "5H": "Exclusion",            
            "5S": "Exclusion"            
        }       


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
        assert len(hand) == 16, "Hand must have 13 cards and each suit delimited by ."
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

    def version(self):
        dll = BBABotBid.get_dll()  # Retrieve the loaded DLL classes through the singleton
        EPBot = dll["EPBot"]
        return EPBot().version()

    def bba_vul(self, vuln):
        return vuln[1] + vuln[0] * 2

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


    def is_key_card_ask(self, auction, explanation):
        if len(auction) > 1:
            # We let BBA answer the question
            if auction[-2] in self.bba_controling and self.bba_controling[auction[-2]] in explanation:
                return True
    
        return False

    def find_info(self, auction):
        if self.verbose:
            print("Searching info for this auction: ", auction)
            print("new_hand", self.position, self.hand_str, self.dealer, self.bba_vul(self.vuln_nsew))

        self.players[self.position].new_hand(self.position, _str_array(self.hand_str), self.dealer, self.bba_vul(self.vuln_nsew))

        arr_bids = []

        for k in range(len(auction)):
            bidid = bidding.BID2ID[auction[k]]
            if bidid < 2:
                continue
            if bidid < 5:
                bidid = bidid - 2
            arr_bids.append(f"{bidid:02}")

        info = {}
        info["position"] = self.position
        info["hand"] = self.hand_str
        info["dealer"] = self.dealer
        info["vuln"] = self.bba_vul(self.vuln_nsew)
        info["arr_bids"] = arr_bids.copy()
        info["auction"] = auction
        # Extend with empty strings until length is 64
        if self.verbose:
            print("Bids sent to BBA", arr_bids)
        arr_bids.extend([''] * (64 - len(arr_bids)))
        self.players[self.position].set_arr_bids(_str_array(arr_bids))

        trump = 4
        info["trump"] = trump
        # Do we have a trump?
        for i in range(4):
            features = self.players[self.position].get_info_feature(i)
            asking_bid = features[425]
            if asking_bid > 0:
                asker = i % 4
                trump = features[424]
                info["asking_bid"] = asking_bid
                info["asker"] = asker
                info["trump"] = trump

        for i in range(8):
            hand_info = {}
            hand_info["player"] = i
            features = self.players[self.position].get_info_feature(i)
            hand_info["HCP"] =  F"{features[402]:02} - {features[403]:02}"
            min_lengths = self.players[self.position].get_info_min_length(i)
            max_lengths = self.players[self.position].get_info_max_length(i)
            probable_lengths = self.players[self.position].get_info_probable_length(i)
            strengths = self.players[self.position].get_info_strength(i)
            stoppers = self.players[self.position].get_info_stoppers(i)
            for j in range(3, -1, -1):
                suit = "CDHS"[j]
                hand_info[suit] = {
                    "length": f"{min_lengths[j]} - {max_lengths[j]}",
                    "probable_length": f"{probable_lengths[j]}",
                    "stoppers": f"{stoppers[j]}",
                    "strengths": f"{strengths[j]}",
                }
                asking_bid = features[425]
            if trump != 4:
                honors = self.players[self.position].get_info_honors(i)
                hand_info["honors"] =  honors[trump]
            hand_info["aces"] = features[406]
            hand_info["kings"] = features[407]
            # BBA has switched the meaning of -1 and 0
            hand_info["queen"] = features[319]
            #print(hand_info)
            info[i] = hand_info

        return info
        
    def find_aces(self, auction):
        if self.verbose:
            print("Searching aces for this auction: ", auction)
            print("new_hand", self.position, self.hand_str, self.dealer, self.bba_vul(self.vuln_nsew))

        info = self.find_info(auction)
        # print("info", info)
        # BBA use C->S
        trump = 3 - info["trump"] 
        result = {}
        if info["trump"] == 4:  
            return result
        # Asker is always based on seating 0=N, 3=W
        asker = info["asker"]
        # 0 = LHO, 1 = Partner, 2 = RHO
        # We do not know anything about the asker, but we should prepare for a 5N showing all keycards
        # Currently BBA assume a number of aces for asker, but this is not used
        lho = (self.position + 1) % 4
        rho = (self.position + 3) % 4
        partner = (self.position + 2) % 4 

        if asker == rho:
            result["RHO"] = (trump, -1, -1, -1)
            result["Partner"] = (trump, -1, -1, -1)
            result["LHO"] = (trump, info[lho]["aces"], info[lho]["kings"], info[partner]["queen"])
        
        elif asker == lho:
            result["LHO"] = (trump, -1, -1, -1)
            result["Partner"] = (trump, -1, -1, -1)
            result["RHO"] = (trump, info[rho]["aces"], info[rho]["kings"], info[partner]["queen"])
        # For the partner we take the calculated information
        elif asker != partner:
            # We have asked
            # if we know something about partners aces we take the calculated information
            result["LHO"] = (trump, -1, -1, -1)
            result["Partner"] = (trump, info[partner]["aces"], info[partner]["kings"], info[partner]["queen"])
            result["RHO"] = (trump, -1, -1, -1)

        if self.verbose:
            print("Information from BBA", result)

        return result

    def explain_auction(self, auction):
        if self.verbose:
            print(auction)
            print("explain_auction", self.position, self.hand_str, self.dealer, self.bba_vul(self.vuln_nsew))

        self.players[self.position].new_hand(self.position, _str_array(self.hand_str), self.dealer, self.bba_vul(self.vuln_nsew))

        meaning_of_bids = []
        bba_controlling = False
        preempted = False
        position = self.dealer
        for k in range(len(auction)):
            bidid = bidding.BID2ID[auction[k]]
            if bidid < 2:
                continue
            if bidid < 5:
                bidid = bidid - 2
            self.players[self.position].set_bid((position) % 4, bidid)
            meaning = self.players[self.position].get_info_meaning((position) % 4)
            if not meaning:
                meaning = ""
            #print(auction[k], meaning)
            meaning_of_bids.append((auction[k], meaning))
            # Are we bidding
            if (position % 2 == self.position % 2):
                if auction[k] in self.bba_controling and self.bba_controling[auction[k]] in meaning:
                    bba_controlling = True
            if (position % 2 != self.position % 2):
                lowered_meaning = meaning.lower()
                if "weak" in lowered_meaning or "preempt" in lowered_meaning:
                    preempted = True
            # We should also look for other keywaords, like splinter, void, exclusion
            position += 1
        
        return meaning_of_bids, bba_controlling, preempted

    def explain_last_bid(self, auction):
        if self.verbose:
            print(auction)
            print("explain_last_bid", self.position, self.hand_str, self.dealer, self.bba_vul(self.vuln_nsew))

        arr_bids = []
        for k in range(len(auction)):
            bidid = bidding.BID2ID[auction[k]]
            if bidid < 2:
                continue
            if bidid < 5:
                bidid = bidid - 2
            arr_bids.append(f"{bidid:02}")

        no_bids = len(arr_bids)
        position = (no_bids + self.dealer) % 4

        self.players[self.position].new_hand(position, _str_array(self.hand_str), self.dealer, self.bba_vul(self.vuln_nsew))

        arr_bids.extend([''] * (64 - len(arr_bids)))
        self.players[self.position].set_arr_bids(_str_array(arr_bids))

        # Now ask for the bid we want explained
        position = (no_bids - 1 + self.dealer) % 4
        # Get information from Player(position) about the interpreted bid
        meaning = self.players[self.position].get_info_meaning(position)
        if meaning is None: meaning = ""
        if meaning.strip() == "calculated bid": meaning = "Nat."
        if meaning:
            meaning = meaning[0].upper() + meaning[1:]
        #if meaning.strip() == "bidable suit": meaning = ""
        length = self.extract_lengths(position)
        meaning = meaning + " -- " + "; ".join(length)

        minhcp, maxhcp, gf, forcing_to, forcing, hcp = self.extract_hcp(position)
        meaning = meaning + (";" + hcp if hcp else "")

        bba_alert = self.players[self.position].get_info_alerting(position)
        if bba_alert:
            meaning += "; Artificial"
        if gf:
            meaning += "; GF"
        elif forcing:
            meaning += "; Forcing"
        elif forcing_to and forcing_to > bidid and bidid > 4:
                meaning += f"; Forcing" # to {bidding.ID2BID[forcing_to]}
        if self.verbose:
            print("explain_last_bid", meaning, bba_alert)
        return meaning, bba_alert

    # Define a Python function to find a bid
    def bid(self, auction):
        # Send all bids to the bot
        # We are to make a bid, so we can use the position
        if self.verbose:
            print(auction)
            print("new_hand", self.position, self.hand_str, self.dealer, self.bba_vul(self.vuln_nsew))
        self.players[self.position].new_hand(self.position, _str_array(self.hand_str), self.dealer, self.bba_vul(self.vuln_nsew))

        position = self.dealer
        for k in range(len(auction)):
            bidid = bidding.BID2ID[auction[k]]
            if bidid < 2:
                continue
            if bidid < 5:
                bidid = bidid - 2
            self.players[self.position].set_bid((position) % 4, bidid)
            position += 1


        new_bid = self.players[self.position].get_bid()

        # Interpret the potential bid
        self.players[self.position].interpret_bid(new_bid)
        if new_bid < 5:
            new_bid += 2

        # Get information from Player(position) about the interpreted player
        meaning = self.players[self.position].get_info_meaning(self.C_INTERPRETED)
        if meaning is None: meaning = ""
        if meaning.strip() == "calculated bid": meaning = "Nat."
        if meaning:
            meaning = meaning[0].upper() + meaning[1:]

        length = self.extract_lengths(self.C_INTERPRETED)
        meaning = meaning + " -- " + "; ".join(length)

        minhcp, maxhcp, gf, forcing_to, forcing, hcp = self.extract_hcp(self.C_INTERPRETED)
        meaning = meaning + (";" + hcp if hcp else "")

        bba_alert = self.players[self.position].get_info_alerting(self.C_INTERPRETED)
        if bba_alert:
            meaning += "; Artificial"
        if gf:
            meaning += "; GF"
        elif forcing:
            meaning += "; Forcing"
        elif forcing_to and forcing_to > new_bid and new_bid > 4:
                meaning += f"; Forcing" # to {bidding.ID2BID[forcing_to]}

        if self.verbose:
            print(f"BBABid: {bidding.ID2BID[new_bid]}={meaning}")

        return BidResp(bid=bidding.ID2BID[new_bid], candidates=[], samples=[], shape=-1, hcp=-1, who = "BBA", quality=None, alert = bba_alert, explanation=meaning)

    def get_attributes(self,value):
        mapping = {4: 'A', 3: 'K', 2: 'Q', 1: 'J'}
        attributes = []

        for points, attribute in mapping.items():
            while value >= points:
                attributes.append(attribute)
                value -= points

        return ''.join(attributes)
    
    def get_honors(self,value):
        mapping = {16: 'A', 8: 'K', 4: 'Q', 2: 'J', 1: 'T'}
        attributes = []

        for points, attribute in mapping.items():
            while value >= points:
                attributes.append(attribute)
                value -= points

        return ''.join(attributes)

    def list_bids(self, auction):
        # Send all bids to the bot
        # We are to make a bid, so we can use the position
        if self.verbose:
            print(auction)
            print("new_hand", self.position, self.hand_str, self.dealer, self.bba_vul(self.vuln_nsew))
        position = (self.dealer + len(auction)) % 4
        self.players[self.position].new_hand(position, _str_array(self.hand_str), self.dealer, self.bba_vul(self.vuln_nsew))

        position = self.dealer
        for k in range(len(auction)):
            bidid = bidding.BID2ID[auction[k]]
            if bidid < 2:
                continue
            if bidid < 5:
                bidid = bidid - 2
            self.players[self.position].set_bid((position) % 4, bidid)
            position += 1

        result = []
        for new_bid in range(40):
            if new_bid < 2:
                continue
            if not bidding.can_bid(bidding.ID2BID[new_bid], auction):
                continue
            if new_bid < 5:
                new_bid = new_bid - 2
            # Interpret the potential bid
            #print("interpret_bid",new_bid)
            # self.players[self.position].set_bid(self.position, new_bid)

            self.players[self.position].interpret_bid(new_bid)

            # Get information from Player(position) about the interpreted player
            meaning = self.players[self.position].get_info_meaning(self.C_INTERPRETED)
            if meaning is None: meaning = ""
            if meaning.strip() == "calculated bid": meaning = "Nat."
            if meaning:
                meaning = meaning[0].upper() + meaning[1:]

            length = self.extract_lengths(self.C_INTERPRETED)
            meaning = meaning + " -- " + "; ".join(length)

            minhcp, maxhcp, gf, forcing_to, forcing, hcp = self.extract_hcp(self.C_INTERPRETED)
            meaning = meaning + (";" + hcp if hcp else "")
                    
            pl = self.players[self.position].get_info_probable_length(self.C_INTERPRETED)
            pl_str = []
            for i in range(len(pl)):
                if pl[i] == 0: continue
                pl_str.append(f"probable length in {self.suitsymbols[i]} {pl[i]}")
                #print(f"probable length in {self.suitsymbols[i]}",pl[i])

            # Suit power is the same as stregngth, and basically the same as Honors
            sp = self.players[self.position].get_info_suit_power(self.C_INTERPRETED)                
            sp_str = []
            for i in range(len(sp)):
                if sp[i] == 0: continue
                sp_str.append(f"{self.get_attributes(sp[i])} in {self.suitsymbols[i]}")
                #print(f"suit power in {self.suitsymbols[i]}", i,sp[i])

            stoppers = self.players[self.position].get_info_stoppers(self.C_INTERPRETED)                
            stoppers_str = []
            for i in range(len(stoppers)):
                if stoppers[i] == 0: continue
                stoppers_str.append(f"Stopper in {self.suitsymbols[i]}")
                meaning += f"; Stopper in {self.suitsymbols[i]}"

            # Strength is the same as Suit Power
            strength = self.players[self.position].get_info_strength(self.C_INTERPRETED) 
            strength_str = []             
            for i in range(len(strength)):
                if strength[i] == 0: continue
                strength_str.append(f"{self.get_attributes(strength[i])} in {self.suitsymbols[i]}")
                #meaning += f"; {self.get_attributes(strength[i])} in {self.suitsymbols[i]}"

            honors = self.players[self.position].get_info_honors(self.C_INTERPRETED)
            honor_str = []
            for i in range(len(honors)):
                if honors[i] == 0: continue
                honor_str.append(f"{self.get_honors(honors[i])} in {self.suitsymbols[i]}")
                meaning += f"; {self.get_honors(honors[i])} in {self.suitsymbols[i]}"

            bba_alert = self.players[self.position].get_info_alerting(self.C_INTERPRETED)
            if bba_alert:
                meaning += "; Artificial"
            if gf:
                meaning += "; GF"
            elif forcing:
                meaning += "; Forcing"
            elif forcing_to and forcing_to > new_bid and new_bid > 4:
                    meaning += f"; Forcing" # to {bidding.ID2BID[forcing_to]}
            if new_bid < 3:
                new_bid = new_bid + 2
            explain = {"bid": bidding.ID2BID[new_bid].replace("PASS","P"), "m": meaning, "Alert": bba_alert, "MinHcp": minhcp, "MaxHcp": maxhcp, "Length": length, "Honors": honor_str, "Stoppers": stoppers_str, "Strength": strength_str, "SuitPower": sp_str, "ProbableLength": pl_str}
            result.append(explain) 
            if self.verbose:
                print(f"{bidding.ID2BID[new_bid]}={meaning}")

        explain["NS"] = self.our_system_file
        explain["EW"] = self.their_system_file
        return result

    def extract_hcp(self, position):
        info = self.players[self.position].get_info_feature(position)

        minhcp = info[402]
        maxhcp = info[403]
        gf = info[443]
        forcing_to = info[411]
        forcing = info[412]
        if minhcp > 0:
            if maxhcp < 37:
                hcp = f" {minhcp}-{maxhcp} HCP"
            else:
                hcp = f" {minhcp}+ HCP"
        else:
            if maxhcp < 37:
                hcp = f" {maxhcp}- HCP"
            else:
                hcp = f""
        return minhcp,maxhcp,gf,forcing_to,forcing,hcp

    def extract_lengths(self, position):
        maxlength = self.players[self.position].get_info_max_length(position)
        minlength = self.players[self.position].get_info_min_length(position)
        length = []
        for i in range(4):
            if minlength[i] == 0: 
                if maxlength[i] != 13:
                    length.append(f"{maxlength[i]}-{self.suitsymbols[i]}")
            else:
                if maxlength[i] == 13:
                    length.append(f"{minlength[i]}+{self.suitsymbols[i]}")
                else:
                    if minlength[i] == maxlength[i]:
                        length.append(f"{minlength[i]}={self.suitsymbols[i]}")
                    else:
                        length.append(f"{minlength[i]}-{maxlength[i]}{self.suitsymbols[i]}")
        return length

    def get_sample(self, auction):

        arr_bids = []
        for k in range(len(auction)):
            bidid = bidding.BID2ID[auction[k]]
            if bidid < 2:
                continue
            if bidid < 5:
                bidid = bidid - 2
            arr_bids.append(f"{bidid:02}")

        no_bids = len(arr_bids)

        self.players[self.position].new_hand(self.position, _str_array(self.hand_str), self.dealer, self.bba_vul(self.vuln_nsew))

        arr_bids.extend([''] * (64 - len(arr_bids)))
        self.players[self.position].set_arr_bids(_str_array(arr_bids))
        # Temporary solution to call get_bid
        new_bid = self.players[self.position].get_bid()
        arr_suits = self.players[self.position].get_arr_suits()
        print("How BBA think the hands looks like:")
        print("Auction: ", auction)
        print(self.players[self.position].get_str_bidding())
        for i in reversed(range(4)):
            print(f"\t{arr_suits[i]}")
        for i in reversed(range(4)):
            print(f"{arr_suits[12 + i]}\t\t{arr_suits[4 + i]}")
        for i in reversed(range(4)):
            print(f"\t{arr_suits[8 + i]}")

        return arr_suits

    def bid_hand(self, auction, deal):
        # To get deterministic result the hand is always North
        position = 0
        dealer = ((self.dealer - self.position) + 4) % 4 

        hands = deal.split(":")[1].split(' ') 
        bba_auction = auction.copy()
        bba_hand = []
        for i in range(4):
            hand_str = hands[i].split('.')
            bba_hand.append(hand_str.copy())
            hand_str.reverse()
            if self.position % 2 == 0:  # N (0) and S (2)
                bba_vuln = self.bba_vul([self.vuln_wethey[0], self.vuln_wethey[1]])
            else:
                bba_vuln = self.bba_vul([self.vuln_wethey[1], self.vuln_wethey[0]])
            # The deal we get is always our hand first
            # First bid is opponent so we switch vulnerability
            self.players[i].new_hand(i, _str_array(hand_str), dealer, bba_vuln)

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
        # Always LHO" to start
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

        if self.verbose: 
            print(deal,bba_auction)
        
        return bba_auction

