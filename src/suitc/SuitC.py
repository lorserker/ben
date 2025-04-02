import sys
import os
import json
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Calculate the parent directory
parent_dir = os.path.join(script_dir, "../..")
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from colorama import Fore, Back, Style, init

BEN_HOME = os.getenv('BEN_HOME') or '..'
if BEN_HOME == '.':
    BIN_FOLDER = os.getcwd().replace(os.path.sep + "src","")+ os.path.sep + 'bin'
else:
    BIN_FOLDER = os.path.join(BEN_HOME, 'bin')

if sys.platform == 'win32':
    SuitCLib = 'SuitCLib.dll'
elif sys.platform == 'darwin':
    SuitCLib_LIB = 'N/A'
else:
    SuitCLib_LIB = 'N/A'

SuitCLib_PATH = os.path.join(BIN_FOLDER, SuitCLib)


import ctypes
from ctypes import c_wchar_p, c_int, POINTER, create_unicode_buffer, byref, cast, addressof

class SuitCLib:
    def __init__(self, verbose):
        if SuitCLib == 'N/A':
            raise RuntimeError("SuitCLib.dll is not available on this platform.")
        try:
            if verbose:
                print(f"loading {SuitCLib_PATH}")
            self.suitc = ctypes.CDLL(SuitCLib_PATH)
            # Define the argument and return types of the C++ function
            self.suitc.call_suitc.argtypes = [POINTER(c_wchar_p), c_int,
                                            POINTER(c_wchar_p), POINTER(c_int),
                                            POINTER(c_wchar_p), POINTER(c_int)]
            self.suitc.call_suitc.restype = c_int
            self.verbose = verbose
        except Exception as ex:
            # Provide a message to the user if the assembly is not found
            print(f"{Fore.RED}Error:", ex)
            print("*****************************************************************************")
            print("Error: Unable to load SuitCLib.dll. Make sure the DLL is in the ./bin directory")
            print("Make sure the dll is not blocked by OS (Select properties and click unblock)")
            print("Make sure the dll is not write protected")
            print(f"*****************************************************************************{Fore.RESET}")
            sys.exit(1)
        self.verbose = verbose

    def version(self):
        self.suitc.version.restype = ctypes.c_char_p
        return self.suitc.version().decode('utf-8')
    
    def calculate(self, max_tricks, north, south, eastwest, east_vacant=None, west_vacant=None, trump = False, entries = 1 ):
        # if matchoint is true, then -M is used
        # if verbose is true, then -a and -b is used
        # -F5 is combines the effect of -F1 and -F4, -F7 combines all 3 options.
        # -ls limits the entries 
        # -ls is most important when the hand to lead has length
        # -Ls set south to lead
        # consider adding vacant places -wn<n> -en<n>
        if self.verbose:
            input_str = " -F1 -u -c100 -a -b "
        else:
            input_str = " -F1 -u -c100 "
        if not trump:
            input_str += f"-ls{entries} "
        if east_vacant:
            input_str +=f'-wv{west_vacant} '
            input_str +=f'-ev{east_vacant} '
        #input_str = ""
        input_str += f"{north} {south} {eastwest}"
        input_length = len(input_str)
        if self.verbose:
            print("SuitC Input: " + input_str)
        
        # Convert input string to a wide char buffer
        input_buffer = create_unicode_buffer(input_str + '\0')  # Ensure null termination

        # Create a pointer to the buffer
        input_buffer_ptr = ctypes.pointer(c_wchar_p(input_buffer.value))
        
        # Create an output buffer
        output_length = 8 * 32768
        output_buffer = create_unicode_buffer(output_length)

        # Create a variable to hold the output buffer size
        output_size = ctypes.c_int()

        # Create details buffer
        details_length = 8 * 32768
        details_buffer = create_unicode_buffer(details_length)
        # Create a variable to hold the output buffer size
        details_size = ctypes.c_int()

        # Pointers to the output and details buffers
        # Create pointers to the output and details buffers
        output_buffer_ptr = c_wchar_p(ctypes.addressof(output_buffer))
        details_buffer_ptr = c_wchar_p(ctypes.addressof(details_buffer))

        result = self.suitc.call_suitc(input_buffer_ptr, input_length, output_buffer_ptr,  byref(output_size), details_buffer_ptr,  byref(details_size))
        if result != 0:
            print("Error: " + result)
            sys.exit(1)
        if self.verbose:
            print(output_buffer.value)
            print(details_buffer.value)

        #print("SuitC Output: " + output_buffer.value)
        response_dict = json.loads(output_buffer.value)
        optimum_plays = response_dict["SuitCAnalysis"]["OptimumPlays"]
        # print("optimum_plays", optimum_plays)
        # We just take the play for MAX as we really don't know how many tricks are needed
        possible_cards = []
        for play in optimum_plays:
            # If we can take all tricks we drop SuitC
            if play['Plays'][0]['Tricks'] == max_tricks:
                if play['Plays'][0]['Percentage'] == 100:
                    if self.verbose:
                        print(f"SuitC dropped as we can take all tricks")
                    return possible_cards
            # We can have more than one play for MAX
            # So currently we are then selecting higest card. Should that be different?
            # We should probably look at the samples to find the best play
            if "MAX" in play["OptimumPlayFor"]:
                if len(play["GameTree"]) > 0:
                    for card in play["GameTree"]:
                        for key, card in card.items():
                            if key == "T":  
                                actual_card = card[-1]
                                if actual_card in north:
                                    if self.verbose:
                                        print(f"Skipping play from North {card} {input_str}")
                                    continue
                                if actual_card in south:
                                    if self.verbose:
                                        print(f"Play from South {card} {input_str}")
                                    possible_cards.append(actual_card) 
                                    continue
                                if self.verbose:
                                    print("SuitC found play not in North or South", card)
                else:
                    if self.verbose:
                        print(f"SuitC found no gametree. {input_str}")
                    return possible_cards
        if self.verbose and len(possible_cards) == 0:
            print(f"SuitC found no Optimum play for MAX. {input_str}")

        return possible_cards
