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

if "src" in script_dir and "suitc" in script_dir: 
    # We are running inside the src/pimc directory
    BIN_FOLDER = parent_dir + os.path.sep + 'bin'
else:

    BEN_HOME = os.getenv('BEN_HOME') or '.'
    if BEN_HOME == '.':
        BIN_FOLDER = 'bin'
    else:
        BIN_FOLDER = os.path.join(BEN_HOME, 'bin')

if sys.platform == 'win32':
    suitclib = 'SuitCLib.dll'
elif sys.platform == 'darwin':
    suitclib = 'libsuitc.so'
else:
    suitclib = 'libsuitc.so'

print(f"SuitCLib: {suitclib}")
print(sys.platform)

SuitCLib_PATH = os.path.join(BIN_FOLDER, suitclib)

import ctypes
from ctypes import c_wchar_p, c_int, POINTER, create_unicode_buffer, byref, cast, addressof

class SuitCLib:
    def __init__(self, verbose):
        if SuitCLib == 'N/A':
            raise RuntimeError("suitclib is not available on this platform.")
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
            print(f"Error: Unable to load {SuitCLib_PATH}. Make sure the file is in the ./bin directory")
            print("Make sure the file is not blocked by OS (Select properties and click unblock)")
            print("Make sure the filw is not write protected")
            print(f"*****************************************************************************{Fore.RESET}")
            sys.exit(1)
        self.verbose = verbose

    def version(self):
        self.suitc.version.restype = ctypes.c_char_p
        return self.suitc.version().decode('utf-8')
    
    def get_suit_tricks(self, declarer, dummy, opponent):
        #if self.verbose:
        #    input_str = " -a -b "
        #else:
        input_str = ""
        #input_str = ""

        input_str += f"{dummy if dummy != '' else '.'} {declarer if declarer != '' else '.'} {opponent}"
        try:
            output, details = self.make_suitc_call(input_str)
            response_dict = json.loads(output)
            optimum_result = round(response_dict["SuitCAnalysis"]["Result"],2)
            if self.verbose:
                print("optimum_result", optimum_result, declarer, dummy, opponent)
        except Exception as ex:
            print('Error:', ex)
            print( declarer, dummy, opponent)
            raise ex
        
        return optimum_result

    def get_trick_potential(self, declarer, dummy):
        declarer_suits = declarer.split('.')
        dummy_suits = dummy.split('.')
        potential = []
        for i in range(4):
            eastwest =  ''.join(c for c in "AKQJT98765432" if c not in declarer_suits[i] and c not in dummy_suits[i])
            optimum_result = self.get_suit_tricks(dummy_suits[i], declarer_suits[i], eastwest)
            potential.append(optimum_result)
            #print(f"{dummy_suits[i]} {declarer_suits[i]} {optimum_result}")
        return potential

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
        output, details = self.make_suitc_call(input_str)

        #print(f"{Fore.GREEN}SuitC Output: {len(output_buffer.value)}{Fore.RESET}")
        #print(f"{Fore.GREEN}SuitC Output details:  {len(details_buffer.value)}{Fore.RESET}")
        response_dict = json.loads(output)
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

    def make_suitc_call(self, input_str):
        input_length = len(input_str)
        if self.verbose:
            print("SuitC Input: " + input_str)
        
        # --- Input Buffer (wchar_t** ppcharInput) ---
        # C side will do (*ppcharInput)[i]
        # So Python needs to create a wchar_t* and pass a pointer to it.
        # Method 1: ctypes.c_wchar_p implicitly handles string literal
        c_input_str_ptr = ctypes.c_wchar_p(input_str) # This is a wchar_t*
        # To pass wchar_t**, we pass a pointer to this wchar_t*
        pp_input_str = ctypes.byref(c_input_str_ptr)

        # --- Output Buffer (wchar_t** ppcharOutput) ---
        # C side will do (*ppcharOutput)[i] = ... to fill Python's buffer
        # Python needs to create a buffer, get a wchar_t* to it, and pass a pointer to that wchar_t*
        output_buffer_capacity = 8 * 32768  # Number of wchar_t characters
        # This is the actual memory for the output string
        actual_output_buffer = ctypes.create_unicode_buffer(output_buffer_capacity) 
        # This is a wchar_t* pointing to the start of actual_output_buffer
        c_output_buffer_ptr = ctypes.c_wchar_p(ctypes.addressof(actual_output_buffer)) 
        # To pass wchar_t**, we pass a pointer to this c_output_buffer_ptr
        pp_output_buffer = ctypes.byref(c_output_buffer_ptr)
        output_size = ctypes.c_int() # For int* pCharOutputSize

        # --- Details Buffer (wchar_t** ppcharOptimumDetails) ---
        details_buffer_capacity = 8 * 32768
        actual_details_buffer = ctypes.create_unicode_buffer(details_buffer_capacity)
        c_details_buffer_ptr = ctypes.c_wchar_p(ctypes.addressof(actual_details_buffer))
        pp_details_buffer = ctypes.byref(c_details_buffer_ptr)
        details_size = ctypes.c_int() # For int* pcharOptimumDetailsSize

        result = self.suitc.call_suitc(
            pp_input_str,
            input_length,
            pp_output_buffer,
            ctypes.byref(output_size),
            pp_details_buffer,
            ctypes.byref(details_size)
        )

        if self.verbose:
            print(f"call_suitc returned: {result}")
            print(f"Output size from C: {output_size.value}")
            print(f"Details size from C: {details_size.value}")

        if result != 0:
            print("Error: " + result)
            sys.exit(1)
        # To get the string values from the buffers:
        # The C code wrote into actual_output_buffer and actual_details_buffer
        # via the pointers.
        returned_output = actual_output_buffer.value[:output_size.value] # Slice to actual length
        returned_details = actual_details_buffer.value[:details_size.value]

        if self.verbose:
            print(f"Returned output: '{returned_output}'")
            print(f"Returned details: '{returned_details}'")

        return returned_output, returned_details