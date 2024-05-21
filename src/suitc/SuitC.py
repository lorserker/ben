import sys
import os
sys.path.append("..")

BEN_HOME = os.getenv('BEN_HOME') or '..'
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
        try:
            self.suitc = ctypes.CDLL(SuitCLib_PATH)
            # Define the argument and return types of the C++ function
            self.suitc.call_suitc.argtypes = [POINTER(c_wchar_p), c_int,
                                            POINTER(c_wchar_p), POINTER(c_int),
                                            POINTER(c_wchar_p), POINTER(c_int)]
            self.suitc.call_suitc.restype = c_int
            self.suitc.call_suitc1.argtypes = [POINTER(c_wchar_p), c_int]
            self.suitc.call_suitc1.restype = c_int
        except Exception as ex:
            # Provide a message to the user if the assembly is not found
            print('Error:', ex)
            print("*****************************************************************************")
            print("Error: Unable to load SuitCLib.dll. Make sure the DLL is in the ./bin directory")
            print("Make sure the dll is not blocked by OS (Select properties and click unblock)")
            print("Make sure the dll is not write protected")
            print("*****************************************************************************")
            sys.exit(1)
        self.verbose = verbose
    
    def calculate(self):
        input_str = "-wv9 -ev8 K3 2 AQJT987654"
        input_length = len(input_str)
        
        # Convert input string to a wide char buffer
        input_buffer = create_unicode_buffer(input_str + '\0')  # Ensure null termination

        # Create a pointer to the buffer
        input_buffer_ptr = ctypes.pointer(c_wchar_p(input_buffer.value))
        
        print(f"Input Buffer Address (Python): {ctypes.addressof(input_buffer)}")
        print(input_buffer_ptr)

        print("_________________________________________________________________________________________")
        result = self.suitc.call_suitc1(input_buffer_ptr, input_length)
        print("Result call_suitc1:", result)

        # Create an output buffer
        output_length = 32768
        output_buffer = create_unicode_buffer(output_length)

        # Create a pointer to the output buffer
        output_buffer_ptr = ctypes.pointer(c_wchar_p(output_buffer.value))
        # Create a variable to hold the output buffer size
        output_size = ctypes.c_int()
                # Create an output buffer
        details_length = 32768
        details_buffer = create_unicode_buffer(details_length)
        # Create a variable to hold the output buffer size
        details_size = ctypes.c_int()

        # Create a pointer to the output buffer
        details_buffer_ptr = ctypes.pointer(c_wchar_p(details_buffer.value))
        print("_________________________________________________________________________________________")
        result = self.suitc.call_suitc(input_buffer_ptr, input_length, output_buffer_ptr,  byref(output_size), details_buffer_ptr,  byref(details_size))
        print("Result call_suitc:", result)
