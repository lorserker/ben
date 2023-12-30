import sys
sys.path.append('../../../src')

import datetime
import os.path
import numpy as np

from bidding.binary import DealData
np.set_printoptions(precision=2, suppress=True, linewidth=220)

LEVELS = [1, 2, 3, 4, 5, 6, 7]

SUITS = ['C', 'D', 'H', 'S', 'N']
SUIT_RANK = {suit: i for i, suit in enumerate(SUITS)}

BID2ID = {
    'PAD_START': 0,
    'PAD_END': 1,
    'PASS': 2,
    'X': 3,
    'XX': 4,
}

SUITBID2ID = {bid: (i+5) for (i, bid) in enumerate(
    ['{}{}'.format(level, suit) for level in LEVELS for suit in SUITS])}

BID2ID.update(SUITBID2ID)

ID2BID = {bid: i for i, bid in BID2ID.items()}

def load_deals(fin):
    deal_str = ''

    for line_number, line in enumerate(fin):
        # For now we remove alerts until we have an useable implementation
        line = line.strip().replace("*",'')
        if line_number % 2 == 0:
            deal_str = line
        else:
            yield DealData.from_deal_auction_string(deal_str, line, ns, ew, 32)

def create_binary(data_it, n, out_dir, ns, ew, alternating):
    if ns == 0 or ew == 0:
        rows_pr_hand = 2
    else:
        rows_pr_hand = 4
    if (ns==-1):
        x = np.zeros((rows_pr_hand * n, 8, 159), dtype=np.float16)
    else:
        x = np.zeros((rows_pr_hand * n, 8, 161), dtype=np.float16)
    y = np.zeros((rows_pr_hand * n, 8, 40), dtype=np.uint8)

    k = 0

    for i, deal_data in enumerate(data_it):
        if (i+1) % 1000 == 0:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i+1)
        if alternating and (i % 2) == 1:
            x_part, y_part = deal_data.get_binary(ew, ns, n_steps=8)
        else:
            x_part, y_part = deal_data.get_binary(ns, ew, n_steps=8)
        if ns == 0:
            # with system = 0 we discard the hand
            x[k:k+1] = x_part[1]
            y[k:k+1] = y_part[1]
            x[k+1:k+2] = x_part[3]
            y[k+1:k+2] = y_part[3]
            k += 2
        elif ew == 0:
            # with system = 0 we discard the hand
            x[k:k+1] = x_part[0]
            y[k:k+1] = y_part[0]
            x[k+1:k+2] = x_part[2]
            y[k+1:k+2] = y_part[2]
            k += 2
        else:
            x[k:k+4] = x_part
            y[k:k+4] = y_part
            k += 4

    np.save(os.path.join(out_dir, 'x.npy'), x)
    np.save(os.path.join(out_dir, 'y.npy'), y)

# Function to extract value from command-line argument
def extract_value(arg):
    return arg.split('=')[1]

def to_numeric(value, default=0):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return default

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python bidding_binary.py inputfile outputdirectory NS=<x> EW=<y> alternate=True")
        print("NS and EW are optional. If set to -1 no information about system is included in the model.")
        print("If set to 0 the hands from that side will not be used for training.")
        print("The input file is the BEN-format (1 line with hands, and next line with the bidding).")
        print("alternate is signaling, that the input file has both open and closed room, so NS/EW will be altarnated")
        sys.exit(1)

    infnm = sys.argv[1] # file where the data is
    outdir = sys.argv[2]
    # Extract NS and EW values from command-line arguments if provided
    ns = next((extract_value(arg) for arg in sys.argv[3:] if arg.startswith("NS=")), -1)
    ew = next((extract_value(arg) for arg in sys.argv[3:] if arg.startswith("EW=")), -1)
    alternating = next((extract_value(arg) for arg in sys.argv[3:] if arg.startswith("alternate")), False)

    print(ns, ew, alternating)
    ns = to_numeric(ns)
    ew = to_numeric(ew)

    with open(infnm, 'r') as file:

        lines = file.readlines()
        # Remove comments at the beginning of the file
        lines = [line for line in lines if not line.strip().startswith('#')]
        n = len(lines) // 2
        print(f"Loading {n} deals")
        if lines[0] == lines[2] and not alternating:
            user_input = input("\n First two boards are identical - did you forget to add alternate=True?")
            if user_input.lower() == "y":
                sys.exit()

        create_binary(load_deals(lines), n, outdir, ns, ew, alternating)

