import sys
sys.path.append('../../../src')

import datetime
import os.path
import numpy as np

from bidding.binary import DealData

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
        line = line.strip()
        if line_number % 2 == 0:
            deal_str = line
        else:
            yield DealData.from_deal_auction_string(deal_str, line, ns, ew, 32)

def create_binary(data_it, n, out_dir, ns, ew):
    if (ns==-1):
        x = np.zeros((4 * n, 8, 159), dtype=np.float16)
    else:
        x = np.zeros((4 * n, 8, 161), dtype=np.float16)
    y = np.zeros((4 * n, 8, 40), dtype=np.uint8)

    k = 0

    for i, deal_data in enumerate(data_it):
        if (i+1) % 1000 == 0:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i+1)
        x_part, y_part = deal_data.get_binary(ns, ew, n_steps=8)
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

    if len(sys.argv) < 3:
        print("Usage: python bidding_binary.py inputfile outputdirectory NS=<x> EW=<y>")
        print("NS and EW are optional. If set to -1 no information about system is included in the model.")
        print("If set to 0 the hands from that side will not be used for training.")
        print("The input file is the BEN-format (1 line with hands, and next line with the bidding).")
        sys.exit(1)

    infnm = sys.argv[1] # file where the data is
    outdir = sys.argv[2]
    # Extract NS and EW values from command-line arguments if provided
    ns = next((extract_value(arg) for arg in sys.argv[3:] if arg.startswith("NS=")), -1)
    ew = next((extract_value(arg) for arg in sys.argv[3:] if arg.startswith("EW=")), -1)

    ns = to_numeric(ns)
    ew = to_numeric(ew)

    with open(infnm, 'r') as file:
        lines = file.readlines()
        n = len(lines)
        create_binary(load_deals(lines), n, outdir, ns, ew)

