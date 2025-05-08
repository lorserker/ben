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
        if line_number % 3 == 0:
            deal_str = line
        else:
            if line_number % 3 == 1:
                auction_str = line
            else:
                yield DealData.from_deal_auction_string(deal_str, auction_str, line, -1, -1, 32)

def create_binary(data_it, n, out_dir):
    x = np.zeros((n, 66), dtype=np.float16)
    # Contract
    y = np.zeros((n, 40), dtype=np.uint8)
    # Doubled
    z = np.zeros((n, 1), dtype=np.uint8)
    # Tricks
    u = np.zeros((n, 14), dtype=np.uint8)
    k = 0

    for i, deal_data in enumerate(data_it):
        if (i+1) % 10000 == 0:
            sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {i+1}\n')
            sys.stderr.flush()
        x_part, y_part, z_part, u_part= deal_data.get_binary_contract()
        x[k] = x_part
        y[k] = y_part
        z[k] = z_part
        u[k] = u_part
        k += 1

    np.save(os.path.join(out_dir, 'x.npy'), x)
    np.save(os.path.join(out_dir, 'y.npy'), y)
    np.save(os.path.join(out_dir, 'z.npy'), z)
    np.save(os.path.join(out_dir, 'u.npy'), u)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python contract_binary.py inputfile outputdirectory")
        print("The input file is the PAR-format (1 line with hands, next line with the vulnerability, and finally a line with optimum results).")
        sys.exit(1)

    infnm = sys.argv[1] # file where the data is
    outdir = sys.argv[2]

    with open(infnm, 'r') as file:

        lines = file.readlines()
        # Remove comments at the beginning of the file
        lines = [line for line in lines if not line.strip().startswith('#')]
        n = len(lines) // 2
        print(f"Loading {n} deals")

        data_it = load_deals(lines)
        create_binary(data_it, n, outdir)

