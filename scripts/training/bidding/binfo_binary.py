import sys
sys.path.append('../../../src')

import datetime
import os.path
import numpy as np

from bidding.binary import DealData

## HCP target format
# 0 = hcp lho
# 1 = hcp part
# 2 = hcp rho

## SHAPE target format
# 0:4 shape S,H,D,C lho
# 4:8 partner
# 8:12 rho


def load_deals_no_contracts(fin):
    deal_str = ''
    auction_str = ''

    for line_number, line in enumerate(fin):
        line = line.strip()
        if line_number % 2 == 0:
            if line_number > 0:
                yield (deal_str, auction_str, None)
                deal_str = ''
                auction_str = ''
            deal_str = line
        elif line_number % 2 == 1:
            auction_str = line

    yield (deal_str, auction_str, None)


def create_binary(data_it, n, out_dir, ns, ew):
    if (ns==-1):
        x = np.zeros((4 * n, 8, 159), dtype=np.float16)
    else:
        x = np.zeros((4 * n, 8, 161), dtype=np.float16)
    y = np.zeros((4 * n, 8, 40), dtype=np.float16)
    HCP = np.zeros((4 * n, 8, 3), dtype=np.float16)
    SHAPE = np.zeros((4 * n, 8, 12), dtype=np.float16)

    for i, (deal_str, auction_str, _) in enumerate(data_it):
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i)
        deal_data = DealData.from_deal_auction_string(deal_str, auction_str, ns, ew, 32)
        x_part, y_part, hcp_part, shape_part = deal_data.get_binary_hcp_shape(ns, ew, n_steps = 8)
        start_ix = i * 4
        end_ix = (i + 1) * 4
        x[start_ix:end_ix,:,:] = x_part
        y[start_ix:end_ix,:,:] = y_part
        HCP[start_ix:end_ix,:,:] = hcp_part
        SHAPE[start_ix:end_ix,:,:] = shape_part

    print(HCP[0])
    np.save(os.path.join(out_dir, 'x.npy'), x)
    np.save(os.path.join(out_dir, 'y.npy'), y)
    np.save(os.path.join(out_dir, 'HCP.npy'), HCP)
    np.save(os.path.join(out_dir, 'SHAPE.npy'), SHAPE)

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
        print("Usage: python binfo_binary.py inputfile outputdirectory NS=<x> EW=<y>")
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
        create_binary(load_deals_no_contracts(lines), n, outdir, ns, ew)
