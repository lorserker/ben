import sys
sys.path.append('../../../src')

import datetime
import os.path
import numpy as np

from bidding.binary import DealData
np.set_printoptions(precision=2, suppress=True, linewidth=220)

def load_deals(fin):
    deal_str = ''

    for line_number, line in enumerate(fin):
        # For now we remove alerts until we have an useable implementation
        # line = line.strip().replace("*",'')
        if line_number % 2 == 0:
            deal_str = line
        else:
            yield DealData.from_deal_auction_string(deal_str, line, "", ns, ew, 32)

def create_binary(data_it, n, out_dir, ns, ew, alternating, bids, alert_supported = False):
    if ns == 0 or ew == 0:
        rows_pr_hand = 2
    else:
        rows_pr_hand = 4
    if (ns==-1):
        x = np.zeros((rows_pr_hand * n, 8, 39 + bids * 40), dtype=np.float16)
    else:
        x = np.zeros((rows_pr_hand * n, 8, 41 + bids * 40), dtype=np.float16)
    y = np.zeros((rows_pr_hand * n, 8, 40), dtype=np.uint8)

    z = np.zeros((rows_pr_hand * n, 8, 1), dtype=np.uint8)
    HCP = np.zeros((rows_pr_hand * n, 8, 3), dtype=np.float16)
    SHAPE = np.zeros((rows_pr_hand * n, 8, 12), dtype=np.float16)
    k = 0

    for i, deal_data in enumerate(data_it):
        if (i+1) % 10000 == 0:
            sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {i+1}\n')
            sys.stderr.flush()
        v = 0
        if alternating and (i % 2) == 1:
            x_part, y_part, hcp_part, shape_part, alert_part = deal_data.get_binary_hcp_shape(ew, ns, bids, n_steps = 8, alert_supported = alert_supported)
        else:
            x_part, y_part, hcp_part, shape_part, alert_part = deal_data.get_binary_hcp_shape(ns, ew, bids, n_steps = 8, alert_supported = alert_supported)
        if ns == 0:
            # with system = 0 we discard the hand
            x[k:k+1] = x_part[1]
            y[k:k+1] = y_part[1]
            z[k:k+1] = alert_part[1]
            HCP[k:k+1] = hcp_part[1]
            SHAPE[k:k+1] = shape_part[1]
            x[k+1:k+2] = x_part[3]
            y[k+1:k+2] = y_part[3]
            z[k+1:k+2] = alert_part[3]
            HCP[k+1:k+2] = hcp_part[3]
            SHAPE[k+1:k+2] = shape_part[3]
            k += 2
        elif ew == 0:
            # with system = 0 we discard the hand
            x[k:k+1] = x_part[0]
            y[k:k+1] = y_part[0]
            z[k:k+1] = alert_part[0]
            HCP[k:k+1] = hcp_part[0]
            SHAPE[k:k+1] = shape_part[0]
            x[k+1:k+2] = x_part[2]
            y[k+1:k+2] = y_part[2]
            z[k+1:k+2] = alert_part[2]
            HCP[k+1:k+2] = hcp_part[2]
            SHAPE[k+1:k+2] = shape_part[2]
            k += 2
        else:
            x[k:k+4] = x_part
            y[k:k+4] = y_part
            HCP[k:k+4] = hcp_part
            SHAPE[k:k+4] = shape_part
            z[k:k+4] = alert_part
            k += 4

    np.save(os.path.join(out_dir, 'x.npy'), x)
    np.save(os.path.join(out_dir, 'y.npy'), y)
    np.save(os.path.join(out_dir, 'z.npy'), z)
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
        print("Usage: python bidding_binary.py inputfile outputdirectory NS=<x> EW=<y> alternate=True version=2 alert_supported=True")
        print("NS and EW are optional. If set to -1 no information about system is included in the model.")
        print("If set to 0 the hands from that side will not be used for training.")
        print("The input file is the BEN-format (1 line with hands, and next line with the bidding).")
        print("alternate is signaling, that the input file has both open and closed room, so NS/EW will be alternated")
        sys.exit(1)

    infnm = sys.argv[1] # file where the data is
    outdir = sys.argv[2]
    # Extract NS and EW values from command-line arguments if provided
    ns = next((extract_value(arg) for arg in sys.argv[3:] if arg.startswith("NS=")), -1)
    ew = next((extract_value(arg) for arg in sys.argv[3:] if arg.startswith("EW=")), -1)
    alternating = next((extract_value(arg) for arg in sys.argv[3:] if arg.startswith("alternate")), False)
    version = next((extract_value(arg) for arg in sys.argv[2:] if arg.startswith("version")), "2")
    alert_supported = next((extract_value(arg) for arg in sys.argv[3:] if arg.startswith("alert_supported")), False)
    sys.stderr.write(f"NS={ns}, EW={ew}, Alternating={alternating}, Version={version}, alert_supported={alert_supported}\n")
    ns = to_numeric(ns)
    ew = to_numeric(ew)
    version = to_numeric(version)

    with open(infnm, 'r') as file:

        lines = file.readlines()
        # Remove comments at the beginning of the file
        lines = [line for line in lines if not line.strip().startswith('#')]
        n = len(lines) // 2
        print(f"Loading {n} deals")
        if n > 1 and lines[0] == lines[2] and not alternating:
            user_input = input("\n First two boards are identical - did you forget to add alternate=True?")
            if user_input.lower() == "y":
                sys.exit()

        data_it = load_deals(lines)
        create_binary(data_it, n, outdir, ns, ew, alternating, 4 if version > 1 else 3, alert_supported=alert_supported)

