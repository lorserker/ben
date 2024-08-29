import os.path
import copy
import sys
sys.path.append('../../../src')

import datetime
from collections import Counter, defaultdict
import numpy as np

from bidding.binary import DealData
np.set_printoptions(precision=2, suppress=True, linewidth=220)

out_dir = "bidding_bin"

def load_deals(fin, n_cards=32):
    deal_str = ''

    for line_number, line in enumerate(fin):
        # For now we remove alerts until we have an useable implementation
        # line = line.strip().replace("*",'')
        if line_number % 2 == 0:
            deal_str = line
        else:
            yield DealData.from_deal_auction_string(deal_str, line, "", ns, ew, n_cards)


def create_arrays(ns, ew, players_pr_hand, n, bids, alert_supported = False, n_cards=32):
    if (ns==-1):
        x = np.zeros((players_pr_hand * n, 8, n_cards + 7 + bids * 40), dtype=np.float16)
    else:
        x = np.zeros((players_pr_hand * n, 8, n_cards + 9 + 40*bids), dtype=np.float16)
    z = np.zeros((players_pr_hand * n, 8, 1), dtype=np.uint8)
    y = np.zeros((players_pr_hand * n, 8, 40), dtype=np.uint8)

    HCP = np.zeros((players_pr_hand * n, 8, 3), dtype=np.float16)
    SHAPE = np.zeros((players_pr_hand * n, 8, 12), dtype=np.float16)
    return x, y, z, HCP, SHAPE

def count_deals(data_it, max_occurrences, auctions, key_counts, deals, alert_supported = False):
    
    for i, deal_data in enumerate(data_it):
        if (i+1) % 50000 == 0:
            sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {i+1}\n')
            sys.stderr.flush()
        v = 0
        if deal_data.vuln_ns and not deal_data.vuln_ew: v = 1
        if not deal_data.vuln_ns and deal_data.vuln_ew: v = 2
        if deal_data.vuln_ns and deal_data.vuln_ew: v = 3
        auction = f"{ew}-{ns} {' '.join(deal_data.auction).replace('PASS','P').replace('PAD_START','-')} {v} "
        if key_counts[auction] < max_occurrences:
            auctions.append(auction)
            key_counts[auction] += 1
            deals.append(deal_data)
    
    return deals, auctions, key_counts

def create_binary(data_it, ns, ew, alternating, x, y, z, HCP, SHAPE, key_counts, k, bids, alert_supported = False):

    for i, deal_data in enumerate(data_it):
        if (i+1) % 10000 == 0:
            sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {i+1}\n')
            sys.stderr.flush()
        v = 0
        if deal_data.vuln_ns and not deal_data.vuln_ew: v = 1
        if not deal_data.vuln_ns and deal_data.vuln_ew: v = 2
        if deal_data.vuln_ns and deal_data.vuln_ew: v = 3
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
            z[k:k+4] = alert_part
            HCP[k:k+4] = hcp_part
            SHAPE[k:k+4] = shape_part
            k += 4

    return x, y, z, HCP, SHAPE, key_counts, k

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
        print("Usage: python bidding_coverage.py inputfile1 inputfile2 NS=<x> EW=<y> alternate=True version=2 alert_supported=True n_cards=32")
        print("NS and EW are optional. If set to -1 no information about system is included in the model.")
        print("If set to 0 the hands from that side will not be used for training.")
        print("The input file is the BEN-format (1 line with hands, and next line with the bidding).")
        print("alternate is signaling, that the input file has both open and closed room, so NS/EW will be alternated")
        print("n_cards is the number of cards in the deck")
        sys.exit(1)

    infnm1 = sys.argv[1] # file where the data is
    if len(sys.argv) > 2 and sys.argv[2] != "None":
        infnm2 = sys.argv[2] # file where the data is
    else:
        infnm2 = None
    
    out_dir = sys.argv[3]

    # Extract NS and EW values from command-line arguments if provided
    ns = next((extract_value(arg) for arg in sys.argv[2:] if arg.startswith("NS=")), -1)
    ew = next((extract_value(arg) for arg in sys.argv[2:] if arg.startswith("EW=")), -1)

    alternating = next((extract_value(arg) for arg in sys.argv[2:] if arg.startswith("alternate")), False)
    version = next((extract_value(arg) for arg in sys.argv[2:] if arg.startswith("version")), "2")
    alert_supported = next((extract_value(arg) for arg in sys.argv[3:] if arg.startswith("alert_supported")), False)
    n_cards = next((extract_value(arg) for arg in sys.argv[3:] if arg.startswith("n_cards=")), 32)
    sys.stderr.write(f"NS={ns}, EW={ew}, Alternating={alternating}, Version={version}, alert_supported={alert_supported}\n")
    max_occurrences = 20
    max_filler_occurrences = 1
    sys.stderr.write(f"n_cards={n_cards}, NS={ns}, EW={ew}, Alternating={alternating}, Version={version}, alert_supported={alert_supported}, outdir={out_dir},  max_occurrences={max_occurrences}, max_filler_occurrences={max_filler_occurrences}\n")

    ns = to_numeric(ns)
    ew = to_numeric(ew)
    n_cards = to_numeric(n_cards, 32)

    if ns == 0 or ew == 0:
        players_pr_hand = 2
    else:
        players_pr_hand = 4

    max_occurrences = 10
    # Count the occurrences of each line
    key_counts = Counter()
    auctions = []
    k = 0

    with open(infnm1, 'r') as file:

        print(f"Loading {infnm1}")
        lines = file.readlines()
        # Remove comments at the beginning of the file
        lines = [line for line in lines if not line.strip().startswith('#')]
        sys.stderr.write(f"Loading {len(lines) // 2} deals\n")
        if len(lines) > 2 and lines[0] == lines[2] and not alternating:
            user_input = input("\n First two boards are identical - did you forget to add alternate=True?")
            if user_input.lower() == "y":
                sys.exit()
        if len(lines) > 2 and lines[0] != lines[2] and alternating:
            user_input = input("\n First two boards are not identical - did you mean alternate=True?")
            if user_input.lower() == "n":
                sys.exit()

        data_it = load_deals(lines, n_cards)
        filtered_deals, auctions, key_counts = count_deals(data_it, max_occurrences, auctions, key_counts, [], alert_supported=alert_supported )

    # Create a new Counter to count all occurrences without limit
    sys.stderr.write(f"Loaded {len(auctions)} auctions\n")   
    key_counts2 = Counter(auctions)
    
    # Sort by count
    sorted_keys_by_count = sorted(key_counts.items(), key=lambda x: (-x[1], x[0]))  # Sort by count and key in descending order

    count = len(sorted_keys_by_count)
    sys.stderr.write(f"Found {count} bidding sequences.\n" )
    sys.stderr.write(f"Filtered_deals {len(filtered_deals)}\n")
    sys.stderr.write(f"Removed {len(lines) // 2 - len(filtered_deals)} deals where the same bidding was seen more than {max_occurrences} times.\n")
    
    # The second training dataset is random deals, and we just use the bidding sequences we don't allready have
    if infnm2:
        max_occurrences = 2
        with open(infnm2, 'r') as file:

            lines = file.readlines()
            # Remove comments at the beginning of the file
            lines = [line for line in lines if not line.strip().startswith('#')]
            sys.stderr.write(f"Loading {len(lines) // 2} deals for fillers\n")
            data_it = load_deals(lines, n_cards)
            filtered_deals, auctions, key_counts = count_deals(data_it, max_filler_occurrences, auctions, key_counts, filtered_deals, alert_supported=alert_supported)

            # Create a new Counter to count all occurrences without limit
            sys.stderr.write(f"Loaded {len(auctions)} auctions from second dataset\n")   
            key_counts = Counter(auctions)
            
            count = len(sorted_keys_by_count)
            sys.stderr.write(f"Found {count} bidding sequences.\n" )

            sys.stderr.write(f"Filtered_deals {len(filtered_deals)}\n")


    # Sorting the list based on vuln_ns, vuln_ew, and auction
    sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Sort by vulnerability and auction\n')
    sorted_filtered_deals = sorted(filtered_deals, key=lambda x: (x.vuln_ns, x.vuln_ew, x.auction))

    combinations_dict = defaultdict(list)
    # Create a set to keep track of processed auctions
    processed_auctions = set()
    for board in sorted_filtered_deals:
        auction = ' '.join([x.replace('PASS', 'P') for x in board.auction if x != 'PAD_START'])
        combinations_dict[auction].append((board.vuln_ns, board.vuln_ew))

    sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Combinations {len(combinations_dict)}\n')

    sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Before add missing vuln {len(sorted_filtered_deals)}\n')
    i = 0
    sorted_filtered_deals = sorted(sorted_filtered_deals, key=lambda x: (x.auction))

    # Print all unique keys sorted
    #for board in sorted_filtered_deals:
    #    print(f"{board.vuln_ns} {board.vuln_ew} {' '.join([x.replace('PASS', 'P') for x in board.auction if x != 'PAD_START'])}")
    
    for board in sorted_filtered_deals:
        missing_combinations = []
        auction = ' '.join([x.replace('PASS', 'P') for x in board.auction if x != 'PAD_START'])
        if auction in processed_auctions:
            continue  # Skip if the auction has already been processed
        
        for vuln_ns in [True, False]:
            for vuln_ew in [True, False]:
                if (vuln_ns, vuln_ew) not in combinations_dict[auction]:
                    missing_combinations.append((vuln_ns, vuln_ew))

        for missing_combination in missing_combinations:
            # Create a copy of the object and update vuln_ns and vuln_ew
            new_board = copy.copy(board)
            new_board.vuln_ns, new_board.vuln_ew = missing_combination
            if (i+1) % 100000 == 0:
                sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {i+1}\n')
                sys.stderr.flush()
            i += 1
            #print(f"Adding {new_board.vuln_ns} {new_board.vuln_ew} {new_board.auction}")
            sorted_filtered_deals.append(new_board)
        processed_auctions.add(auction)

    sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} After add missing vuln {len(sorted_filtered_deals)}\n')

    sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Creating arrays with deals\n')

    sorted_filtered_deals = sorted(sorted_filtered_deals, key=lambda x: (x.auction))

    x, y, z, HCP, SHAPE = create_arrays(ns, ew, players_pr_hand, len(sorted_filtered_deals), 4 if version == "2" else 3, alert_supported=alert_supported, n_cards=n_cards )

    x, y, z, HCP, SHAPE, key_counts, k = create_binary(sorted_filtered_deals, ns, ew, alternating, x, y, z, HCP, SHAPE, key_counts, k, 4 if version == "2" else 3, alert_supported=alert_supported)

    sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} created {k} hands\n')

    np.save(os.path.join(out_dir, 'x.npy'), x)
    np.save(os.path.join(out_dir, 'y.npy'), y)
    np.save(os.path.join(out_dir, 'z.npy'), z)
    np.save(os.path.join(out_dir, 'HCP.npy'), HCP)
    np.save(os.path.join(out_dir, 'SHAPE.npy'), SHAPE)


    sys.stderr.write(f"Saved to {out_dir}")
    # Get unique keys and sort them
    unique_sorted_keys = sorted(set({auction[4:-1] for auction in auctions}))

    # Print all unique keys sorted
    for key in unique_sorted_keys:
        print(key)