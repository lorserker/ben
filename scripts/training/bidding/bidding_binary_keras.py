import os
import sys
sys.path.append('../../../src')
import copy

import datetime
import multiprocessing
from collections import Counter, defaultdict
import numpy as np
from bidding import bidding

from bidding.binary import DealData
from pathlib import Path
import re


np.set_printoptions(precision=2, suppress=True, linewidth=300,threshold=np.inf)

def load_pbn(fin):
    lines = []
    boards = [] 
    auction_lines = []
    inside_auction_section = False
    dealer, vulnerable = None, None
    for line in fin:
        if line.startswith("% PBN") or line == "\n":
            if dealer != None:
                board = {
                    'deal': ' '.join(hands_nesw),      
                    'auction': dealer + " " + vulnerable + " " + ' '.join(auction_lines)
                }
                boards.append(board)            
                auction_lines = []
                dealer= None
        if line.startswith('[Dealer'):
            dealer = extract_value(line)
        if line.startswith('[Vulnerable'):
            vuln_str = extract_value(line)
            vulnerable = {'NS': 'N-S', 'EW': 'E-W', 'All': 'Both'}.get(vuln_str, vuln_str)
        if line.startswith('[Deal '):
            hands_pbn = extract_value(line)
            [seat, hands] = hands_pbn.split(':')
            hands_nesw = [''] * 4
            first_i = 'NESW'.index(seat)
            for hand_i, hand in enumerate(hands.split()):
                hands_nesw[(first_i + hand_i) % 4] = hand
        if line.startswith('[Auction'):
            inside_auction_section = True
            continue  
        if inside_auction_section:
            if line.startswith('[') or line == "\n":  # Check if it's the start of the next tag
                inside_auction_section = False
            else:
                # Convert bids
                line = line.strip().upper().replace('.','').replace("NT","N").replace("PASS","P").replace("DOUBLE","X").replace("REDOUBLE","XX").replace('AP','P P P')
                # Remove extra spaces
                line = re.sub(r'\s+', ' ', line)
                # update alerts
                line = re.sub(r' =\d{1,2}=', '*', line)
                auction_lines.append(line)  

        else:
            continue
    if dealer != None:
        board = {
            'deal': ' '.join(hands_nesw),      
            'auction': dealer + " " + vulnerable + " " + ' '.join(auction_lines)
        }

        boards.append(board)      
    
    for board in boards:
        lines.append(board['deal'])
        lines.append(board['auction'])

    return lines

def get_binary_hcp_shape(deal, ns, ew, n_steps=8):
    X = np.zeros((4, n_steps, 2 + 2 + 1 + 4 + deal.n_cards + 4 * 40), dtype=np.float16)

    y = np.zeros((4, n_steps, 40), dtype=np.uint8)
    z = np.zeros((4, n_steps, 1), dtype=np.uint8)
    HCP = np.zeros((4, n_steps, 3), dtype=np.float16)
    SHAPE = np.zeros((4, n_steps, 12), dtype=np.float16)
    
    padded_auction = deal.auction + (['PAD_END'] * 4 * n_steps)

    #print(deal.auction)
    times_seen = [0, 0, 0, 0]

    i = 0
    while sum(times_seen) < 4 * n_steps:
        if padded_auction[i] == 'PAD_START':
            i += 1
            continue

        hand_ix = i % 4

        t = times_seen[hand_ix]
    
        v_we = deal.vuln_ns if hand_ix % 2 == 0 else deal.vuln_ew
        v_them = deal.vuln_ew if hand_ix % 2 == 0 else deal.vuln_ns
        vuln = np.array([[v_we, v_them]], dtype=np.float16)
        hcp = deal.hcp[hand_ix]
        shape = deal.shapes[hand_ix]
        
        my_bid = padded_auction[i - 4] if i - 4 >= 0 else 'PAD_START'
        lho_bid = padded_auction[i - 3] if i - 3 >= 0 else 'PAD_START'
        partner_bid = padded_auction[i - 2] if i - 2 >= 0 else 'PAD_START'
        rho_bid = padded_auction[i - 1] if i - 1 >= 0 else 'PAD_START'
        target_bid = padded_auction[i]

        ftrs = np.concatenate((
            np.array([ns, ew], ndmin=2),
            vuln,
            hcp,
            shape,
            deal.hands[hand_ix],
            bidding.encode_bid(my_bid),
            bidding.encode_bid(lho_bid),
            bidding.encode_bid(partner_bid),
            bidding.encode_bid(rho_bid)
        ), axis=1)

        X[hand_ix, t, :] = ftrs
        y[hand_ix, t, :] = bidding.encode_bid(target_bid, False)
        z[hand_ix, t, :] = 1 if "*" in target_bid else 0

        HCP[hand_ix, t, 0] = deal.hcp[(hand_ix - 3) % 4][0,0]
        HCP[hand_ix, t, 1] = deal.hcp[(hand_ix - 2) % 4][0,0]
        HCP[hand_ix, t, 2] = deal.hcp[(hand_ix - 1) % 4][0,0]

        SHAPE[hand_ix, t, 0:4] = deal.shapes[(hand_ix - 3) % 4][0]
        SHAPE[hand_ix, t, 4:8] = deal.shapes[(hand_ix - 2) % 4][0]
        SHAPE[hand_ix, t, 8:12] = deal.shapes[(hand_ix - 1) % 4][0]


        times_seen[hand_ix] += 1
        i += 1

    for n in times_seen:
        assert n == n_steps

    return X, y, HCP, SHAPE, z


def load_deals(fin, n_cards=32, rotate=False):
    deal_str = ''

    for line_number, line in enumerate(fin):
        # For now we remove alerts until we have an useable implementation
        # line = line.strip().replace("*",'')
        if line_number % 2 == 0:
            deal_str = line
        else:
            yield DealData.from_deal_auction_string(deal_str, line, "", ns, ew, n_cards, rotate=False)

def create_arrays(n, n_cards=32):
    rows_pr_hand = 4
    x = np.zeros((rows_pr_hand * n, 8, 2 + 2 + 1 + 4 + n_cards + 4 * 40), dtype=np.float16)
    y = np.zeros((rows_pr_hand * n, 8, 40), dtype=np.uint8)
    z = np.zeros((rows_pr_hand * n, 8, 1), dtype=np.uint8)

    HCP = np.zeros((rows_pr_hand * n, 8, 3), dtype=np.float16)
    SHAPE = np.zeros((rows_pr_hand * n, 8, 12), dtype=np.float16)
    return x, y, z, HCP, SHAPE

def count_deals(data_it, max_occurrences, auctions, key_counts, deals):
    
    sys.stderr.write(f'Counting deals with same auction\n')
    for i, deal_data in enumerate(data_it):
        if (i+1) % 100000 == 0:
            sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {i+1}\n')
            sys.stderr.flush()
        v = 0
        if deal_data.vuln_ns and not deal_data.vuln_ew: v = 1
        if not deal_data.vuln_ns and deal_data.vuln_ew: v = 2
        if deal_data.vuln_ns and deal_data.vuln_ew: v = 3
        auction = f"{ew}-{ns} {' '.join([x.replace('PASS', 'P') for x in deal_data.auction if x != 'PAD_START'])} {v}"
        #print(auction)
        #print(deal_data.deal_str)
        
        if key_counts[auction] < max_occurrences:
            auctions.append(auction)
            key_counts[auction] += 1
            deals.append(deal_data)
        #else:
        #    print("Skipping duplicate", auction)
    return deals, auctions, key_counts

def create_binary(data_it, ns, ew, alternating, x, y, z, HCP, SHAPE, k):

    sys.stderr.write(f'Creating binary data for {len(data_it)} deals\n')
    for i, deal_data in enumerate(data_it):
        #if i == 0:
        #    print(deal_data, deal_data.auction)
        if (i) % 100000 == 0:
            sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {i} {deal_data.auction}\n')
            sys.stderr.flush()
        if alternating and (i % 2) == 1:
            x_part, y_part, hcp_part, shape_part, alert_part = get_binary_hcp_shape(deal_data, ew, ns)
        else:
            x_part, y_part, hcp_part, shape_part, alert_part = get_binary_hcp_shape(deal_data, ns, ew)
        #print(deal_data.deal_str)
        #print(deal_data.auction)
        #for k in range(4):
        #    print(x_part[k], hcp_part[k], shape_part[k], y_part[k], alert_part[k])
        
        x[k:k+4] = x_part
        y[k:k+4] = y_part
        z[k:k+4] = alert_part
        HCP[k:k+4] = hcp_part
        SHAPE[k:k+4] = shape_part
        k += 4

    return x, y, z, HCP, SHAPE, k

# --- The same work spread over worker processes -------------------------------------------
# Parsing the deals and building the arrays is per-deal Python, about 0.6 ms a deal in one
# process. The workers get slices of the input lines; the parent only applies the limit of
# boards per auction, in file order, so the result is the same as from one process.

CHUNK_DEALS = 4000

def auction_key(deal_data, ns, ew):
    v = 0
    if deal_data.vuln_ns and not deal_data.vuln_ew: v = 1
    if not deal_data.vuln_ns and deal_data.vuln_ew: v = 2
    if deal_data.vuln_ns and deal_data.vuln_ew: v = 3
    return f"{ew}-{ns} {' '.join([x.replace('PASS', 'P') for x in deal_data.auction if x != 'PAD_START'])} {v}"

def load_pbn_parallel(pool, pbnlines, n_parts):
    # load_pbn on parts of the file that each start after the empty line ending a board
    cuts = [0]
    size = max(1, len(pbnlines) // n_parts)
    i = size
    while i < len(pbnlines):
        while i < len(pbnlines) and pbnlines[i] != "\n":
            i += 1
        if i + 1 < len(pbnlines):
            cuts.append(i + 1)
        i += size
    cuts.append(len(pbnlines))
    lines = []
    for part in pool.imap(load_pbn, [pbnlines[a:b] for a, b in zip(cuts, cuts[1:])]):
        lines.extend(part)
    return lines

def keys_chunk(args):
    lines, ns, ew, n_cards = args
    return [auction_key(DealData.from_deal_auction_string(lines[j], lines[j + 1], "", ns, ew, n_cards, rotate=False), ns, ew)
            for j in range(0, len(lines), 2)]

def count_deals_parallel(pool, lines, ns, ew, n_cards, max_occurrences):
    # Returns the indexes of the deals to keep, their auction keys and the count per key
    n_deals = len(lines) // 2
    sys.stderr.write('Counting deals with same auction\n')
    tasks = [(lines[2 * a:2 * min(a + CHUNK_DEALS, n_deals)], ns, ew, n_cards) for a in range(0, n_deals, CHUNK_DEALS)]
    kept, auctions, key_counts = [], [], Counter()
    i = 0
    for keys in pool.imap(keys_chunk, tasks):
        for auction in keys:
            if key_counts[auction] < max_occurrences:
                auctions.append(auction)
                key_counts[auction] += 1
                kept.append(i)
            i += 1
    return kept, auctions, key_counts

def binary_chunk(args):
    lines, first, ns, ew, n_cards, alternating, out_dir = args
    n = len(lines) // 2
    x, y, z, HCP, SHAPE = create_arrays(n, n_cards)
    for j in range(n):
        deal_data = DealData.from_deal_auction_string(lines[2 * j], lines[2 * j + 1], "", ns, ew, n_cards, rotate=False)
        if alternating and ((first + j) % 2) == 1:
            parts = get_binary_hcp_shape(deal_data, ew, ns)
        else:
            parts = get_binary_hcp_shape(deal_data, ns, ew)
        x[4 * j:4 * j + 4], y[4 * j:4 * j + 4], HCP[4 * j:4 * j + 4], SHAPE[4 * j:4 * j + 4], z[4 * j:4 * j + 4] = parts
    # One write per array at this slice's place in the .npy file
    for name, array in (('x', x), ('y', y), ('z', z), ('HCP', HCP), ('SHAPE', SHAPE)):
        path = os.path.join(out_dir, name + '.npy')
        data_start = np.load(path, mmap_mode='r').offset
        with open(path, 'r+b') as f:
            f.seek(data_start + 4 * first * array[0].nbytes)
            f.write(array.data)
    return n

def create_binary_parallel(pool, lines, kept, ns, ew, alternating, n_cards, out_dir):
    # Writes x, y, z, HCP and SHAPE straight into .npy files in out_dir; returns the number of hands
    n = len(kept)
    sys.stderr.write(f'Creating binary data for {n} deals\n')
    os.makedirs(out_dir, exist_ok=True)
    shapes = {'x': ((8, 2 + 2 + 1 + 4 + n_cards + 4 * 40), np.float16), 'y': ((8, 40), np.uint8), 'z': ((8, 1), np.uint8),
              'HCP': ((8, 3), np.float16), 'SHAPE': ((8, 12), np.float16)}
    for name, (shape, dtype) in shapes.items():
        array = np.lib.format.open_memmap(os.path.join(out_dir, name + '.npy'), mode='w+', dtype=dtype, shape=(4 * n,) + shape)
        del array
    tasks = []
    for a in range(0, n, CHUNK_DEALS):
        chunk_lines = []
        for i in kept[a:a + CHUNK_DEALS]:
            chunk_lines.append(lines[2 * i])
            chunk_lines.append(lines[2 * i + 1])
        tasks.append((chunk_lines, a, ns, ew, n_cards, alternating, out_dir))
    done = 0
    for count in pool.imap_unordered(binary_chunk, tasks):
        done += count
    return 4 * done

# Function to extract value from command-line argument
def extract_value_cmd(arg):
    return arg.split('=')[1]

def extract_value(s: str) -> str:
    return s[s.index('"') + 1 : s.rindex('"')]

def to_numeric(value, default=0):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return default

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print("Usage: python bidding_binary_keras.py inputfile inputfile2 outputdirectory NS=<x> EW=<y> alternate=True n_cards=32 rotate=true alert_supported=True max_occurrences=<z> workers=<n>")
        print("The input file is the BEN-format or PBN-format")
        print("Use None if no inputfile2")
        print("alternate is signaling, that the input file has both open and closed room, so NS/EW will be alternated")
        print("n_cards is the number of cards in the deck")
        print("Setting rotate to true will rotate all deals, so North is first to bid")
        print("Setting alert_supported to true will record alert in the output")
        print("Setting max_occurrences will limit the number of boards per auction")
        print("workers is the number of processes (default: half the logical CPUs, at most 64; 1 for a single process)")
        sys.exit(1)

    infnm1 = sys.argv[1] # file where the data is
    if sys.argv[2] != "None":
        infnm2 = sys.argv[2] # file where the data is
    else:
        infnm2 = None
    out_dir = sys.argv[3]

    # Extract NS and EW values from command-line arguments if provided
    ns = next((extract_value_cmd(arg) for arg in sys.argv[3:] if arg.startswith("NS=")), 1)
    ew = next((extract_value_cmd(arg) for arg in sys.argv[3:] if arg.startswith("EW=")), 1)
    n_cards = next((extract_value_cmd(arg) for arg in sys.argv[3:] if arg.startswith("n_cards=")), 24)
    alternating = next((extract_value_cmd(arg) for arg in sys.argv[3:] if arg.startswith("alternate=")), False)
    rotate = next((extract_value_cmd(arg) for arg in sys.argv[3:] if arg.startswith("rotate=")), False)
    max_occurrences = next((extract_value_cmd(arg) for arg in sys.argv[3:] if arg.startswith("max_occurrences=")), 25)
    alert_supported = next((extract_value_cmd(arg) for arg in sys.argv[3:] if arg.startswith("alert_supported=")), False)
    workers = to_numeric(next((extract_value_cmd(arg) for arg in sys.argv[3:] if arg.startswith("workers=")), max(1, min(64, (os.cpu_count() or 2) // 2))), 1)
    version = 3
    add_missing_vulns = False
    max_filler_occurrences = 0
    sys.stderr.write(f"n_cards={n_cards}, NS={ns}, EW={ew}, Alternating={alternating}, Version={version}, alert_supported={alert_supported}, rotate={rotate}, outdir={out_dir},  max_occurrences={max_occurrences}, max_filler_occurrences={max_filler_occurrences}\n")
    max_occurrences = to_numeric(max_occurrences, 100)
    ns = to_numeric(ns)
    ew = to_numeric(ew)
    n_cards = to_numeric(n_cards, 24)

    
    # Worker processes when there is one input file; with a second (filler) file, a single process
    parallel = workers > 1 and infnm2 is None and not add_missing_vulns
    if parallel:
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Using {workers} processes\n')
        pool = multiprocessing.Pool(workers)

    # Count the occurrences of each line
    key_counts = Counter()
    auctions = []
    k = 0

    file_path = Path(infnm1)
    file_extension = file_path.suffix
    if file_extension == ".pbn":
        with open(infnm1, "r", encoding='utf-8') as file:  # Open the input file with UTF-8 encoding
            pbnlines = file.readlines()
            lines = None
            if parallel:
                try:
                    lines = load_pbn_parallel(pool, pbnlines, 4 * workers)
                except Exception as e:   # a board that leans on tags of the board before it
                    sys.stderr.write(f'Reading the PBN file in parts failed ({e}), reading it in one process\n')
            if lines is None:
                lines = load_pbn(pbnlines)
    else:
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

    if parallel:
        kept, auctions, key_counts = count_deals_parallel(pool, lines, ns, ew, n_cards, max_occurrences)
        filtered_deals = kept
    else:
        data_it = load_deals(lines, n_cards, rotate)
        filtered_deals, auctions, key_counts = count_deals(data_it, max_occurrences, auctions, key_counts, [])

    # Create a new Counter to count all occurrences without limit
    sys.stderr.write(f"Loaded {len(auctions)} auctions\n")   
    key_counts2 = Counter(auctions)
    
    # Sort by count
    sorted_keys_by_count = sorted(key_counts.items(), key=lambda x: (-x[1], x[0]))  # Sort by count and key in descending order

    count = len(sorted_keys_by_count)
    sys.stderr.write(f"Found {count} bidding sequences.\n" )
    sys.stderr.write(f"Filtered_deals {len(filtered_deals)}\n")
    sys.stderr.write(f"Removed {len(lines) // 2 - len(filtered_deals)} deals where the same bidding was seen more than {max_occurrences} times.\n")
    
    # The second training dataset is random deals, and we just use the bidding sequences we don't already have
    if infnm2:
        with open(infnm2, 'r') as file:

            print(f"Loading {infnm2}")
            lines = file.readlines()
            # Remove comments at the beginning of the file
            lines = [line for line in lines if not line.strip().startswith('#')]
            sys.stderr.write(f"Loading {len(lines) // 2} deals for fillers\n")
            data_it = load_deals(lines, n_cards, rotate)
            filtered_deals, auctions, key_counts = count_deals(data_it, max_filler_occurrences, auctions, key_counts, filtered_deals)

            # Create a new Counter to count all occurrences without limit
            sys.stderr.write(f"Loaded {len(auctions)} auctions from second dataset\n")   
            key_counts = Counter(auctions)
            
            count = len(sorted_keys_by_count)
            sys.stderr.write(f"Found {count} bidding sequences.\n" )

            sys.stderr.write(f"Filtered_deals {len(filtered_deals)}\n")


    if add_missing_vulns:
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
                    sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} added {i+1}\n')
                    sys.stderr.flush()
                i += 1
                #print(f"Adding {new_board.vuln_ns} {new_board.vuln_ew} {new_board.deal_str}")
                sorted_filtered_deals.append(new_board)
            processed_auctions.add(auction)

        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} added {i+1} missing combinations\n')
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} After add missing vuln {len(sorted_filtered_deals)}\n')


        filtered_deals = sorted(sorted_filtered_deals, key=lambda x: (x.auction))

    sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Creating arrays with deals\n')

    if parallel:
        k = create_binary_parallel(pool, lines, kept, ns, ew, bool(alternating), n_cards, out_dir)
        pool.close()
        pool.join()
        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} created {k} hands\n')
    else:
        x, y, z, HCP, SHAPE = create_arrays(len(filtered_deals), n_cards)
        x, y, z, HCP, SHAPE, k = create_binary(filtered_deals, ns, ew, alternating, x, y, z, HCP, SHAPE,  k)

        sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} created {k} hands\n')

        os.makedirs(out_dir, exist_ok=True)
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