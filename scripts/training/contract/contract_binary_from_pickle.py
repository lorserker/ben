import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
sys.path.append('../../../src')

import datetime
import os.path
import numpy as np
import pickle
import binary
from bidding import bidding


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

def parse_contract(contract):
    #print("Contract: ",contract)
    # Split the string into parts
    if 'X' in contract:
        doubled = True
        contract = contract.replace('X', '')
    else:
        doubled = False
    x = contract.split()
    direction = x[0]  # First two characters
    level = x[1][0:1]  # Level and suit (characters 3 and 4)
    suit = x[1][1:2]  # Level and suit (characters 3 and 4)
    trick_count_str = x[1][2:]  # Remaining part (trick count or "=")

    try:
        # Determine trick count
        if trick_count_str == "=":
            trick_count = int(level) + 6
        else:
            trick_count = int(level) + 6 +int(trick_count_str)  # Convert trick count to integer
    except ValueError:
        print("Invalid trick count:", trick_count_str, contract)
        trick_count = 0  # Invalid trick count, set to 0
        raise ValueError
    # Return parsed components
    return direction, level, suit, doubled, trick_count

def parse_hands(deal):
    # Extract the portion inside the quotes
    hands_data = deal.split('"')[1]  # Get the part inside the quotes
    hands = hands_data.split(':')[1]  # Remove the direction prefix (e.g., "N:")
    
    # Split the hands into North, East, South, and West
    hand_list = hands.split()
    hands_dict = {
        "North": hand_list[0],
        "East": hand_list[1],
        "South": hand_list[2],
        "West": hand_list[3],
    }
    
    return hands_dict
def get_binary_contract(deal, score, n_cards):
    """Processes a deal and score to generate binary contract data."""
    try:
        hands = parse_hands(deal)
        _, par_contract, vuln = score
        vuln_array = np.array([vuln], dtype=np.float16)
        contracts = par_contract.split('"')[1].split('; ')

        results = []

        for contract in contracts:
            #print(contract)
            if contract == "NSEW PASS":
                continue
            direction, level, suit, doubled, trick_count = parse_contract(contract)
            suit_one_hot = np.zeros(5, dtype=np.float16)
            strain_i = 'NSHDC'.index(suit)
            suit_one_hot[strain_i] = 1
            tricks_one_hot = np.zeros(14, dtype=np.float16)
            tricks_one_hot[trick_count] = 1

            # Encode bid as one-hot vector
            encoded_bid = bidding.encode_bid(f"{level}{suit}")

            for seat in direction:
                hand, dummy = get_hands(hands, seat)
                features = build_features(hand, dummy, vuln_array, n_cards)

                results.append((features, encoded_bid, doubled, suit_one_hot, tricks_one_hot))

        return results

    except ValueError as ve:
        print(f"Value error in parsing deal or contract: {deal}, {score}")
        raise ve
    except Exception as e:
        print(f"Unexpected error in deal processing: {deal}, {score}")
        raise e


def get_hands(hands, seat):
    """Fetches hand and dummy strings based on the seat."""
    mapping = {
        "N": ("North", "South"),
        "S": ("South", "North"),
        "E": ("East", "West"),
        "W": ("West", "East")
    }
    hand_str = hands[mapping[seat][0]]
    dummy_str = hands[mapping[seat][1]]
    return hand_str, dummy_str


def build_features(hand_str, dummy_str, vuln, n_cards):
    """Builds the feature vector for the given hand and dummy."""
    hand = binary.parse_hand_f(n_cards)(hand_str).reshape(n_cards)
    dummy = binary.parse_hand_f(n_cards)(dummy_str).reshape(n_cards)
    return np.concatenate((vuln, [hand], [dummy]), axis=1)

def create_binary(dataset, out_dir, n_cards):
    # Collect results dynamically in lists
    x_list, y_list, z_list, s_list, u_list = [], [], [], [], []

    for i, (key, value) in enumerate(dataset.items()):
        if (i + 1) % 10000 == 0:
            sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {i+1}\n')
            sys.stderr.flush()

        # Get binary contract data
        result = get_binary_contract(key, value, n_cards)

        # Append results to the lists
        for x_part, y_part, z_part, s_part, u_part in result:
            x_list.append(x_part[0])
            y_list.append(y_part[0])
            z_list.append([z_part])
            s_list.append(s_part)
            u_list.append(u_part)

    # Convert lists to NumPy arrays
    x = np.array(x_list, dtype=np.float16)
    y = np.array(y_list, dtype=np.uint8)
    z = np.array(z_list, dtype=np.uint8)
    s = np.array(s_list, dtype=np.uint8)
    u = np.array(u_list, dtype=np.uint8)

    # Save the arrays to disk
    np.save(os.path.join(out_dir, 'x.npy'), x)
    np.save(os.path.join(out_dir, 'y.npy'), y)
    np.save(os.path.join(out_dir, 'z.npy'), z)
    np.save(os.path.join(out_dir, 's.npy'), s)
    np.save(os.path.join(out_dir, 'u.npy'), u)

    print("crated hands:     ", x.shape)
    print("crated contracts: ", y.shape)
    print("crated doubled:   ", z.shape)
    print("crated suit:      ", s.shape)
    print("crated tricks:    ", u.shape)

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

def load_optimumscores(pickle_path):
    with open(pickle_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python contract_binary_from_pickle.py inputfile outputdirectory n_cards=24")
        print("The input file is a pickle file, with deals and par-score for each deal")
        print("n_cards is the number of cards in the deck. Default 24")
        sys.exit(1)

    infnm = sys.argv[1] # file where the data is
    outdir = sys.argv[2]
    n_cards = next((extract_value_cmd(arg) for arg in sys.argv[3:] if arg.startswith("n_cards=")), 24)

    dataset = load_optimumscores(infnm)
    #for key, value in dataset.items():
    #    print(f"{key}: {value}")
    n = len(dataset)
    print(f"Loading {n} deals")

    create_binary(dataset, outdir, int(n_cards))

