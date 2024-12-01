import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import sys
sys.path.append('../../../../src')

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
def get_binary_contract( deal, score, n_cards=32):
    X = np.zeros(2 + 2 * n_cards, dtype=np.float16)
    tricks_one_hot = np.zeros((1, 14), dtype=np.float16)
    hands = parse_hands(deal)
    #print(hands)
    #print(score)
    # Unpack the tuple into variables
    optimum_score, par_contract, vuln = score

    # Print the variables
    #print(f"Optimum Score: {optimum_score}")
    #print(f"Par Contract: {par_contract}")
    # Extract the value in quotes
    contract_data = par_contract.split('"')[1]
    contracts = contract_data.split('; ')
    try:
        for contract in contracts:
            direction, level, suit, doubled, trick_count = parse_contract(contract)
    except Exception as e:
        print(deal, score)
        print(e)
        return None, None, None, None, 0
    hand_str = hands["North"]
    dummy_str = hands["South"]
    if direction == 'E':
        hand_str = hands["East"]
        dummy_str = hands["West"]
    if direction == 'W':
        hand_str = hands["West"]
        dummy_str = hands["East"]
    if direction == 'EW':
        hand_str = hands["East"]
        dummy_str = hands["West"]
    #print(f"Contract: {contract}")
    #print(f"Vuln: {vuln}")
    print(hand_str, dummy_str, doubled, trick_count, direction, level, suit)
    vuln = np.array([vuln], dtype=np.float16)
    hand = binary.parse_hand_f(n_cards)(hand_str).reshape(n_cards)
    dummy = binary.parse_hand_f(n_cards)(dummy_str).reshape(n_cards)
    ftrs = np.concatenate((
        vuln,
        [hand],
        [dummy],
    ), axis=1)
    X = ftrs
    u = doubled
    tricks_one_hot[0, trick_count] = 1
    z = tricks_one_hot
    y = bidding.encode_bid(str(level)+str(suit))
    return X, y, u, z, 1

def create_binary(dataset, n, out_dir, n_cards=32):
    x = np.zeros((n, 2 + 2 * n_cards), dtype=np.float16)
    # Contract
    y = np.zeros((n, 40), dtype=np.uint8)
    # Doubled
    z = np.zeros((n, 1), dtype=np.uint8)
    # Tricks
    u = np.zeros((n, 14), dtype=np.uint8)
    k = 0

    for i, (key, value) in enumerate(dataset.items()):
        #print(key, value)

        if (i+1) % 10000 == 0:
            sys.stderr.write(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {i+1}\n')
            sys.stderr.flush()
        x_part, y_part, z_part, u_part, count = get_binary_contract(key, value, n_cards)
        if count > 0:
            x[k] = x_part
            y[k] = y_part
            z[k] = z_part
            u[k] = u_part
            k += count

    np.save(os.path.join(out_dir, 'x.npy'), x)
    np.save(os.path.join(out_dir, 'y.npy'), y)
    np.save(os.path.join(out_dir, 'z.npy'), z)
    np.save(os.path.join(out_dir, 'u.npy'), u)

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
        print("Usage: python contract_binary.py inputfile outputdirectory n_cards=32")
        print("The input file is the BEN-format (1 line with hands, and next line with the contract/Bidding).")
        print("n_cards is the number of cards in the deck")
        sys.exit(1)

    infnm = sys.argv[1] # file where the data is
    outdir = sys.argv[2]
    n_cards = next((extract_value_cmd(arg) for arg in sys.argv[3:] if arg.startswith("n_cards=")), 32)

    dataset = load_optimumscores(infnm)
    #for key, value in dataset.items():
    #    print(f"{key}: {value}")
    n = len(dataset)
    print(f"Loading {n} deals")

    create_binary(dataset, n, outdir, int(n_cards))

