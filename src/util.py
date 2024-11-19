import numpy as np
import re
import hashlib

from typing import NamedTuple, List

from bidding import bidding
from binary import get_cards_from_binary_hand, get_binary_hand_from_cards

def save_for_training(deal, auction):
    with open("training.ben", "a") as file:
        file.write(deal + " #" + auction + "\n")

VULN = {
    'None': [False, False],
    'N-S': [True, False],
    'E-W': [False, True],
    'Both': [True, True]
}

def find_vuln_text(boolean_array):
    for text, array in VULN.items():
        if array == boolean_array:
            return text
    return None  # If no match is found

def hand_to_str(hand, n_cards=32):
    x = hand.reshape((4, n_cards // 4))
    symbols = 'AKQJT98765432'
    symbols_list = list(symbols)  # Convert the string to a list of characters

    for i in range(13):
        if i >= (n_cards // 4) - 1:
            symbols_list[i] = 'x'  # Modify the character

    symbols = ''.join(symbols_list)

    suits = []
    for i in range(4):
        s = ''
        for j in range(n_cards // 4):
            if x[i,j] > 0:
                s += symbols[j] * int(x[i,j])
        suits.append(s)
    return '.'.join(suits)

def expected_tricks_sd(tricks_softmax):
    t_soft = tricks_softmax.reshape((-1, 14))
    result = np.zeros((t_soft.shape[0], 1))

    for i in range(t_soft.shape[1]):
        result = result + i * t_soft[:,i:i+1]
    return result

def p_defeat_contract(contract, tricks_softmax):
    level = int(contract[0])
    tricks_needed = level + 6

    declarerwins =  np.sum(tricks_softmax.reshape((-1, 14))[:,tricks_needed:], axis=1, keepdims=True)
    defeat = 1 - declarerwins
    return defeat

def view_samples(hand1, hand2):
    for i in range(min(5, hand1.shape[0])):
        h1str = hand_to_str(hand1[i].astype(int))
        h2str = hand_to_str(hand2[i].astype(int))

        print('%s\t%s' % (h1str, h2str))

def get_all_hidden_cards(visible_cards):
    all_cards_hand = np.array([
        [1, 1, 1, 1, 1, 1, 1, 6],
        [1, 1, 1, 1, 1, 1, 1, 6],
        [1, 1, 1, 1, 1, 1, 1, 6],
        [1, 1, 1, 1, 1, 1, 1, 6],
    ]).reshape(32)

    return get_cards_from_binary_hand(all_cards_hand - get_binary_hand_from_cards(visible_cards))

def display_lin(lin):
    from IPython.core.display import HTML
    
    return HTML(f'<iframe width="800" height="600" src="http://bridgebase.com/tools/handviewer.html?lin={lin}" />')


SUIT_MASK = np.array([
    [1] * 8 + [0] * 24,
    [0] * 8 + [1] * 8 + [0] * 16,
    [0] * 16 + [1] * 8 + [0] * 8,
    [0] * 24 + [1] * 8,
])

def follow_suit(cards_softmax, own_cards, trick_suit, n_cards=32):
    assert cards_softmax.shape[1] == n_cards
    assert own_cards.shape[1] == n_cards
    assert trick_suit.shape[1] == 4
    assert trick_suit.shape[0] == cards_softmax.shape[0]
    assert cards_softmax.shape[0] == own_cards.shape[0]

    suit_defined = np.max(trick_suit, axis=1) > 0
    trick_suit_i = np.argmax(trick_suit, axis=1)

    mask = (own_cards > 0).astype(np.int32)

    has_cards_of_suit = np.sum(mask * SUIT_MASK[trick_suit_i], axis=1) > 1e-9

    mask[suit_defined & has_cards_of_suit] *= SUIT_MASK[trick_suit_i[suit_defined & has_cards_of_suit]]

    legal_cards_softmax = cards_softmax * mask

    s = np.sum(legal_cards_softmax, axis=1, keepdims=True)
    s[s < 1e-9] = 1
    return legal_cards_softmax / s

def calculate_seed(input):
    # Calculate the SHA-256 hash
    hash_object = hashlib.sha256(input.encode())
    hash_bytes = hash_object.digest()

    # Convert the first 4 bytes of the hash to an integer and take modulus
    hash_integer = int.from_bytes(hash_bytes[:4], byteorder='big') % (2**32 - 1)
    return hash_integer

def convert_to_probability_with_weight(x, states, counts):
    """Compute weighted softmax values for each set of scores in x using counts."""
    
    # Initialize weights array with the correct shape
    weights = np.ones(states[0].shape[0])    
    
    # Calculate weights based on counts
    for i in range(states[0].shape[0]):
        sample = '%s %s %s %s' % (
            hand_to_str(states[0][i, 0, :32].astype(int)),
            hand_to_str(states[1][i, 0, :32].astype(int)),
            hand_to_str(states[2][i, 0, :32].astype(int)),
            hand_to_str(states[3][i, 0, :32].astype(int)),
        )
        
        # Ensure that you get a scalar count for each sample
        weights[i] = counts.get(sample, 0)  # Use .get to avoid KeyError

    # Apply weights to x before calculating probabilities
    # Ensure x is 2D: (num_samples, num_features) and weights is 1D: (num_samples,)
    weighted_x = x * weights
    sum_of_proba = np.sum(weighted_x, axis=0)

    # Avoid division by zero if sum_of_proba contains zeros
    x = np.divide(weighted_x, sum_of_proba, out=np.zeros_like(weighted_x), where=sum_of_proba != 0)
    return x

def convert_to_probability(x):
    """Compute softmax values for each sets of scores in x."""
    sum_of_proba = np.sum(x, axis=0)
    x =  np.divide(x, sum_of_proba)
    return x

class Board(NamedTuple):
    dealer: str
    vuln: List[bool]
    hands: List[str]
    auction: List[str]
    play: List[str]

def parse_lin(lin):
    rx_bid = r'mb\|([0-9a-zA-Z]+?)!?\|'
    rx_card = r'pc\|([C,D,H,S,c,d,h,s][2-9A,K,Q,J,T])\|'
    rx_hand = r'S(?P<S>[2-9A,K,Q,J,T]*?)H(?P<H>[2-9A,K,Q,J,T]*?)D(?P<D>[2-9A,K,Q,J,T]*?)C(?P<C>[2-9A,K,Q,J,T]*?)$'

    bid_trans = {
        'P': 'PASS',
        'D': 'X',
        'R': 'XX'
    }

    play = [card.upper() for card in re.findall(rx_card, lin)]
    auction = []
    for bid in re.findall(rx_bid, lin):
        bid = bid.upper()
        auction.append(bid_trans.get(bid, bid))
    
    vuln = [False, False]
    lin_vuln = re.findall(r'sv\|(.)\|', lin)[0]
    if lin_vuln == 'n':
        vuln = [True, False]
    elif lin_vuln == 'e':
        vuln = [False, True]
    elif lin_vuln == 'b':
        vuln = [True, True]

    lin_deal = re.findall(r'(?<=md\|)(.*?)(?=\|)', lin)[0]
    dealer = {'1': 'S', '2': 'W', '3': 'N', '4': 'E'}[lin_deal[0]]
    lin_hands = lin_deal[1:].split(',')

    hd_south = re.search(rx_hand, lin_hands[0]).groupdict()
    hd_west = re.search(rx_hand, lin_hands[1]).groupdict()
    hd_north = re.search(rx_hand, lin_hands[2]).groupdict()

    def seen_cards(suit):
        return set(hd_south[suit]) | set(hd_west[suit]) | set(hd_north[suit])

    hd_east = {suit: set('23456789TJQKA') - seen_cards(suit) for suit in 'SHDC'}

    def to_pbn(hd):
        return '.'.join([''.join(list(reversed(list(hd[suit])))) for suit in 'SHDC'])

    hands = [to_pbn(hd) for hd in [hd_north, hd_east, hd_south, hd_west]]

    return Board(dealer, vuln, hands, auction, play)

def get_play_status(hand, current_trick, strain):
    suits = np.array(hand).reshape((4, -1))
    shape = suits.sum(axis=1)
    if current_trick == [] or len(current_trick) == 4:
        return "Lead"
    suitlead = current_trick[0] // 13
    if shape[suitlead] == 0:
        if strain == 0:
            return "Discard"
        else:
            if shape[strain-1] == 0:
                return "Discard"
            else:
                return "Ruff"
    elif shape[suitlead] == 1:
        return "Forced"
    else:
        return "Follow"

def get_singleton(hand, current_trick):
    suits = np.array(hand).reshape((4, -1))
    suitlead = current_trick[0] // 13
    card = np.argmax(suits[suitlead])
    card52 = card + 13 * suitlead
    return card52

def check_sequence(arr, suitlead):
    # Get the indices where the array has 1's
    indices_of_ones = np.where(arr == 1)[0]

    if len(indices_of_ones) == 0:
        return -1, 0  # No 1's in the array

    # Check if the difference between consecutive indices is 1
    if np.all(np.diff(indices_of_ones) == 1):
        first_index = indices_of_ones[0]
        count_of_ones = len(indices_of_ones)
        high_card =  first_index + 13 * suitlead
        low_card =  first_index + count_of_ones - 1 + 13 * suitlead

        return high_card, low_card
    else:
        return -1, 0  # Not in sequence

def get_possible_cards(hand, current_trick):
    suits = np.array(hand).reshape((4, -1))
    suitlead = current_trick[0] // 13
    cards = suits[suitlead]
    result = check_sequence(cards, suitlead)
    return result
