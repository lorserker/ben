import numpy as np
import re

from typing import NamedTuple, List

from bidding import bidding
from binary import get_cards_from_binary_hand, get_binary_hand_from_cards

def hand_to_str(hand):
    x = hand.reshape((4, 8))
    symbols = 'AKQJT98x'
    suits = []
    for i in range(4):
        s = ''
        for j in range(8):
            if x[i,j] > 0:
                s += symbols[j] * x[i,j]
        suits.append(s)
    return '.'.join(suits)

def expected_tricks(tricks_softmax):
    t_soft = tricks_softmax.reshape((-1, 14))
    result = np.zeros((t_soft.shape[0], 1))

    for i in range(t_soft.shape[1]):
        result = result + i * t_soft[:,i:i+1]
    return result

def p_make_contract(contract, tricks_softmax):
    level = int(contract[0])
    tricks_needed = level + 6

    return np.sum(tricks_softmax.reshape((-1, 14))[:,tricks_needed:], axis=1, keepdims=True)

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


# TODO: maybe we don't need this function
def to_bbo_viewer(deal_data, tricks, trick_won_by):
    decl_i = bidding.get_decl_i(bidding.get_contract(deal_data.auction))

    nesw_indexes = [(decl_i + 1) % 4, (decl_i + 2) % 4, (decl_i + 3) % 4, decl_i]

    player_i = 0

    hands = [[[],[],[],[]], [[],[],[],[]], [[],[],[],[]], [[],[],[],[]]]

    play = []

    for trick, won_by in zip(tricks, trick_won_by):
        for card in trick:
            suit = card // 13
            rank = 'AKQJT98765432'[card % 13]
            hands[nesw_indexes[player_i]][suit].append(rank)
            play.append('SHDC'[suit] + rank)
            player_i = (player_i + 1) % 4
        player_i = won_by

    auction = []
    for call in deal_data.auction:
        if call == 'PAD_START':
            continue
        if call == 'PAD_END':
            break
        if call == 'PASS':
            auction.append('p')
        elif call == 'X':
            auction.append('d')
        elif call == 'XX':
            auction.append('r')
        else:
            auction.append(call.lower())

    def hand_to_bbo(hand):
        return 's{}h{}d{}c{}'.format(
            ''.join(hand[0]), ''.join(hand[1]), ''.join(hand[2]), ''.join(hand[3])
        )

    url = 'http://www.bridgebase.com/tools/handviewer.html'

    vuln = {
        (True, True): 'b',
        (True, False): 'n',
        (False, True): 'e',
        (False, False): '-'
    }[(deal_data.vuln_ns, deal_data.vuln_ew)]

    params = 'd={}&v={}&a={}&w={}&n={}&e={}&s={}&p={}'.format(
        'NESW'[deal_data.dealer],
        vuln,
        ''.join(auction),
        hand_to_bbo(hands[3]),
        hand_to_bbo(hands[0]),
        hand_to_bbo(hands[1]),
        hand_to_bbo(hands[2]),
        ''.join(play)
    )

    return url + '?' + params
