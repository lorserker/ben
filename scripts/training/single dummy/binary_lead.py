import itertools
import os
import sys
sys.path.append('../../../src')
import numpy as np

from bidding import bidding

from lead_binary_util import DealMeta, convert_auction, encode_card
from data_access import card_index_lookup

from nn.bid_info import BidInfo


class DealData(object):

    def __init__(self, meta, dealer, vuln_ns, vuln_ew, hands, auction, lead_card_str):
        self.meta = meta
        self.dealer = dealer
        self.vuln_ns = vuln_ns
        self.vuln_ew = vuln_ew
        self.hands = list(map(parse_hand_f(32), hands))
        self.shapes = list(map(lambda shape: (shape - 3.25)/1.75, map(get_shape, hands)))
        self.hcp = list(map(lambda point_count: (np.array([[point_count]]) - 10) / 4, map(get_hcp, hands)))
        self.auction = auction
        self.lead_card_str = lead_card_str

    @classmethod
    def from_deal_meta_auction_play_string(cls, deal_str, meta_str, auction_str, play_str):
        dealer = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        vuln = {'NS': (True, False), 'EW': (False, True), '-': (False, False), 'ALL': (True, True)}
        hands_wnes = deal_str[2:].strip().split()
        hands = hands_wnes[1:] + hands_wnes[:1]

        deal_meta = DealMeta.from_str(meta_str)
        dealer_ix = dealer[deal_meta.dealer]

        auction = convert_auction(auction_str)
        auction = (['PAD_START'] * dealer_ix) + auction

        vuln_ns, vuln_ew = vuln[deal_meta.vuln]

        lead_card_str = play_str[:2]

        return cls(deal_meta, dealer_ix, vuln_ns, vuln_ew, hands, auction, lead_card_str)


    def get_binary(self, n_steps=8):
        A = np.zeros((1, n_steps, 2 + 1 + 4 + 32 + 3 * 40), dtype=np.float16)
        X = np.zeros((1, 1 + 5 + 1 + 1 + 1 + 1 + 32), dtype=np.float16)
        y = np.zeros((1, 32), dtype=np.float16)

        padded_auction = self.auction + (['PAD_END'] * 4 * n_steps)

        times_seen = [0, 0, 0, 0]

        declarer_ix = {'N':0, 'E':1, 'S':2, 'W':3}[self.meta.declarer]
        leader_ix = (declarer_ix + 1) % 4

        i = 0
        while sum(times_seen) < 4 * n_steps:
            if padded_auction[i] == 'PAD_START':
                i += 1
                continue

            hand_ix = i % 4

            t = times_seen[hand_ix]
        
            if hand_ix == leader_ix:
                # we only generate data for the opening lead hand
                v_we = self.vuln_ns if hand_ix % 2 == 0 else self.vuln_ew
                v_them = self.vuln_ew if hand_ix % 2 == 0 else self.vuln_ns
                vuln = np.array([[v_we, v_them]], dtype=np.float32)
                hcp = self.hcp[hand_ix]
                shape = self.shapes[hand_ix]
                
                lho_bid = padded_auction[i - 3] if i - 3 >= 0 else 'PAD_START'
                partner_bid = padded_auction[i - 2] if i - 2 >= 0 else 'PAD_START'
                rho_bid = padded_auction[i - 1] if i - 1 >= 0 else 'PAD_START'
                target_bid = padded_auction[i]

                ftrs = np.concatenate((
                    vuln,
                    hcp,
                    shape,
                    self.hands[hand_ix],
                    bidding.encode_bid(lho_bid),
                    bidding.encode_bid(partner_bid),
                    bidding.encode_bid(rho_bid)
                ), axis=1)

                A[0, t, :] = ftrs

            times_seen[hand_ix] += 1
            i += 1

        for n in times_seen:
            assert n == n_steps

        #- 0 = level
        #- 1,2,3,4,5 = strain one-hot (N, S, H, D, C)
        #- 6 = doubled
        #- 7 = redoubled
        #- 8 = vuln leader
        #- 9 = vuln declarer
        #- 10:42 = cards themselves one-hot encoded, 32 cards. 7 and smaller are treated as x
        X[0, 0] = int(self.meta.level)
        X[0, 1 + 'NSHDC'.index(self.meta.strain)] = 1
        X[0, 6] = int(self.meta.doubled)
        X[0, 7] = int(self.meta.redoubled)
        X[0, 8:10] = A[0, 0, 0:2]  # vulnerability us vs them is the same as in the auction
        X[0, 10:42] = self.hands[leader_ix].reshape(32)

        y[0, :] = encode_card(self.lead_card_str)

        return A, X, y

def get_card_index(card, n_cards):
    assert(n_cards % 4 == 0)
    x_card_index = n_cards // 4 - 1
    if card not in card_index_lookup:
        return x_card_index
    return min(card_index_lookup[card], x_card_index)


def parse_hand_f(n_cards):
    def f(hand_str):
        x = np.zeros((1, n_cards))
        suits = hand_str.split('.')
        assert(len(suits) == 4)
        for suit_index in [0, 1, 2, 3]:
            for card in suits[suit_index]:
                card_index = get_card_index(card, n_cards)
                x[0, suit_index * n_cards // 4 + card_index] += 1
        return x
    return f

def get_shape(hand):
    suits = hand.split('.')
    return np.array([len(suit) for suit in suits]).reshape((1, 4))

def get_hcp(hand):
    hcp = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
    return sum([hcp.get(c, 0) for c in hand])


def create_binary(data_it, out_dir, model, n_steps=8):
    data_it, copy = itertools.tee(data_it)  # Create a copy of the iterator
    n = sum(1 for _ in copy)  # Count items in the copy

    A = np.zeros((n, n_steps, 2 + 1 + 4 + 32 + 3 * 40), dtype=np.float16)
    B = np.zeros((n, 15), dtype=np.float16)
    X = np.zeros((n, 42), dtype=np.float16)
    y = np.zeros((n, 32), dtype=np.float16)
    
    for i, (deal_str, meta_str, auction_str, play_str) in enumerate(data_it):
        if (i+1) % 1000 == 0:
            print(i+1)
        deal_data = DealData.from_deal_meta_auction_play_string(deal_str, meta_str, auction_str, play_str)
        a_part, x_part, y_part = deal_data.get_binary(n_steps)
        
        A[i:i+1, :, :] = a_part
        X[i:i+1, :] = x_part
        y[i:i+1, :] = y_part

        p_hcp, p_shp = model.model(a_part)

        b = np.zeros(15)
        b[:3] = p_hcp.reshape((-1, n_steps, 3))[:,-1,:].reshape(3)
        b[3:] = p_shp.reshape((-1, n_steps, 12))[:,-1,:].reshape(12)
        B[i:i+1, :] = b


    np.save(os.path.join(out_dir, 'A.npy'), A)
    np.save(os.path.join(out_dir, 'B.npy'), B)
    np.save(os.path.join(out_dir, 'X.npy'), X)
    np.save(os.path.join(out_dir, 'y.npy'), y)


def test_debug():
    deal_data = DealData.from_deal_meta_auction_play_string(
        'W:KQ75.A.K54.J9872 T642.9432.QJ632. AJ8.J76.T9.A6543 93.KQT85.A87.KQT',
        'E NS 5CX.-1.E',
        'PP 1H DD 3H 4C PP 5C PP PP DD PP PP PP',
        'HKHAH2H6C2D6CACTH7H8C7H4CJH3C3CQS9S5S2SAHJH5C8H9S7S4SJS3S8HTSQS6SKSTD9CKDAD4D2DTD8D5DJC4C6D7C9D3DKDQC5HQ'
    )
    print(deal_data.dealer)
    print(deal_data.vuln_ns)
    print(deal_data.vuln_ew)
    for hand in deal_data.hands:
        print(hand.reshape((4, 8)))
    print(deal_data.shapes)
    print(deal_data.hcp)
    print(deal_data.auction)
    print(deal_data.lead_card_str)

    A, X, y = deal_data.get_binary()

    import pdb; pdb.set_trace()


def jack_data_iterator(fin):
    '''
    yields (deal_str, meta_str, auction_str, play_str)
    '''
    lines = []
    for line in fin:
        lines.append(line.strip())
        if len(lines) == 4:
            yield tuple(lines)
            lines = []

if __name__ == '__main__':

    out_dir = './lead_bin'

    data_it = jack_data_iterator(itertools.chain(
        open('../data/Jack/BW5C_N.txt'), 
        open('../data/Jack/BW5C_S.txt'))) 

    model = BidInfo("../bidding_info/model/binfo-51000")

    create_binary(data_it, out_dir, model, 8)
