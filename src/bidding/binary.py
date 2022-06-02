import numpy as np

from bidding import bidding

CARD_INDEX_LOOKUP = dict(
    zip(
        ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'],
        range(13)
    )
)

class DealData(object):

    def __init__(self, dealer, vuln_ns, vuln_ew, hands, auction, n_cards=52):
        self.n_cards = n_cards
        self.dealer = dealer
        self.vuln_ns = vuln_ns
        self.vuln_ew = vuln_ew
        self.hands = hands
        self.shapes = list(map(lambda shape: (shape - 3.25)/1.75, map(get_shape, hands)))
        self.hcp = list(map(lambda point_count: (np.array([[point_count[0]]]) - 10) / 4, map(get_hcp, hands)))
        self.auction = auction

    @classmethod
    def from_deal_auction_string(cls, deal_str, auction_str, n_cards=52):
        dealer = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        vuln = {'N-S': (True, False), 'E-W': (False, True), 'None': (False, False), 'Both': (True, True)}
        hands = list(map(parse_hand_f(n_cards), deal_str.strip().split()))
        auction_parts = auction_str.strip().replace('P', 'PASS').split()
        dealer_ix = dealer[auction_parts[0]]
        vuln_ns, vuln_ew = vuln[auction_parts[1]]
        auction = (['PAD_START'] * dealer_ix) + auction_parts[2:]

        return cls(dealer_ix, vuln_ns, vuln_ew, hands, auction, n_cards)

    def reset_auction(self):
        self.auction = [bid for bid in self.auction if bid == 'PAD_START']

    def get_binary(self, n_steps=8):
        X = np.zeros((4, n_steps, 2 + 1 + 4 + self.n_cards + 3 * 40), dtype=np.float16)
        y = np.zeros((4, n_steps, 40), dtype=np.float16)

        padded_auction = self.auction + (['PAD_END'] * 4 * n_steps)

        times_seen = [0, 0, 0, 0]

        i = 0
        while sum(times_seen) < 4 * n_steps:
            if padded_auction[i] == 'PAD_START':
                i += 1
                continue

            hand_ix = i % 4

            t = times_seen[hand_ix]
        
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

            X[hand_ix, t, :] = ftrs
            y[hand_ix, t, :] = bidding.encode_bid(target_bid)

            times_seen[hand_ix] += 1
            i += 1

        for n in times_seen:
            assert n == n_steps

        return X, y

def get_card_index(card, n_cards):
    assert(n_cards % 4 == 0)
    x_card_index = n_cards // 4 - 1
    if card not in CARD_INDEX_LOOKUP:
        return x_card_index
    return min(CARD_INDEX_LOOKUP[card], x_card_index)

def parse_hand_f(n_cards):
    def f(hand):
        x = np.zeros((1, n_cards), dtype=np.int32)
        suits = hand.split('.')
        assert(len(suits) == 4)
        for suit_index in [0, 1, 2, 3]:
            for card in suits[suit_index]:
                card_index = get_card_index(card, n_cards)
                x[0, suit_index * n_cards // 4 + card_index] += 1
        return x
    return f

def get_shape(hand):
    return np.sum(hand.reshape((hand.shape[0], 4, -1)), axis=2)

def get_hcp(hand):
    x = hand.reshape((hand.shape[0], 4, -1))
    A = np.zeros_like(x)
    A[:,:,0] = 1
    K = np.zeros_like(x)
    K[:,:,1] = 1
    Q = np.zeros_like(x)
    Q[:,:,2] = 1
    J = np.zeros_like(x)
    J[:,:,3] = 1

    points = 4 * A * x + 3 * K * x + 2 * Q * x + J * x

    return np.sum(points, axis=(1,2))

def get_bid_ids(auction, player_i, n_steps):
    i = player_i
    result = []

    while len(result) < n_steps:
        if i >= len(auction):
            result.append(bidding.BID2ID['PAD_END'])
            continue
        call = auction[i]
        if not (call == 'PAD_START' and len(result) == 0):
            result.append(bidding.BID2ID[call])
        i = i + 4

    return np.array(result)
