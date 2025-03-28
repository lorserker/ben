import numpy as np

from bidding import bidding

CARD_INDEX_LOOKUP = dict(
    zip(
        ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'],
        range(13)
    )
)

class DealData(object):

    def __init__(self, dealer, vuln_ns, vuln_ew, hands, auction, par, ns, ew, deal_str, auction_str, n_cards=52):
        self.n_cards = n_cards
        self.dealer = dealer
        self.vuln_ns = vuln_ns
        self.vuln_ew = vuln_ew
        self.hands = hands
        self.shapes = list(map(lambda shape: (shape - 3.25)/1.75, map(get_shape, hands)))
        self.hcp = list(map(lambda point_count: (np.array([[point_count[0]]]) - 10) / 4, map(get_hcp, hands)))
        self.auction = auction
        self.ns = ns
        self.ew = ew
        self.deal_str = deal_str
        self.auction_str = auction_str
        self.par = par


    def __str__(self):
        return f"DealData: n_cards={self.n_cards}, NS={self.ns}, EW={self.ew}, dealer={self.dealer}, vuln_ns={self.vuln_ns}, vuln_ew={self.vuln_ew},\n hands={self.hands},\n shapes={self.shapes},\n hcp={self.hcp},\n auction={self.auction}"


    @classmethod
    def from_deal_auction_string(cls, deal_str, auction_str, par, ns, ew, n_cards=32, rotate=False):
        dealer = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        vuln = {'N-S': (True, False), 'E-W': (False, True), 'None': (False, False), 'Both': (True, True)}
        hands = list(map(parse_hand_f(n_cards), deal_str.strip().split()))
        auction_parts = auction_str.strip().replace('P', 'PASS').split()
        dealer_ix = dealer[auction_parts[0]]
        if rotate:
            if dealer_ix == 1 or dealer_ix == 3:
                vuln_ew, vuln_ns = vuln[auction_parts[1]]
            else:
                vuln_ns, vuln_ew = vuln[auction_parts[1]]
            hands =  hands[dealer_ix:] + hands[:dealer_ix]
            deal = deal_str.strip().split()
            deal = deal[dealer_ix:] + deal[:dealer_ix] 
            deal_str = ' '.join(deal)
            dealer_ix = 0
            auction = auction_parts[2:]
        else:
            vuln_ns, vuln_ew = vuln[auction_parts[1]]
            auction = (['PAD_START'] * dealer_ix) + auction_parts[2:]
        return cls(dealer_ix, vuln_ns, vuln_ew, hands, auction, par, ns, ew, deal_str, auction_str, n_cards)

    def get_binary(self, ns, ew, bids, n_steps=8):
        if ns == -1:
            X = np.zeros((4, n_steps, 2 + 1 + 4 + self.n_cards + bids * 40), dtype=np.float16)
        else: 
            X = np.zeros((4, n_steps, 2 + 2 + 1 + 4 + self.n_cards + bids * 40), dtype=np.float16)
        y = np.zeros((4, n_steps, 40), dtype=np.uint8)

        padded_auction = self.auction + (['PAD_END'] * 4 * n_steps)

        #print(self.auction)
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
            vuln = np.array([[v_we, v_them]], dtype=np.float16)
            hcp = self.hcp[hand_ix]
            shape = self.shapes[hand_ix]
            
            if bids == 4:
                my_bid = padded_auction[i - 4] if i - 4 >= 0 else 'PAD_START'
            lho_bid = padded_auction[i - 3] if i - 3 >= 0 else 'PAD_START'
            partner_bid = padded_auction[i - 2] if i - 2 >= 0 else 'PAD_START'
            rho_bid = padded_auction[i - 1] if i - 1 >= 0 else 'PAD_START'
            target_bid = padded_auction[i]

            # Create an array with [ns, ew] only if neither ns nor ew is -1
            if hand_ix % 2 == 0:
                ns_ew_array = np.array([ns, ew], ndmin=2) if ns != -1 and ew != -1 else np.array([],ndmin=2)
            else:
                ns_ew_array = np.array([ew, ns], ndmin=2) if ns != -1 and ew != -1 else np.array([],ndmin=2)

            ftrs = np.concatenate((
                ns_ew_array,
                vuln,
                hcp,
                shape,
                self.hands[hand_ix],
                bidding.encode_bid(my_bid) if bids == 4 else np.array([], ndmin=2),
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
    
    def get_binary_hcp_shape(self, ns, ew, bids, n_steps=8, alert_supported=False):
        if ns == -1:
            X = np.zeros((4, n_steps, 2 + 1 + 4 + self.n_cards + bids * 40), dtype=np.float16)
        else: 
            X = np.zeros((4, n_steps, 2 + 2 + 1 + 4 + self.n_cards + bids * 40), dtype=np.float16)

        y = np.zeros((4, n_steps, 40), dtype=np.uint8)
        z = np.zeros((4, n_steps, 1), dtype=np.uint8)
        HCP = np.zeros((4, n_steps, 3), dtype=np.float16)
        SHAPE = np.zeros((4, n_steps, 12), dtype=np.float16)
        
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
            vuln = np.array([[v_we, v_them]], dtype=np.float16)
            hcp = self.hcp[hand_ix]
            shape = self.shapes[hand_ix]
            
            if bids == 4:
                my_bid = padded_auction[i - 4] if i - 4 >= 0 else 'PAD_START'
            lho_bid = padded_auction[i - 3] if i - 3 >= 0 else 'PAD_START'
            partner_bid = padded_auction[i - 2] if i - 2 >= 0 else 'PAD_START'
            rho_bid = padded_auction[i - 1] if i - 1 >= 0 else 'PAD_START'
            target_bid = padded_auction[i]

            # Create an array with [ns, ew] only if neither ns nor ew is -1
            if hand_ix % 2 == 0:
                ns_ew_array = np.array([ns, ew], ndmin=2) if ns != -1 and ew != -1 else np.array([],ndmin=2)
            else:
                ns_ew_array = np.array([ew, ns], ndmin=2) if ns != -1 and ew != -1 else np.array([],ndmin=2)

            ftrs = np.concatenate((
                ns_ew_array,
                vuln,
                hcp,
                shape,
                self.hands[hand_ix],
                bidding.encode_bid(my_bid) if bids == 4 else np.array([], ndmin=2),
                bidding.encode_bid(lho_bid),
                bidding.encode_bid(partner_bid),
                bidding.encode_bid(rho_bid)
            ), axis=1)

            X[hand_ix, t, :] = ftrs
            y[hand_ix, t, :] = bidding.encode_bid(target_bid, alert_supported)
            z[hand_ix, t, :] = 1 if "*" in target_bid else 0

            HCP[hand_ix, t, 0] = self.hcp[(hand_ix - 3) % 4][0,0]
            HCP[hand_ix, t, 1] = self.hcp[(hand_ix - 2) % 4][0,0]
            HCP[hand_ix, t, 2] = self.hcp[(hand_ix - 1) % 4][0,0]

            SHAPE[hand_ix, t, 0:4] = self.shapes[(hand_ix - 3) % 4][0]
            SHAPE[hand_ix, t, 4:8] = self.shapes[(hand_ix - 2) % 4][0]
            SHAPE[hand_ix, t, 8:12] = self.shapes[(hand_ix - 1) % 4][0]

            times_seen[hand_ix] += 1
            i += 1

        for n in times_seen:
            assert n == n_steps

        return X, y, HCP, SHAPE, z

    def get_binary_contract(self):
        X = np.zeros(2 + 2 * self.n_cards, dtype=np.float16)
        
        parscore = self.par.split(" ")

        contract = parscore[0]
        hand_ix = 2 if parscore[1][0] == "S" else 3 if parscore[1][0] == "W" else 1 if parscore[1][0] == "E" else 0
        
        v_we = self.vuln_ns if hand_ix % 2 == 0 else self.vuln_ew
        v_them = self.vuln_ew if hand_ix % 2 == 0 else self.vuln_ns
        vuln = np.array([[v_we, v_them]], dtype=np.float16)
        
        ftrs = np.concatenate((
            vuln,
            self.hands[hand_ix],
            self.hands[(hand_ix + 2) % 4],
        ), axis=1)
        X = ftrs
        u = len(contract) > 2 and contract[2] == "X"
        tricks_one_hot = np.zeros((1, 14), dtype=np.float16)
        tricks_one_hot[0, int(parscore[2])] = 1
        z = tricks_one_hot
        y = bidding.encode_bid(contract[0:2])
        return X, y, u, z

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
    x = np.sum(hand.reshape((hand.shape[0], 4, -1)), axis=2)
    return x

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
    _sum = np.sum(points, axis=(1, 2))

    return _sum

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
