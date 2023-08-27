import numpy as np

from bidding import bidding

## input
#   0:32  player hand
#  32:64  seen hand (dummy, or declarer if we create data for dummy)
#  64:96  last trick card 0
#  96:128 last trick card 1
# 128:160 last trick card 2
# 160:192 last trick card 3
# 192:224 this trick lho card
# 224:256 this trick pard card
# 256:288 this trick rho card
# 288:292 last trick lead player index one-hot
#     292 level
# 293:298 strain one hot N, S, H, D, C
class BinaryInput:

    def __init__(self, x):
        self.x = x
        self.n_samples, self.n_ftrs = x.shape

    def set_player_hand(self, hand_bin):
        self.x[:, :32] = hand_bin

    def get_player_hand(self):
        return self.x[:, :32]

    def set_public_hand(self, hand_bin):
        self.x[:, 32:64] = hand_bin 

    def get_public_hand(self):
        # public hand is usually dummy (unless the player is dummy, then public is declarer)
        return self.x[:, 32:64]

    def set_last_trick(self, last_trick):
        self.x[:, 64:192] = last_trick.reshape((self.n_samples, 4*32))

    def get_last_trick(self):
        return self.x[:, 64:192].reshape((self.n_samples, 4, 32))

    def set_this_trick(self, this_trick):
        self.x[:, 192:288] = this_trick.reshape((self.n_samples, 3*32))

    def get_this_trick(self):
        return self.x[:, 192:288].reshape((self.n_samples, 3, 32))

    def set_last_trick_lead(self, last_trick_lead_i):
        self.x[:, 288:292] = 0
        #self.x[]
        self.x[last_trick_lead_i == 0, 288] = 1
        self.x[last_trick_lead_i == 1, 289] = 1
        self.x[last_trick_lead_i == 2, 290] = 1
        self.x[last_trick_lead_i == 3, 291] = 1

    def get_last_trick_lead(self):
        return np.argmax(self.x[:, 288:292], axis=1)

    def set_level(self, level):
        self.x[:, 292] = level

    def get_level(self):
        return self.x[:, 292]

    def set_strain(self, strain):
        self.x[:, 293:298] = strain

    def get_strain(self):
        return self.x[:, 293:298]

    def get_this_trick_lead_suit(self):
        '''
        returns one-hot encoded suit of the card lead in this trick [S, H, D, C]
        '''
        this_trick = self.get_this_trick()
        this_trick_lead_suit = np.zeros((self.n_samples, 4))

        for k in (0, 1, 2):
            card = np.argmax(this_trick[:,k,:], axis=1)
            was_played = np.max(this_trick[:,k,:], axis=1) > 0
            lead_found = np.max(this_trick_lead_suit, axis=1) > 0

            card_suit = card // 8

            this_trick_lead_suit[was_played & (~lead_found), card_suit[was_played & (~lead_found)]] = 1

        return this_trick_lead_suit


def get_cards_from_binary_hand(hand):
    cards = []
    for i, count in enumerate(hand):
        for _ in range(int(count)):
            cards.append(i)
    return np.array(cards)

def get_binary_hand_from_cards(cards):
    hand = np.zeros(32)
    for card in cards:
        hand[int(card)] += 1
    return hand


CARD_INDEX_LOOKUP = dict(
    zip(
        ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'],
        range(13)
    )
)


def get_card_index(card, n_cards):
    assert(n_cards % 4 == 0)
    x_card_index = n_cards // 4 - 1
    if card not in CARD_INDEX_LOOKUP:
        return x_card_index
    return min(CARD_INDEX_LOOKUP[card], x_card_index)


def parse_hand_f(n_cards):
    def f(hand):
        x = np.zeros((1, n_cards))
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


def get_auction_binary(n_steps, auction_input, hand_ix, hand, vuln, ns, ew):
    assert(len(hand.shape) == 2)
    assert(hand.shape[1] == 32)

    n_samples = hand.shape[0]

    # Do not add 2 cells for biddingsystem
    X = np.zeros((n_samples, n_steps, 2 + 1 + 4 + 32 + 3*40), dtype=np.float16)

    vuln_us_them = np.array(
        [vuln[hand_ix % 2], vuln[(hand_ix + 1) % 2]], 
        dtype=np.float32
    )
    shp = (get_shape(hand) - 3.25) / 1.75
    hcp = (get_hcp(hand) - 10) / 4
    
    auction = auction_input
    if isinstance(auction, list):
        auction_input = auction_input + ['PAD_END'] * 4 * n_steps
        auction = bidding.BID2ID['PAD_END'] * np.ones((n_samples, len(auction_input)), dtype=np.int32)
        for i, bid in enumerate(auction_input):
            auction[:,i] = bidding.BID2ID[bid]
    
    bid_i = hand_ix
    while np.all(auction[:, bid_i] == bidding.BID2ID['PAD_START']):
        bid_i += 4

    X[:,:,:2] = vuln_us_them
    X[:,:,2:3] = hcp.reshape((n_samples, 1, 1))
    X[:,:,3:7] = shp.reshape((n_samples, 1, 4))
    X[:,:,7:39] = hand.reshape((n_samples, 1, 32))

    step_i = 0
    s_all = np.arange(n_samples, dtype=np.int32)
    while step_i < n_steps:
        lho_bid = auction[:, bid_i - 3] if bid_i - 3 >= 0 else bidding.BID2ID['PAD_START']
        partner_bid = auction[:, bid_i - 2] if bid_i - 2 >= 0 else bidding.BID2ID['PAD_START']
        rho_bid = auction[:, bid_i - 1] if bid_i - 1 >= 0 else bidding.BID2ID['PAD_START']
        
        X[s_all,step_i,39+lho_bid] = 1
        X[s_all,step_i,(39+40)+partner_bid] = 1
        X[s_all,step_i,(39+2*40)+rho_bid] = 1

        step_i += 1
        bid_i += 4

    #Insert bidding system, -1 means no system
    if (ns == -1):
        return X
    padding_width = ((0, 0),(0, 0), (2, 0))
    X_padded = np.pad(X, padding_width, mode='constant', constant_values=(ns, ew))
    return X_padded


def get_lead_binary(auction, hand, binfo, vuln, ns, ew):
    contract = bidding.get_contract(auction)

    level = int(contract[0])
    strain = bidding.get_strain_i(contract)
    doubled = int('X' in contract)
    redbld = int('XX' in contract)

    decl_index = bidding.get_decl_i(contract)
    lead_index = (decl_index + 1) % 4

    vuln_us = vuln[lead_index % 2]
    vuln_them = vuln[decl_index % 2]

    x = np.zeros(42)

    x[0] = level
    x[1 + strain] = 1
    x[6] = doubled
    x[7] = redbld
    x[8] = vuln_us
    x[9] = vuln_them
    x[10:] = hand.reshape(32)

    b = np.zeros(15)
    n_steps = 1 + len(auction) // 4
    A = get_auction_binary(n_steps, auction, lead_index, hand, vuln, ns, ew)

    p_hcp, p_shp = binfo.model(A)

    b[:3] = p_hcp.reshape((-1, n_steps, 3))[:,-1,:].reshape(3)
    b[3:] = p_shp.reshape((-1, n_steps, 12))[:,-1,:].reshape(12)

    return x.reshape((1, -1)), b.reshape((1, -1))
