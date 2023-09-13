import numpy as np


LEVELS = [1, 2, 3, 4, 5, 6, 7]

SUITS = ['C', 'D', 'H', 'S', 'N']
SUIT_RANK = {suit:i for i, suit in enumerate(SUITS)}

BID2ID = {
    'PAD_START': 0,
    'PAD_END': 1,
    'PASS': 2,
    'X': 3,
    'XX': 4,
}

SUITBID2ID = {bid:(i+5) for (i, bid) in enumerate(['{}{}'.format(level, suit) for level in LEVELS for suit in SUITS])}

BID2ID.update(SUITBID2ID)

ID2BID = {bid:i for i, bid in BID2ID.items()}

def get_action_as_string(auction):
    bid_strings = []
    for bid in auction:
        # Ignore PAD_START and PAD_END
        if (bid > 4):
            bid_strings.append(ID2BID[bid])
        else:
            if (bid == 2):
                bid_strings.append("P")
            if (bid == 3):
                bid_strings.append("X")
            if (bid == 4):
                bid_strings.append("XX")
    return "-".join(bid_strings)


def encode_bid(bid):
    bid_one_hot = np.zeros((1, len(BID2ID)), dtype=np.float32)
    bid_one_hot[0, BID2ID[bid]] = 1
    return bid_one_hot

def get_input(lho_bid, partner_bid, rho_bid, hand, v_we, v_them):
    vuln = np.array([[v_we, v_them]], dtype=np.float32)
    return np.concatenate((vuln, encode_bid(lho_bid), encode_bid(partner_bid), encode_bid(rho_bid), hand), axis=1)

def is_contract(bid):
    return bid[0].isdigit()

def can_double(auction):
    if len(auction) == 0:
        return False
    if is_contract(auction[-1]):
        return True
    if len(auction) >= 3 and is_contract(auction[-3]) and auction[-2] == 'PASS' and auction[-1] == 'PASS':
        return True
    return False

def can_redouble(auction):
    if len(auction) == 0:
        return False
    if auction[-1] == 'X':
        return True
    if len(auction) >= 3 and auction[-3] == 'X' and auction[-2] == 'PASS' and auction[-1] == 'PASS':
        return True
    return False

def last_contract(auction):
    for bid in reversed(auction):
        if is_contract(bid):
            return bid
    return None

def contract_level_step(contract):
    return int(contract[0])*5 + SUIT_RANK[contract[1]]

def is_higher_contract(this_contract, other_contract):
    return contract_level_step(this_contract) > contract_level_step(other_contract)

def can_bid_contract(bid, auction):
    assert is_contract(bid)
    contract = last_contract(auction)
    if contract is None:
        return True
    return is_higher_contract(bid, contract)

def auction_over(auction):
    if len(auction) < 4:
        return False
    if auction[-1] == 'PAD_END':
        return True
    contract = last_contract(auction)
    if contract is None:
        return all([bid == 'PASS' for bid in auction[-4:]]) and all([bid == 'PAD_START' for bid in auction[:-4]])
    else:
        return all([bid == 'PASS' for bid in auction[-3:]])

def can_bid(bid, auction):
    if bid == 'PASS':
        return True
    if bid == 'X':
        return can_double(auction)
    if bid == 'XX':
        return can_redouble(auction)
    if is_contract(bid):
        return can_bid_contract(bid, auction)
    return False

def sample_bid(auction, from_bids):
    from_bids = from_bids / (np.sum(from_bids) + 1e-6)
    if auction_over(auction):
        return 'PAD_END'
    while True:
        bid_one_hot = np.random.multinomial(1, from_bids[0])
        bid_id = np.argmax(bid_one_hot)
        bid = ID2BID[bid_id]
        if can_bid(bid, auction):
            return bid

def bid_max_bid(auction, from_bids):
    bid = ID2BID[np.argmax(from_bids)]
    if can_bid(bid, auction):
        return bid
    else:
        return sample_bid(auction, from_bids)
        
def get_contract(auction):
    contract = None
    doubled = False
    redoubled = False
    last_bid_i = None
    for i in reversed(range(len(auction))):
        bid = auction[i]
        if is_contract(bid):
            contract = bid
            last_bid_i = i
            break
        if bid == 'X':
            doubled = True
        if bid == 'XX':
            redoubled = True
    
    if contract is None:
        return None
    
    declarer_i = None
    for i in range(last_bid_i + 1):
        bid = auction[i]
        if not is_contract(bid):
            continue
        if (i + last_bid_i) % 2 != 0:
            continue
        if bid[1] != contract[1]:
            continue
        declarer_i = i
        break
        
    declarer = ['N', 'E', 'S', 'W'][declarer_i % 4]
    
    xx = '' if not doubled else 'X' if not redoubled else 'XX'
    
    return contract + xx + declarer

def get_strain_i(contract):
    return 'NSHDC'.index(contract[1])

def get_decl_i(contract):
    return 'NESW'.index(contract[-1])

def get_bid_ids(auction, player_i, n_steps):
    i = player_i
    result = []

    while len(result) < n_steps:
        if i >= len(auction):
            result.append(BID2ID['PAD_END'])
            continue
        call = auction[i]
        if not (call == 'PAD_START' and len(result) == 0):
            result.append(BID2ID[call])
        i = i + 4

    return np.array(result)
