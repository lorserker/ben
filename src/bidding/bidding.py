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

def get_auction_as_string(auction):
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

def get_auction_as_list(auction):
    bid_strings = []
    for bid in auction:
        # Ignore PAD_START and PAD_END
        if (bid > 4):
            bid_strings.append(ID2BID[bid])
        else:
            if (bid == 2):
                bid_strings.append("PASS")
            if (bid == 3):
                bid_strings.append("X")
            if (bid == 4):
                bid_strings.append("XX")
    return bid_strings

def encode_bid(bid, alert_supported=False):
    bid = bid.replace("*",'')
    bid_one_hot = np.zeros((1, len(BID2ID)), dtype=np.float16)
    bid_one_hot[0, BID2ID[bid]] = 1
    return bid_one_hot

def get_input(lho_bid, partner_bid, rho_bid, hand, v_we, v_them):
    vuln = np.array([[v_we, v_them]], dtype=np.float16)
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

def get_level(bid):
    if bid =="P" or bid =="X" or bid == "XX":
        return 0
    return int(bid[0])

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

game_contracts = {'3N', '4N', '5N', '4H', '5H', '4S', '5S', '5C', '5D'}
slam_contracts = {'6C', '6D', '6H', '6S', '6N'}
grand_contracts = {'7C', '7D', '7H', '7S', '7N'}
game_contract_with_x = {'2S', '2H', '2N', '3C', '3D', '3H', '3S', '4C', '4D'}
game_contract_with_xx = {'1H', '1S', '1N', '2C', '3C'}
# We assume we have a valid auction
def is_game_bid(auction):
    # Strip down to just non-PASS bids
    #print("Have we bid game?",auction)
    non_pass_bids = [bid for bid in auction if bid != 'PASS']
    last_bid = non_pass_bids[-1]
    if last_bid == 'XX':
        if non_pass_bids[-3] in game_contract_with_xx:
            return True
        last_bid = non_pass_bids[-2]

    if last_bid == 'X':
        if non_pass_bids[-2] in game_contract_with_x:
            return True
        last_bid = non_pass_bids[-2]
    if last_bid in game_contracts or last_bid in slam_contracts or last_bid in grand_contracts:
        return True
    return False

def undisturbed(auction):
    their_bids = auction[::-2]
    return len([bid for bid in their_bids if BID2ID[bid] > 3]) == 0

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
    if contract is None or contract == "PASS":
        return None
    return 'NSHDC'.index(contract[1])

def get_decl_i(contract):
    if contract is None or contract == "PASS":
        return None
    return 'NESW'.index(contract[-1])

def get_bid_ids(auction, player_i, n_steps):
    i = player_i
    result = []
    bid_no = 0
    while len(result) < n_steps:
        if i >= len(auction):
            result.append(BID2ID['PAD_END'])
            continue
        call = auction[i]
        if not (call == 'PAD_START' and len(result) == 0):
            result.append(BID2ID[call])
            bid_no = bid_no + 1
        i = i + 4

    return np.array(result), bid_no

def get_partner_suit(seat, auction):
    partner_seat = (seat + 2) % 4
    for i in range(len(auction)):
        if i % 4 == partner_seat:
            call = auction[i]
            # We should check if it was a splinter or support of our suit
            if is_contract(call):
                return "SHDCN".index(call[1])
            else:
                # Need to find if it was a lead directing double
                if call == 'X':
                    call = auction[i]
                    if is_contract(call):
                        if call[0] > 1:
                            return "SHDCN".index(call[1])
