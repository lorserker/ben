import numpy as np

def deal_to_str(deal):
    x = deal.reshape((4, 13))
    symbols = 'AKQJT98765432'
    suits = []
    for i in range(4):
        s = ''
        for j in range(13):
            if x[i,j] > 0:
                s += symbols[j]
        suits.append(s)
    return '.'.join(suits)

def hand_to_str(hand):
    symbols = 'AKQJT98765432'
    suits = ["","","",""]
    hand.sort()
    for card in hand:
        suit_i = card // 13
        card_i = card % 13
        suits[suit_i] += symbols[card_i]
    return '.'.join(suits)


def encode_card(card_str):
    suit_i = 'SHDC'.index(card_str[0])
    card_i = 'AKQJT98765432'.index(card_str[1])
    return suit_i*13 + card_i

def decode_card(card52):
    suit_i = card52 // 13
    card_i = card52 % 13
    return 'SHDC'[suit_i] + 'AKQJT98765432'[card_i]


def random_deal():
    all_cards = list(range(52))
    np.random.shuffle(all_cards)

    hands_cards = [all_cards[:13], all_cards[13:26], all_cards[26:39], all_cards[39:]]
    hands = []

    for cards in hands_cards:
        hand = np.zeros(52, dtype=int)
        for c in cards:
            hand[c] += 1
        hands.append(hand)

    return ' '.join(map(deal_to_str, hands))


def random_dealer_vuln():
    dealer = np.random.choice(['N', 'E', 'S', 'W'])
    vuln = np.random.choice(['None', 'N-S', 'E-W', 'Both'])

    return '%s %s' % (dealer, vuln)

def board_dealer_vuln(number):
    dealerList = ['N', 'E', 'S', 'W']
    vulnList = ['None', 'N-S', 'E-W', 'Both', 
        'N-S', 'E-W', 'Both', 'None',
        'E-W', 'Both', 'None', 'N-S', 
        'Both', 'None', 'N-S', 'E-W']

    if number:
        dealer = dealerList[(number-1) % 4]
        vuln = vulnList[(number-1) % 16]
    else:         
        dealer = np.random.choice(['N', 'E', 'S', 'W'])
        vuln = np.random.choice(['None', 'N-S', 'E-W', 'Both'])

    return '%s %s' % (dealer, vuln)


def card32to52(c32):
    suit = int(c32 // 8)
    rank = int(c32 % 8)
    return suit * 13 + rank

def card52to32(c52):
    suit = int(c52 // 13)
    rank = int(c52 % 13)
    return suit * 8 + min(7, rank)


def hand32to52str(hand32):
    x = hand32.reshape((4, 8))
    symbols = 'AKQJT98x'
    suits = []
    for i in range(4):
        s = ''
        for j in range(8):
            if x[i, j] > 0:
                symbol = symbols[j]
                if symbol == 'x':
                    s += 'x' * int(x[i, j])
                else:
                    s += symbol
        suits.append(s)
    card_string = '.'.join(suits)
    return card_string

def handxxto52str(handxx, n_cards=32):
    x = handxx.reshape((4, n_cards // 4))
    full_symbols = 'AKQJT98765432'
    suits = []
    for i in range(4):
        s = ''
        for j in range((n_cards // 4) - 1):
            if x[i, j] > 0:
                s += full_symbols[j]
        s += 'x' * int(x[i, (n_cards // 4) - 1])
        suits.append(s)
    card_string = '.'.join(suits)
    return card_string

def reorder_hand(hands_string):
    # Define the order of the cards
    card_order = "AKQJT98765432"

    # Split the hands into a list
    hands_list = hands_string.split()

    # Function to sort a hand by suit
    def sort_hand_by_suit(hand):
        # Split the hand into suits
        suits = hand.split('.')
        # Sort each suit in the specified order
        sorted_suits = [''.join(sorted(suit, key=lambda card: card_order.index(card))) for suit in suits]
        # Join the sorted suits back together
        return '.'.join(sorted_suits)

    # Apply sorting to each hand
    sorted_hands = [sort_hand_by_suit(hand) for hand in hands_list]

    # Join the sorted hands back into a string
    return ' '.join(sorted_hands)

def convert_cards(card_string, opening_lead, hand_str, rng, n_cards=32):
    updated_card_string = card_string
    pips_in_suit = 13 - (n_cards // 4) + 1
    pips = [[True for _ in range(pips_in_suit)] for _ in range(4)]
    if opening_lead % 13 >= 13 - pips_in_suit :
        pips[opening_lead // 13][12 - (opening_lead % 13)] = False
    suit = 0
    for k in range(len(hand_str)):
        if hand_str[k] == '.':
            suit += 1
            continue
        if not hand_str[k].isdigit():
            continue
        card = int(hand_str[k])
        if card < pips_in_suit + 2:
            pips[suit][card-2] = False

    hands = updated_card_string.split(' ')

    for k in range(0, 4):
        suits = hands[k % 4].split(".")
        for j in range(4):
            numbers = list(range(pips_in_suit))
            rng.shuffle(numbers)
            for l in numbers:
                if pips[j][l] and "x" in suits[j]: 
                    suits[j] = suits[j].replace("x",str(l+2),1) 
                    pips[j][l] = False
        hands[k % 4] = ".".join(suits)

    updated_card_string = " ".join(hands)
    assert 'x' not in updated_card_string, f"All pips not replaced {updated_card_string}"
    return updated_card_string

def get_trick_winner_i(trick, strain_i):
    trick_cards_suit = [card // 13 for card in trick]

    is_trumped = any([suit_i == strain_i for suit_i in trick_cards_suit])

    highest_trump_i = 0
    highest_trump = 99
    for i in range(4):
        if trick_cards_suit[i] == strain_i and trick[i] < highest_trump:
            highest_trump_i = i
            highest_trump = trick[i]

    lead_suit = trick_cards_suit[0]

    highest_lead_suit_i = 0
    highest_lead_suit = 99
    for i in range(4):
        if trick_cards_suit[i] == lead_suit and trick[i] < highest_lead_suit:
            highest_lead_suit_i = i
            highest_lead_suit = trick[i]

    return highest_trump_i if is_trumped else highest_lead_suit_i
