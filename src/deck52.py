import random
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
    random.shuffle(all_cards)

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


def card32to52(c32):
    suit = c32 // 8
    rank = c32 % 8
    return suit * 13 + rank

def card52to32(c52):
    suit = c52 // 13
    rank = c52 % 13
    return suit * 8 + min(7, rank)


def hand32to52str(hand32, known_pips_suits, free_pips_suits):
    x = hand32.reshape((4, 8))
    for i in range(4):
        random.shuffle(free_pips_suits[i])
    suit_pip_i = [0, 0, 0, 0]
    free_i = [0, 0, 0, 0]
    symbols = 'AKQJT98x'
    full_symbols = 'AKQJT98765432'
    suits = []
    for i in range(4):
        s = ''
        for j in range(8):
            if x[i, j] > 0:
                symbol = symbols[j]
                if symbol == 'x':
                    if suit_pip_i[i] <= len(known_pips_suits[i]):
                        s += full_symbols[known_pips_suits[i][suit_pip_i[i]] % 13]
                        suit_pip_i[i] += 1
                    else:
                        s += full_symbols[free_pips[i][free_i[i]] % 13]
                        free_i[i] += 1
                else:
                    s += symbol
        suits.append(s)
    return '.'.join(suits)


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
