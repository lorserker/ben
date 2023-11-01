import numpy as np


card_index_lookup = dict(
    zip(
        ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'],
        range(13)
    )
)

def set_data(X, i, deal_str):
    hands = deal_str.split('\t')
    assert(len(hands) == 4)

    for hand_index in [0, 1, 2, 3]:
        assert(len(hands[hand_index]) == 20)
        suits = hands[hand_index].split()
        assert(len(suits) == 4)
        for suit_index in [0, 1, 2, 3]:
            for card in suits[suit_index][1:]:
                card_index = card_index_lookup[card]
                X[i, suit_index, card_index, hand_index] = 1


def load_deals(fin):
    X = np.zeros((1, 4, 13, 4))
    contracts = {}
    deal_str = ''
    for line_number, line in enumerate(fin):
        line = line.decode('ascii').strip()
        if line_number % 421 == 0:
            if line_number > 0:
                yield (X, contracts, deal_str)
                contracts = {}
                X = np.zeros((1, 4, 13, 4))
                deal_str = ''
            deal_str = line
            set_data(X, 0, deal_str.replace('  ', '\t'))
        else:
            cols = line.split('\t')
            contracts[cols[0]] = (int(cols[1]), int(cols[2]))
