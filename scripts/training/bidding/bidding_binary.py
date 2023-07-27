import sys
sys.path.append('../../../src')

import os.path
import numpy as np

from bidding.binary import DealData

LEVELS = [1, 2, 3, 4, 5, 6, 7]

SUITS = ['C', 'D', 'H', 'S', 'N']
SUIT_RANK = {suit: i for i, suit in enumerate(SUITS)}

BID2ID = {
    'PAD_START': 0,
    'PAD_END': 1,
    'PASS': 2,
    'X': 3,
    'XX': 4,
}

SUITBID2ID = {bid: (i+5) for (i, bid) in enumerate(
    ['{}{}'.format(level, suit) for level in LEVELS for suit in SUITS])}

BID2ID.update(SUITBID2ID)

ID2BID = {bid: i for i, bid in BID2ID.items()}

def load_deals(fin):
    deal_str = ''
    deal_data = ''

    for line_number, line in enumerate(fin):
        line = line.strip()
        if line_number % 2 == 0:
            deal_str = line
        else:
            yield DealData.from_deal_auction_string(deal_str, line, 32)

def create_binary(data_it, n, out_dir):
    X = np.zeros((4 * n, 8, 159), dtype=np.float16)
    y = np.zeros((4 * n, 8, 40), dtype=np.uint8)

    k = 0

    for i, deal_data in enumerate(data_it):
        #if i % 10000 == 0:
        #    print(i)

        #print(deal_data)
        X_part, y_part = deal_data.get_binary(n_steps=8)
        X[k:k+4] = X_part
        y[k:k+4] = y_part
        for j in range(y_part.shape[1]):  # Iterate over the second dimension
            for i in range(y_part.shape[0]):  # Iterate over the first dimension
            #print(f"Element ({i}, {j}):")
                one_hot_array = y_part[i, j, :]
                index = np.argmax(one_hot_array)
                print(f" {ID2BID[index]}")

        k += 4

    np.save(os.path.join(out_dir, 'X.npy'), X)
    np.save(os.path.join(out_dir, 'y.npy'), y)


if __name__ == '__main__':
    n = int(sys.argv[1]) # the number of hands to train on
    infnm = sys.argv[2] # file where the data is
    outdir = sys.argv[3]

    create_binary(load_deals(open(infnm)), n, outdir)

