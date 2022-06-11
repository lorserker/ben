import sys
sys.path.append('../../../src')

import numpy as np

from bidding.binary import DealData


def get_binary(n, infnm):
    X = np.zeros((4 * n, 8, 159), dtype=np.float16)
    y = np.zeros((4 * n, 8, 40), dtype=np.uint8)

    k = 0

    deal_str = None
    for i, line in enumerate(open(infnm)):
        if i % 10000 == 0:
            print(i)
        if i % 2 == 0:
            deal_str = line
        else:
            deal_data = DealData.from_deal_auction_string(deal_str, line, 32)

            X_part, y_part = deal_data.get_binary(n_steps=8)

            X[k:k+4] = X_part
            y[k:k+4] = y_part

            k += 4

    return X, y


if __name__ == '__main__':
    n = int(sys.argv[1]) # the number of hands to train on
    infnm = sys.argv[2] # file where the data is

    X, y = get_binary(n, infnm)

    np.save('X.npy', X)
    np.save('y.npy', y)
