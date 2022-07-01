import sys
sys.path.append('../../../src')

import os.path
import numpy as np

from bidding.binary import DealData

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
        if i % 10000 == 0:
            print(i)

            X_part, y_part = deal_data.get_binary(n_steps=8)

            X[k:k+4] = X_part
            y[k:k+4] = y_part

            k += 4

    np.save(os.path.join(out_dir, 'X.npy'), X)
    np.save(os.path.join(out_dir, 'y.npy'), y)


if __name__ == '__main__':
    n = int(sys.argv[1]) # the number of hands to train on
    infnm = sys.argv[2] # file where the data is
    outdir = sys.argv[3]

    create_binary(load_deals(open(infnm)), n, outdir)

