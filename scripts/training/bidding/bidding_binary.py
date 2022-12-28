import sys
sys.path.append('../../../src')

import os.path
import numpy as np

from bidding.binary import DealData

## HCP target format
# 0 = hcp lho
# 1 = hcp part
# 2 = hcp rho

## SHAPE target format
# 0:4 shape S,H,D,C lho
# 4:8 partner
# 8:12 rho


def load_deals(fin):
    deal_str = ''
    deal_data = ''

    for line_number, line in enumerate(fin):
        line = line.strip()
        if line_number % 2 == 0:
            deal_str = line
        else:
            yield DealData.from_deal_auction_string(deal_str, line, 32)

def create_binary(data_it, n, out_dir, n_steps=8):
    X = np.zeros((4 * n, n_steps, 2 + 1 + 4 + 32 + 3 * 40), dtype=np.float16)
    y = np.zeros((4 * n, n_steps, 40), dtype=np.float16)

    k = 0

    for i, deal_data in enumerate(data_it):
        if i >= n:
            break
        if i % 100000 == 0:
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

