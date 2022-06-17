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


def load_deals_no_contracts(fin):
    deal_str = ''
    auction_str = ''

    for line_number, line in enumerate(fin):
        line = line.strip()
        if line_number % 2 == 0:
            if line_number > 0:
                yield (deal_str, auction_str, None)
                deal_str = ''
                auction_str = ''
            deal_str = line
        elif line_number % 2 == 1:
            auction_str = line

    yield (deal_str, auction_str, None)


def create_binary(data_it, n, out_dir, n_steps=8):
    X = np.zeros((4 * n, n_steps, 2 + 1 + 4 + 32 + 3 * 40), dtype=np.float16)
    y = np.zeros((4 * n, n_steps, 40), dtype=np.float16)
    HCP = np.zeros((4 * n, n_steps, 3), dtype=np.float16)
    SHAPE = np.zeros((4 * n, n_steps, 12), dtype=np.float16)

    for i, (deal_str, auction_str, _) in enumerate(data_it):
        if i % 10000 == 0:
            print(i)
        deal_data = DealData.from_deal_auction_string(deal_str, auction_str, 32)
        x_part, y_part, hcp_part, shape_part = deal_data.get_binary_hcp_shape(n_steps)
        start_ix = i * 4
        end_ix = (i + 1) * 4
        X[start_ix:end_ix,:,:] = x_part
        y[start_ix:end_ix,:,:] = y_part
        HCP[start_ix:end_ix,:,:] = hcp_part
        SHAPE[start_ix:end_ix,:,:] = shape_part

    np.save(os.path.join(out_dir, 'X.npy'), X)
    np.save(os.path.join(out_dir, 'y.npy'), y)
    np.save(os.path.join(out_dir, 'HCP.npy'), HCP)
    np.save(os.path.join(out_dir, 'SHAPE.npy'), SHAPE)


if __name__ == '__main__':
    n = int(sys.argv[1])
    infnm = sys.argv[2]
    outdir = sys.argv[3]

    create_binary(load_deals_no_contracts(open(infnm)), n, outdir, n_steps=8)
