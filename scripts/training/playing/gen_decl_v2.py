"""
Declarer training data, v2: the 298 inputs of binary_keras_suit_nt.binary_data_decl, plus 21 per
trick that the play models never had:

  298:313  bidding info from declarer's seat -- the bidding-info model's estimate of the hidden
           hands at declarer's last bidding step (3 HCP + 12 shape), exactly as BEN computes it
           at runtime (binary.get_auction_binary_sampling + models.binfo_model)
  313:315  doubled, redoubled
  315:317  vulnerable: declaring side, defenders
  317      tricks the declaring side has taken before this trick / 13
  318      tricks it still needs for the contract (level + 6 - taken, at least 0) / 13

Rows are written in source order with the same filter as binary_keras_suit_nt.py, so the last 5%
are the same held-out deals as the existing X_decl_<suit|nt>.npy.

Usage (from scripts/training/playing):
  python gen_decl_v2.py <config> <play.txt> <out_dir> <suit|nt>
"""
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import binary
import conf
from bidding import bidding
from nn.models_tf2 import Models
from binary_keras_suit_nt import binary_data_decl
from binary_util import DealMeta, get_cards, get_tricks, wins_trick_index

N_STEPS, N_BASE, N_EXTRA = 11, 298, 21
SEATS = 'NESW'
BID = {'PP': 'PASS', 'DD': 'X', 'RR': 'XX'}
BATCH = 4096


def records(path):
    with open(path) as f:
        while True:
            lines = [f.readline().strip() for _ in range(4)]
            if not lines[0]:
                return
            yield lines


def keep(meta_str, kind):
    strain = DealMeta.from_str(meta_str).strain
    return (strain == 'N') == (kind == 'nt')


def tricks_before(play_str, trump):
    """Declaring-side tricks taken before each of the 13 tricks. Seats as in get_play_labels:
    0 = opening leader, 1 = dummy, 2 = the other defender, 3 = declarer."""
    out, won, lead = [], 0, 0
    for trick in get_tricks(get_cards(play_str)):
        out.append(won)
        lead = wins_trick_index(trick, trump, lead)
        won += lead in (1, 3)
    return out


def main():
    config, source, out_dir, kind = sys.argv[1:5]
    ben_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
    models = Models.from_conf(conf.load(config), ben_root)

    n = sum(1 for _, meta, _, _ in records(source) if keep(meta, kind))
    os.makedirs(out_dir, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(out_dir, f'X_decl_{kind}.npy'), mode='w+',
                                  dtype=np.float16, shape=(n, N_STEPS, N_BASE + N_EXTRA))
    Y = np.lib.format.open_memmap(os.path.join(out_dir, f'Y_decl_{kind}.npy'), mode='w+',
                                  dtype=np.float16, shape=(n, N_STEPS, 32))
    print(f'{n} {kind} tables')

    pending = defaultdict(list)   # n_steps -> [(row, A)]

    def flush(n_steps):
        items = pending.pop(n_steps, [])
        if not items:
            return
        A = np.concatenate([a for _, a in items], axis=0)
        p_hcp, p_shp = models.binfo_model.pred_fun(A)
        p_hcp = p_hcp.reshape((-1, n_steps, 3))[:, -1, :]
        p_shp = p_shp.reshape((-1, n_steps, 12))[:, -1, :]
        for j, (row, _) in enumerate(items):
            X[row, :, N_BASE:N_BASE + 3] = p_hcp[j]
            X[row, :, N_BASE + 3:N_BASE + 15] = p_shp[j]

    t0, row = time.time(), 0
    for deal, meta, auction_str, play in records(source):
        if not keep(meta, kind):
            continue
        x, y = binary_data_decl(deal, meta, play)
        X[row, :, :N_BASE] = x[0]
        Y[row] = y[0]

        d = DealMeta.from_str(meta)
        dealer_c, vuln_s = meta.split()[:2]
        hands = dict(zip('WNES', deal[2:].split()))
        decl_i = SEATS.index(d.declarer)
        vuln = [vuln_s.upper() in ('NS', 'ALL', 'BOTH'), vuln_s.upper() in ('EW', 'ALL', 'BOTH')]
        extra = X[row, :, N_BASE:]
        extra[:, 15] = d.doubled and not d.redoubled
        extra[:, 16] = d.redoubled
        extra[:, 17] = vuln[decl_i % 2]
        extra[:, 18] = vuln[(decl_i + 1) % 2]
        taken = tricks_before(play, d.strain)[:N_STEPS]
        extra[:, 19] = np.array(taken) / 13
        extra[:, 20] = np.maximum(0, d.level + 6 - np.array(taken)) / 13

        auction = ['PAD_START'] * SEATS.index(dealer_c) + [BID.get(b, b) for b in auction_str.split()]
        hand_bid = binary.parse_hand_f(models.n_cards_bidding)(hands[d.declarer])
        n_steps = binary.calculate_step_bidding_info(auction)
        A = binary.get_auction_binary_sampling(n_steps, auction, decl_i, hand_bid, vuln, models, models.n_cards_bidding)
        pending[n_steps].append((row, A))
        if len(pending[n_steps]) >= BATCH:
            flush(n_steps)

        row += 1
        if row % 100000 == 0:
            print(f'{row}/{n} ({time.time() - t0:.0f}s)', flush=True)

    for n_steps in list(pending):
        flush(n_steps)
    X.flush(); Y.flush()
    print(f'done: {row} rows, {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
