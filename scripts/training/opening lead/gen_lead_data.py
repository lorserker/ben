"""
Opening-lead training data, encoded EXACTLY as BEN encodes a lead at runtime
(binary.get_auction_binary_for_lead): X = contract, vulnerability and the leader's 32-card hand
(42 values), B = the bidding-info model's estimate of the hidden hands at the leader's last
bidding step (15 values), y = the card actually led (32-card one-hot).

B depends on the bidding-info model in the configuration, so regenerate the data whenever that
model changes. Data written with an older bidding-info model (lead_keras_* from 2024) had B up to
0.5 off what BEN computes at runtime -- the lead model was trained on a different picture of the
auction from the one it is given.

Usage (from scripts/training/opening lead):
  python gen_lead_data.py <config> <out_dir> <source.txt> [<source.txt> ...]
Writes <out_dir>/suit and <out_dir>/nt, each with X.npy, B.npy, y.npy and src.npy (source file
index, record index), in source order -- so the LAST rows can be held out as a test set.
Source format: 4 lines per table (deal "W:...", "dealer vuln contract.result.declarer", auction,
play), as in data/WBC/play.txt and data/Jack/*.txt.
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

SEATS = 'NESW'
BID = {'PP': 'PASS', 'DD': 'X', 'RR': 'XX'}
RANKS = 'AKQJT98765432'
BATCH = 4096


def records(path):
    with open(path) as f:
        while True:
            lines = [f.readline().strip() for _ in range(4)]
            if not lines[0]:
                return
            yield lines


def lead32(card):
    return 'SHDC'.index(card[0]) * 8 + min(RANKS.index(card[1]), 7)


def main():
    config, out_dir, sources = sys.argv[1], sys.argv[2], sys.argv[3:]
    ben_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
    models = Models.from_conf(conf.load(config), ben_root)

    rows = {'suit': defaultdict(list), 'nt': defaultdict(list)}
    # Auction encodings waiting for the bidding-info model, grouped by number of steps
    pending = defaultdict(list)   # n_steps -> [(kind, row index, A)]
    t0, skipped = time.time(), 0

    def flush(n_steps):
        items = pending.pop(n_steps, [])
        if not items:
            return
        A = np.concatenate([a for _, _, a in items], axis=0)
        p_hcp, p_shp = models.binfo_model.pred_fun(A)
        p_hcp = p_hcp.reshape((-1, n_steps, 3))[:, -1, :]
        p_shp = p_shp.reshape((-1, n_steps, 12))[:, -1, :]
        for j, (kind, i, _) in enumerate(items):
            b = np.zeros(15, dtype=np.float16)
            b[:3] = p_hcp[j]
            b[3:] = p_shp[j]
            rows[kind]['B'][i] = b

    for s_i, path in enumerate(sources):
        for r_i, (deal, meta, auction_str, play) in enumerate(records(path)):
            try:
                dealer_c, vuln_s, outcome = meta.split()
                contract_s, _, decl_c = outcome.split('.')
                if not contract_s or contract_s[0] not in '1234567' or len(play) < 2:
                    skipped += 1
                    continue
                wnes = deal[2:].split()
                hands = dict(zip('WNES', wnes))
                nesw = [hands[s] for s in SEATS]
                dealer_i = SEATS.index(dealer_c)
                vuln = [vuln_s.upper() in ('NS', 'ALL', 'BOTH'), vuln_s.upper() in ('EW', 'ALL', 'BOTH')]
                auction = ['PAD_START'] * dealer_i + [BID.get(b, b) for b in auction_str.split()]
                contract = bidding.get_contract(auction)
                if contract is None:
                    skipped += 1
                    continue
                decl_i = bidding.get_decl_i(contract)
                lead_i = (decl_i + 1) % 4
                hand_bid = binary.parse_hand_f(models.n_cards_bidding)(nesw[lead_i])
                hand_play = binary.parse_hand_f(models.n_cards_play)(nesw[lead_i])

                level = int(contract[0])
                strain = bidding.get_strain_i(contract)
                x = np.zeros(10 + models.n_cards_play, dtype=np.float16)
                x[0] = level
                x[1 + strain] = 1
                x[6] = int('X' in contract)
                x[7] = int('XX' in contract)
                x[8] = vuln[lead_i % 2]
                x[9] = vuln[decl_i % 2]
                x[10:] = hand_play.reshape(models.n_cards_play)
                y = np.zeros(32, dtype=np.float16)
                y[lead32(play[:2].upper())] = 1

                n_steps = binary.calculate_step_bidding_info(auction)
                A = binary.get_auction_binary_sampling(n_steps, auction, lead_i, hand_bid, vuln, models, models.n_cards_bidding)
            except Exception:
                skipped += 1
                continue

            kind = 'nt' if strain == 0 else 'suit'
            i = len(rows[kind]['X'])
            rows[kind]['X'].append(x)
            rows[kind]['y'].append(y)
            rows[kind]['src'].append((s_i, r_i))
            rows[kind]['B'].append(None)
            pending[n_steps].append((kind, i, A))
            if len(pending[n_steps]) >= BATCH:
                flush(n_steps)

            n = len(rows['suit']['X']) + len(rows['nt']['X'])
            if n % 100000 == 0:
                print(f'{n} rows ({time.time() - t0:.0f}s)', flush=True)

    for n_steps in list(pending):
        flush(n_steps)

    for kind in ('suit', 'nt'):
        d = os.path.join(out_dir, kind)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, 'X.npy'), np.array(rows[kind]['X'], dtype=np.float16))
        np.save(os.path.join(d, 'B.npy'), np.array(rows[kind]['B'], dtype=np.float16))
        np.save(os.path.join(d, 'y.npy'), np.array(rows[kind]['y'], dtype=np.float16))
        np.save(os.path.join(d, 'src.npy'), np.array(rows[kind]['src'], dtype=np.int32))
        print(f'{kind}: {len(rows[kind]["X"])} rows -> {d}')
    print(f'skipped {skipped}, {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
