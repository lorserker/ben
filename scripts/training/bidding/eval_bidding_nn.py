"""Measure how closely a bidding NN reproduces the auctions in a PBN file.

Usage: python eval_bidding_nn.py model.keras test.pbn [--ns 1] [--ew 1] [--n_cards 24] [--limit N] [--dump diffs.txt]

Reports
  1. Per-call agreement (teacher forced): given the real auction so far, does the NN
     pick the same call?
  2. Self-play agreement: the NN bids all four seats on its own; how often does it
     reproduce the whole auction / the final contract?
  3. Double dummy cost of the self-play differences, using the PBN's
     OptimumResultTable, in IMPs (NS view, NN contract vs. PBN contract).
"""
import os
import sys
sys.path.append('../../../src')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import re
import time
from collections import Counter, defaultdict

import numpy as np

from bidding import bidding
from bidding.binary import DealData
from bidding_binary_keras import load_pbn, get_binary_hcp_shape
import scoring

N_STEPS = 8
LEGAL_CANDIDATES = [bid for bid, i in bidding.BID2ID.items() if i >= 2]


def load_dd_tables(path):
    """Return one {(declarer, strain): tricks} per board, in file order."""
    tables, cur, inside = [], None, False
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('[Deal '):
                cur = {}
                tables.append(cur)
            if line.startswith('['):
                inside = line.startswith('[OptimumResultTable')
                continue
            if inside:
                m = re.match(r'([NESW])\s+(NT|[SHDC])\s+(\d+)', line)
                if m:
                    cur[(m.group(1), m.group(2)[0])] = int(m.group(3))
    return tables


def ns_score(contract, dd, vuln_ns, vuln_ew):
    if contract is None:
        return 0
    declarer = contract[-1]
    vul = vuln_ns if declarer in 'NS' else vuln_ew
    s = scoring.score(contract[:-1], vul, dd[(declarer, contract[1])])
    return s if declarer in 'NS' else -s


def seat_first_index(seat, dealer):
    return seat if seat >= dealer else seat + 4


def legal_mask(auction):
    mask = np.zeros(len(bidding.BID2ID), dtype=bool)
    for bid in LEGAL_CANDIDATES:
        if bidding.can_bid(bid, auction):
            mask[bidding.BID2ID[bid]] = True
    return mask


def pick(probs, auction):
    p = np.where(legal_mask(auction), probs, -1)
    return bidding.ID2BID[int(np.argmax(p))]


def predict(model, x, batch=4096):
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i:i + batch].astype(np.float16), training=False).numpy())
    return np.concatenate(out)


def bid_kind(bid):
    return 'pass' if bid == 'PASS' else 'X/XX' if bid in ('X', 'XX') else 'contract'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('pbn')
    ap.add_argument('--ns', type=int, default=1)
    ap.add_argument('--ew', type=int, default=1)
    ap.add_argument('--n_cards', type=int, default=24)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dump', default=None, help='write self-play differences here')
    args = ap.parse_args()

    from tensorflow.keras.models import load_model
    model = load_model(args.model, compile=False)

    with open(args.pbn, encoding='utf-8') as f:
        lines = load_pbn(f.readlines())
    dd_tables = load_dd_tables(args.pbn)
    deals = [DealData.from_deal_auction_string(lines[2 * i], lines[2 * i + 1], '', args.ns, args.ew, args.n_cards)
             for i in range(len(lines) // 2)]
    if args.limit:
        deals, dd_tables = deals[:args.limit], dd_tables[:args.limit]
    for d in deals:
        d.auction = [b.replace('*', '') for b in d.auction]
    n = len(deals)
    print(f'{n} boards from {args.pbn}')

    # 1. Teacher forced, per call
    t0 = time.time()
    X = np.zeros((4 * n, N_STEPS, 2 + 2 + 1 + 4 + args.n_cards + 4 * 40), dtype=np.float16)
    for b, d in enumerate(deals):
        X[4 * b:4 * b + 4] = get_binary_hcp_shape(d, args.ns, args.ew)[0]
    P = predict(model, X)

    hits, totals = Counter(), Counter()
    for b, d in enumerate(deals):
        auction = d.auction
        for seat in range(4):
            i = seat_first_index(seat, d.dealer)
            for t in range(N_STEPS):
                if i >= len(auction):
                    break
                target = auction[i]
                guess = pick(P[4 * b + seat, t], auction[:i])
                ok = guess == target
                for key in ('all', f'round {t + 1}', f'Brill {bid_kind(target)}',
                            'opening' if all(x in ('PAD_START', 'PASS') for x in auction[:i]) else 'later'):
                    totals[key] += 1
                    hits[key] += ok
                i += 4
    print(f'\n1. Per-call agreement, Brill auction given ({time.time() - t0:.0f}s)')
    for key in ['all', 'opening', 'later', 'Brill pass', 'Brill X/XX', 'Brill contract'] + [f'round {r}' for r in range(1, N_STEPS + 1)]:
        if totals[key]:
            print(f'   {key:16} {100 * hits[key] / totals[key]:6.2f}%  of {totals[key]}')

    # 2. Self play
    t0 = time.time()
    auctions = [['PAD_START'] * d.dealer for d in deals]
    open_boards = list(range(n))
    while open_boards:
        xs, ts = [], []
        for b in open_boards:
            d, auction = deals[b], auctions[b]
            seat = len(auction) % 4
            t = (len(auction) - seat_first_index(seat, d.dealer)) // 4
            partial = DealData(d.dealer, d.vuln_ns, d.vuln_ew, d.hands, auction, '', args.ns, args.ew, d.deal_str, '', args.n_cards)
            partial.shapes, partial.hcp = d.shapes, d.hcp
            xs.append(get_binary_hcp_shape(partial, args.ns, args.ew)[0][seat])
            ts.append(t)
        probs = predict(model, np.stack(xs))
        still_open = []
        for k, b in enumerate(open_boards):
            auction = auctions[b]
            bid = pick(probs[k, ts[k]], auction) if ts[k] < N_STEPS else 'PASS'
            auction.append(bid)
            if not bidding.auction_over(auction):
                still_open.append(b)
        open_boards = still_open

    same_auction = same_contract = 0
    imps, diffs = [], []
    by_brill_contract = defaultdict(lambda: [0, 0])
    for b, d in enumerate(deals):
        brill_c, nn_c = bidding.get_contract(d.auction), bidding.get_contract(auctions[b])
        a_ok = auctions[b] == d.auction
        c_ok = brill_c == nn_c
        same_auction += a_ok
        same_contract += c_ok
        level = 'passout' if brill_c is None else 'slam' if brill_c[0] in '67' else 'game' if bidding.is_game_bid(d.auction) else 'partscore'
        by_brill_contract[level][0] += 1
        by_brill_contract[level][1] += c_ok
        if not c_ok:
            dd = dd_tables[b]
            diff = ns_score(nn_c, dd, d.vuln_ns, d.vuln_ew) - ns_score(brill_c, dd, d.vuln_ns, d.vuln_ew)
            imps.append(scoring.diff_to_imps(abs(diff)) * (1 if diff >= 0 else -1))
        if not a_ok:
            diffs.append((b, d, brill_c, nn_c, imps[-1] if not c_ok else 0, auctions[b]))

    print(f'\n2. Self play, NN bids all four seats ({time.time() - t0:.0f}s)')
    print(f'   identical auction  {100 * same_auction / n:6.2f}%')
    print(f'   identical contract {100 * same_contract / n:6.2f}%  (level, strain, doubling, declarer)')
    for level in ('passout', 'partscore', 'game', 'slam'):
        tot, ok = by_brill_contract[level]
        if tot:
            print(f'      Brill {level:10} {100 * ok / tot:6.2f}%  of {tot}')

    imps = np.array(imps)
    print(f'\n3. Double dummy cost of different contracts (NS view, NN minus Brill)')
    print(f'   boards with different contract {len(imps)}')
    if len(imps):
        print(f'   mean |IMPs| per differing board {np.abs(imps).mean():.2f}')
        print(f'   same DD score (0 IMPs)          {100 * np.mean(imps == 0):.1f}% of differing boards')
        print(f'   total |IMPs| per board overall  {np.abs(imps).sum() / n:.3f}')

    if args.dump:
        with open(args.dump, 'w', encoding='utf-8') as f:
            for b, d, brill_c, nn_c, imp, nn_auc in diffs:
                fmt = lambda a: ' '.join(x.replace('PASS', 'P') for x in a if x != 'PAD_START')
                f.write(f"Board {b + 1}  Dealer {'NESW'[d.dealer]}  Vul NS={d.vuln_ns} EW={d.vuln_ew}\n")
                f.write(f'  {d.deal_str.strip()}\n')
                f.write(f'  Brill: {fmt(d.auction):60} {brill_c}\n')
                f.write(f'  NN:    {fmt(nn_auc):60} {nn_c}  ({imp:+d} IMP DD, NS view)\n\n')
        print(f'\nWrote {len(diffs)} differing auctions to {args.dump}')


if __name__ == '__main__':
    main()
