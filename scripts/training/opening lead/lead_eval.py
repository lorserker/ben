"""
Evaluate opening-lead models on the rows lead_train.py held out (the last test_fraction of each
data folder), on both jobs a lead model has in BEN:

  1. choosing candidates for our own lead -- coverage (a double-dummy safe lead is among BEN's
     candidates: probability >= lead_threshold, at least min_opening_leads) and DDOLAR / ADDOLAR
     of the model's top choice;
  2. scoring samples by how plausible the opponents' actual lead was -- top-1 accuracy and nll
     (-log p) of the card actually led.

The actual leads' own DDOLAR (the experts in WBC, the robots in Jack) is printed as reference.

Usage: python lead_eval.py <set> [<set> ...] -- <model.keras> [<model.keras> ...] [key=value]
  set:  <label>=<data_dir>:<source.txt>[,<source.txt>...]   (the sources gen_lead_data.py read,
        in the same order), e.g. wbc-suit=lead_v3/wbc/suit:../data/WBC/play.txt
  options: test_fraction=0.05 max_dd=8000 lead_threshold=0.2 min_leads=4
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from ddsolver.ddsolver import DDSolver

SEATS = 'NESW'
RANKS = 'AKQJT98765432'

argv = sys.argv[1:]
sep = argv.index('--')
sets, rest = argv[:sep], argv[sep + 1:]
model_paths = [a for a in rest if '=' not in a]
opts = dict(a.split('=', 1) for a in rest if '=' in a)
test_fraction = float(opts.get('test_fraction', 0.05))
max_dd = int(opts.get('max_dd', 8000))
threshold = float(opts.get('lead_threshold', 0.2))
min_leads = int(opts.get('min_leads', 4))


def bucket32(c52):
    s, r = divmod(c52, 13)
    return s * 8 + min(r, 7)


def read_sources(paths):
    out = []
    for p in paths:
        lines = open(p).read().split('\n')
        out.append([lines[i:i + 4] for i in range(0, len(lines) - 3, 4)])
    return out


def dd_for_rows(data_dir, sources, src_rows):
    """Double-dummy tricks for the leader's side, per lead card (32-bucket -> best value in bucket),
    the set of safe buckets, and whether every lead gives the same result. Cached per folder."""
    cache = os.path.join(data_dir, f'dd_test_{len(src_rows)}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return z['safe'], z['all_equal'], z['lead_safe']
    dds = DDSolver()
    recs = read_sources(sources)
    safe = np.zeros((len(src_rows), 32), dtype=bool)
    all_equal = np.zeros(len(src_rows), dtype=bool)
    lead_safe = np.zeros(len(src_rows), dtype=bool)
    pbns, leaders, strains, leads = [], [], [], []
    for s_i, r_i in src_rows:
        deal, meta, auction, play = recs[s_i][r_i]
        dealer, vuln, outcome = meta.split()
        contract, _, decl = outcome.split('.')
        hands = dict(zip('WNES', deal[2:].split()))
        pbns.append('N:' + ' '.join(hands[s] for s in SEATS))
        leaders.append((SEATS.index(decl) + 1) % 4)
        strains.append('NSHDC'.index(contract[1]))
        leads.append('SHDC'.index(play[0]) * 13 + RANKS.index(play[1]))
    for i in range(len(src_rows)):
        res = dds.solve(strains[i], leaders[i], [], [pbns[i]], 3)
        dd = {c: v[0] for c, v in res.items()}
        best = max(dd.values())
        for c, v in dd.items():
            if v == best:
                safe[i, bucket32(c)] = True
        all_equal[i] = len(set(dd.values())) == 1
        lead_safe[i] = dd.get(leads[i], -1) == best
        if (i + 1) % 2000 == 0:
            print(f'  dd {i + 1}/{len(src_rows)}', flush=True)
    np.savez(cache, safe=safe, all_equal=all_equal, lead_safe=lead_safe)
    return safe, all_equal, lead_safe


test_sets = []
for s in sets:
    label, spec = s.split('=', 1)
    # Split on the LAST ':' -- a Windows drive letter may precede it
    data_dir, srcs = spec.rsplit(':', 1)
    X = np.load(os.path.join(data_dir, 'X.npy')); B = np.load(os.path.join(data_dir, 'B.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy')); src = np.load(os.path.join(data_dir, 'src.npy'))
    start = len(X) - int(len(X) * test_fraction)
    X, B, y, src = X[start:], B[start:], y[start:], src[start:]
    k = min(len(X), max_dd)
    print(f'{label}: {len(X)} held-out rows, double dummy on the first {k}')
    safe, all_equal, lead_safe = dd_for_rows(data_dir, srcs.split(','), src[:k])
    test_sets.append((label, X, B, y, safe, all_equal, lead_safe, k))

print()
print(f"{'model':42s} {'set':10s} {'acc':>6s} {'nll':>6s} {'cover':>6s} {'DDOLAR':>7s} {'ADDOLAR':>8s} {'leader DDOLAR/ADDOLAR':>22s}")
for path in model_paths:
    model = keras.models.load_model(path, compile=False)
    for label, X, B, y, safe, all_equal, lead_safe, k in test_sets:
        p = np.asarray(model.predict([X.astype(np.float16), B.astype(np.float16)], batch_size=4096, verbose=0))
        actual = y.argmax(1)
        acc = np.mean(p.argmax(1) == actual)
        nll = -np.mean(np.log(np.maximum(p[np.arange(len(p)), actual], 1e-7)))
        # BEN's candidate rule on the cards the leader holds
        held = X[:, 10:42] > 0
        q = np.where(held, p, 0)
        q = q / q.sum(1, keepdims=True)
        q = q[:k]
        order = np.argsort(-q, axis=1)
        covered = np.zeros(k, dtype=bool)
        for i in range(k):
            n = 0
            for c in order[i]:
                if q[i, c] < threshold and n >= min_leads:
                    break
                if not held[i, c]:
                    break
                n += 1
                if safe[i, c]:
                    covered[i] = True
                    break
        top_safe = safe[np.arange(k), order[:, 0]]
        m = ~all_equal
        name = os.path.basename(path)[:42]
        print(f"{name:42s} {label:10s} {acc:6.3f} {nll:6.3f} {covered.mean():6.3f} {top_safe.mean():7.3f} {top_safe[m].mean():8.3f}"
              f" {lead_safe.mean():10.3f} / {lead_safe[m].mean():.3f}")
