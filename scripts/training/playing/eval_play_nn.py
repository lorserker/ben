"""
Compare card play models on the deals held out by play_nn_keras.py (the last
val_fraction of the binary files, 5% by default).

Usage: python eval_play_nn.py inputdirectory hand model.keras [model.keras ...] [val_fraction=0.05]

For each model, per trick and in total:
  acc       the model's top card is the card actually played
  acc_held  the same, counting only cards the player holds
  nll       -log p(card actually played), the number that matters when BEN uses the
            model to score samples (lower is better)
"""
import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras

args = [a for a in sys.argv[1:] if '=' not in a]
options = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
if len(args) < 3:
    print("Usage: python eval_play_nn.py inputdirectory hand model.keras [model.keras ...] [val_fraction=0.05]")
    sys.exit(1)

bin_dir, hand, model_paths = args[0], args[1], args[2:]
val_fraction = float(options.get('val_fraction', 0.05))

X_all = np.load(os.path.join(bin_dir, f'X_{hand}.npy'), mmap_mode='r')
y_all = np.load(os.path.join(bin_dir, f'Y_{hand}.npy'), mmap_mode='r')
n_val = int(X_all.shape[0] * val_fraction)
X = np.asarray(X_all[-n_val:])
y = np.asarray(y_all[-n_val:])
played = y.argmax(-1)
held = X[:, :, 0:32] > 0
n_tricks = X.shape[1]
print(f"{hand}: {n_val} held-out deals, {n_tricks} tricks each")

rows = []
for path in model_paths:
    model = keras.models.load_model(path, compile=False)
    p = np.concatenate([np.asarray(model(X[i:i + 2048], training=False)) for i in range(0, n_val, 2048)])
    p_played = np.take_along_axis(p, played[..., None], -1)[..., 0]
    acc = (p.argmax(-1) == played)
    acc_held = (np.where(held, p, -1).argmax(-1) == played)
    nll = -np.log(np.maximum(p_played, 1e-7))
    rows.append((os.path.basename(path), acc, acc_held, nll))

width = max(len(r[0]) for r in rows)
print()
print(f"{'model':{width}s}  {'acc':>6s} {'acc_held':>8s} {'nll':>6s}")
for name, acc, acc_held, nll in rows:
    print(f"{name:{width}s}  {acc.mean():6.4f} {acc_held.mean():8.4f} {nll.mean():6.4f}")

print()
print("acc_held per trick")
print(f"{'model':{width}s}  " + " ".join(f"{t + 1:>6d}" for t in range(n_tricks)))
for name, acc, acc_held, nll in rows:
    print(f"{name:{width}s}  " + " ".join(f"{v:6.4f}" for v in acc_held.mean(0)))

print()
print("nll per trick")
print(f"{'model':{width}s}  " + " ".join(f"{t + 1:>6d}" for t in range(n_tricks)))
for name, acc, acc_held, nll in rows:
    print(f"{name:{width}s}  " + " ".join(f"{v:6.3f}" for v in nll.mean(0)))
