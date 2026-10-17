"""
Train an opening-lead model on data from gen_lead_data.py.

Usage: python lead_train.py <name> <data_dir> [<data_dir> ...] [key=value ...]
  data_dir: a folder with X.npy, B.npy, y.npy (e.g. lead_v3/wbc/suit). Several folders are
            concatenated -- e.g. WBC and Jack together.
  options:  layers=3 units=512 keep=0.7 epochs=60 batch_size=1024 learning_rate=0.001
            test_fraction=0.05 val_fraction=0.03 patience=4

The LAST test_fraction of EACH folder is held out (never trained on, never used for early
stopping), so every variant is tested on the same deals by lead_eval.py. Early stopping uses a
random val_fraction of the remaining rows. The inputs are a list [X, B] concatenated in that
order -- never a dict, whose order changed between Keras 3.6 and 3.7.
"""
import os
import sys
import time
import datetime
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras
from keras import layers, callbacks

for d in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(d, True)

args = [a for a in sys.argv[1:] if '=' not in a]
opts = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
if len(args) < 2:
    print(__doc__)
    sys.exit(1)
name, dirs = args[0], args[1:]
n_layers = int(opts.get('layers', 3))
units = int(opts.get('units', 512))
keep = float(opts.get('keep', 0.7))
epochs = int(opts.get('epochs', 60))
batch_size = int(opts.get('batch_size', 1024))
lr = float(opts.get('learning_rate', 0.001))
test_fraction = float(opts.get('test_fraction', 0.05))
val_fraction = float(opts.get('val_fraction', 0.03))
patience = int(opts.get('patience', 4))

Xs, Bs, ys = [], [], []
for d in dirs:
    X = np.load(os.path.join(d, 'X.npy')); B = np.load(os.path.join(d, 'B.npy')); y = np.load(os.path.join(d, 'y.npy'))
    n_train = len(X) - int(len(X) * test_fraction)
    Xs.append(X[:n_train]); Bs.append(B[:n_train]); ys.append(y[:n_train])
    print(f'{d}: {len(X)} rows, {n_train} for training, {len(X) - n_train} held out')
X, B, y = np.concatenate(Xs), np.concatenate(Bs), np.concatenate(ys)

rng = np.random.default_rng(1337)
perm = rng.permutation(len(X))
n_val = int(len(X) * val_fraction)
val, tr = perm[:n_val], perm[n_val:]

def dataset(idx, shuffle):
    ds = tf.data.Dataset.from_tensor_slices(((X[idx], B[idx]), y[idx]))
    if shuffle:
        ds = ds.shuffle(min(len(idx), 1_000_000), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

x_in = keras.Input(shape=(X.shape[1],), dtype='float16', name='X_input')
b_in = keras.Input(shape=(B.shape[1],), dtype='float16', name='B_input')
h = layers.Concatenate()([x_in, b_in])
for _ in range(n_layers):
    h = layers.Dense(units, activation='relu')(h)
    h = layers.Dropout(1 - keep)(h)
out = layers.Dense(y.shape[1], activation='softmax', name='lead_softmax')(h)
model = keras.Model(inputs=[x_in, b_in], outputs=out)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

os.makedirs('model', exist_ok=True)
stamp = datetime.datetime.now().strftime('%Y-%m-%d')
t0 = time.time()
model.fit(dataset(tr, True), validation_data=dataset(val, False), epochs=epochs, verbose=2,
          callbacks=[
              callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=1, verbose=1),
              callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
              callbacks.CSVLogger(os.path.join('model', f'{name}_{stamp}.csv')),
          ])
path = os.path.join('model', f'{name}_{stamp}.keras')
model.save(path)
print(f'Saved {path} after {time.time() - t0:.0f}s')
