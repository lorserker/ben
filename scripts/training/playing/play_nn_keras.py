import sys
import datetime
import os
import time
import numpy as np

# Set logging level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
from keras import layers, models, optimizers, callbacks

# Set TensorFlow to only allocate as much GPU memory as needed
physical_devices = tf.config.list_physical_devices('GPU')
print("physical_devices", physical_devices)
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("Using GPU: ", physical_devices)
else:
    print("Using CPU only")

# Check for correct usage
if len(sys.argv) < 3:
    print("Usage: python play_nn_keras.py inputdirectory hand [key=value ...]")
    print("  hand: decl_nt, decl_suit, dummy_nt, dummy_suit, lefty_nt, lefty_suit, righty_nt, righty_suit")
    print("  options: lstm_size=128 n_layers=3 dense_layers=0 dense_units=512 epochs=30 batch_size=1024 learning_rate=0.001")
    print("           keep=0.8 val_fraction=0.05 patience=3 max_examples=0 name=<hand>")
    sys.exit(1)

bin_dir = sys.argv[1]
hand = sys.argv[2]
print(sys.argv)

options = dict(arg.split('=', 1) for arg in sys.argv[3:])
lstm_size = int(options.get('lstm_size', 128))
n_layers = int(options.get('n_layers', 3))
# Optional Dense layers per trick before the LSTM layers
dense_layers = int(options.get('dense_layers', 0))
dense_units = int(options.get('dense_units', 512))
epochs = int(options.get('epochs', 30))
batch_size = int(options.get("batch_size", 1024))
learning_rate = float(options.get('learning_rate', 0.001))
keep = float(options.get('keep', 0.8))
# The last deals in the file are held out for validation; eval_play_nn.py uses the same split
val_fraction = float(options.get('val_fraction', 0.05))
patience = int(options.get('patience', 3))
max_examples = int(options.get('max_examples', 0))
name = options.get('name', hand)

X_all = np.load(os.path.join(bin_dir, f'X_{hand}.npy'), mmap_mode='r')
y_all = np.load(os.path.join(bin_dir, f'Y_{hand}.npy'), mmap_mode='r')
if max_examples:
    X_all, y_all = X_all[:max_examples], y_all[:max_examples]

n_all = X_all.shape[0]
n_val = int(n_all * val_fraction)
n_train = n_all - n_val
n_ftrs = X_all.shape[2]
n_card = y_all.shape[2]

print("Loading data into memory")
t = time.time()
X_train, y_train = np.asarray(X_all[:n_train]), np.asarray(y_all[:n_train])
X_val, y_val = np.asarray(X_all[n_train:]), np.asarray(y_all[n_train:])
print(f"Loaded in {time.time() - t:0.1f} seconds")

steps_per_epoch = n_train // batch_size
model_name = f'{name}_{datetime.datetime.now().strftime("%Y-%m-%d")}'

print("-------------------------")
print("Examples for training:   ", n_train)
print("Examples for validation: ", n_val)
print("Model path:              ", model_name)
print("-------------------------")
print("Tricks per example:      ", X_all.shape[1])
print("Size input:              ", n_ftrs)
print("Size output card:        ", n_card)
print("-------------------------")
print("Dense layers:            ", dense_layers, "x", dense_units)
print("LSTM size:               ", lstm_size)
print("Layers:                  ", n_layers)
print("Batch size:              ", batch_size)
print("steps_per_epoch          ", steps_per_epoch)
print("Learning rate:           ", learning_rate)
print("Keep:                    ", keep)


def batches(X, y, batch_size, shuffle):
    """Batches gathered from the in-memory arrays, prepared in parallel with training."""
    with tf.device('/CPU:0'):
        X_t, y_t = tf.constant(X), tf.constant(y)
    ds = tf.data.Dataset.range(len(X))
    if shuffle:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=shuffle)
    ds = ds.map(lambda i: (tf.gather(X_t, i), tf.gather(y_t, i)), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model():
    # A recurrent model, so the card played can depend on everything seen earlier in
    # the play (signals, show-outs, suits played), not only on the last trick.
    # The output is a plain Dense layer: it works per trick on the sequence, and
    # TimeDistributed is broken in Keras 3.15 for an unknown number of tricks.
    inputs = keras.Input(shape=(None, n_ftrs), dtype='float16', name='x_play')
    x = inputs
    for _ in range(dense_layers):
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(1 - keep)(x)
    for _ in range(n_layers):
        x = layers.LSTM(lstm_size, return_sequences=True)(x)
        x = layers.Dropout(1 - keep)(x)
    card = layers.Dense(n_card, activation='softmax', name='card')(x)
    model = models.Model(inputs=inputs, outputs=card)
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


checkpoint_dir = "model"
os.makedirs(checkpoint_dir, exist_ok=True)

# Continue from the latest checkpoint for this name, if there is one
initial_epoch = 0
checkpoints = sorted((int(f[len(model_name) + 2:-len('.keras')]), f) for f in os.listdir(checkpoint_dir)
                     if f.startswith(model_name + '-E') and f.endswith('.keras') and f[len(model_name) + 2:-len('.keras')].isdigit())
if checkpoints:
    initial_epoch, latest = checkpoints[-1]
    model_path = os.path.join(checkpoint_dir, latest)
    print(f"Loading saved model from {model_path}, continuing after epoch {initial_epoch}")
    model = keras.models.load_model(model_path)
else:
    model = build_model()
model.summary()

if initial_epoch >= epochs:
    print("Model training complete after", initial_epoch, "epochs")
    sys.exit(0)


class EpochTimer(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.t_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print(f'Epoch took {time.time() - self.t_start:0.1f} seconds')


train_batches = batches(X_train, y_train, batch_size, shuffle=True)
val_batches = batches(X_val, y_val, 1024, shuffle=False)

print("Training started")
model.fit(train_batches, validation_data=val_batches,
          epochs=epochs, initial_epoch=initial_epoch,
          callbacks=[
              callbacks.ModelCheckpoint(os.path.join(checkpoint_dir, f"{model_name}-E{{epoch:02d}}.keras")),
              callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=1, verbose=1),
              callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
              callbacks.CSVLogger(os.path.join(checkpoint_dir, f"{model_name}.csv"), append=True),
              EpochTimer(),
          ])

final_model_path = os.path.join(checkpoint_dir, f"{model_name}-best.keras")
model.save(final_model_path)
print("Saved model with the best validation loss:", final_model_path)
