import sys
import datetime
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Set logging level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_NUMA_ENABLED'] = '0'
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, initializers, metrics
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
import time
import psutil
import GPUtil
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import gpu_epochs

# Redirect standard output and error
logging.getLogger('tensorflow').disabled = True

# Limit the number of CPU threads used
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
print("os.cpu_count()", os.cpu_count())
# TensorFlow thread settings
tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)

#from tensorflow.keras import mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)

# Set TensorFlow to only allocate as much GPU memory as needed
physical_devices = tf.config.list_physical_devices('GPU')
print("physical_devices", physical_devices)
if physical_devices:
    try:
        for device in physical_devices:
            print("set_memory_growth", device)
            tf.config.experimental.set_memory_growth(device, True)
        tf.config.set_visible_devices(physical_devices, 'GPU')
        print("Using GPU: ", physical_devices)
    except RuntimeError as e:
        print(e)
else:
    # Ensure TensorFlow uses only CPU
    tf.config.set_visible_devices([], 'GPU')
    print("Using CPU only")


# Check for correct usage
if len(sys.argv) < 2:
    print("Usage: python bidding_nn_keras.py inputdirectory system [lstm_size=128] [n_layers=3] [batch_size=512] [epochs=30] [learning_rate=0.001] [decay_steps=1000] [keep=0.8] [schedule=exp|cosine] [final_lr=0.05] [gpu_data=1]")
    print("decay_steps: the learning rate is multiplied by 0.96 every decay_steps batches")
    print("keep: 1 - dropout rate after each LSTM layer")
    print("schedule=cosine: cosine decay over all epochs to final_lr times the learning rate, instead of decay_steps")
    print("gpu_data=0: feed batches from memory with model.fit() even when a GPU is present")
    sys.exit(1)

def arg_value(name, default):
    value = next((arg.split('=', 1)[1] for arg in sys.argv[3:] if arg.startswith(name + '=')), None)
    return default if value is None else type(default)(value)

bin_dir = sys.argv[1]
print(sys.argv)

system = "bidding"
if len(sys.argv) > 2:
    system = sys.argv[2]

# Load training data
#X_train = np.load(os.path.join(bin_dir, 'x.npy'), mmap_mode='r')
#y_train = np.load(os.path.join(bin_dir, 'y.npy'), mmap_mode='r')
y_train = np.load(os.path.join(bin_dir, 'y.npy'), mmap_mode='r')
X_train = np.load(os.path.join(bin_dir, 'x.npy'), mmap_mode='r')
# With a GPU the training set is held on it in compact form (see gpu_epochs.py); otherwise, or
# when the array is not of the expected form, it is loaded into memory for model.fit()
gpu_inputs = None
if physical_devices and arg_value('gpu_data', 1):
    gpu_inputs = gpu_epochs.compact_inputs(X_train, y_train.shape[2])
    print("Training data on the GPU:", gpu_inputs is not None)
if gpu_inputs is None:
    X_train = np.load(os.path.join(bin_dir, 'x.npy'))
    y_train = np.load(os.path.join(bin_dir, 'y.npy'))

n_examples = X_train.shape[0]
n_sequence = X_train.shape[1]
n_ftrs = X_train.shape[2]
n_bids = y_train.shape[2]

n_cards = X_train.shape[2] - 169

batch_size = arg_value('batch_size', 512)
epochs = arg_value('epochs', 30)
learning_rate = arg_value('learning_rate', 0.001)
decay_steps = arg_value('decay_steps', 1000)
keep = arg_value('keep', 0.8)
steps_per_epoch = n_examples // batch_size
# If no improvement in validation loss after 10 epochs, stop training
patience = 10

model_name = f'{system}_{datetime.datetime.now().strftime("%Y-%m-%d")}'

lstm_size = arg_value('lstm_size', 128)
n_layers = arg_value('n_layers', 3)

print("-------------------------")
print("Examples for training:   ", n_examples)
print("Model path:              ", model_name )
print("-------------------------")
print("Size input hand:         ", n_ftrs)
print("Number of Cards:         ", n_cards)
print("Number of Sequences:     ", n_sequence)
print("Size output bid:         ", n_bids)
print("-------------------------")
print("dtype X_train:           ", X_train.dtype)
print("dtype y_train:           ", y_train.dtype)
print("-------------------------")
print("Batch size:              ", batch_size)
print("steps_per_epoch          ", steps_per_epoch)
print("decay_steps              ", decay_steps)
print("patience                 ", patience)
print("-------------------------")
print("Learning rate:           ", learning_rate)
print("Keep:                    ", keep)
print("-------------------------")
print("lstm_size:               ", lstm_size)
print("n_layers:                ", n_layers)

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Build the model
def build_model(input_shape, lstm_size, n_layers, n_bids):
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float16)
    x = inputs

    for i in range(n_layers):
        x = layers.LSTM(units=lstm_size, return_sequences=True,  kernel_initializer=initializers.GlorotUniform(seed=seed_value))(x)
        x = layers.Dropout(1-keep)(x)
        x = layers.BatchNormalization()(x)

    bid_outputs = layers.TimeDistributed(layers.Dense(n_bids, activation='softmax'), name='bid_output')(x)

    model = models.Model(inputs=inputs, outputs=bid_outputs)

    # Custom loss function integrated into the model compilation
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate_schedule),
                loss={'bid_output': 'categorical_crossentropy'},
                metrics={'bid_output': 'accuracy'})     
    print(model.summary())

    return model

# Learning rate scheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
learning_rate_schedule = ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=decay_steps,
    decay_rate=0.96,
    staircase=True
)
if arg_value('schedule', 'exp') == 'cosine':
    # Half a cosine from learning_rate down to final_lr * learning_rate over the whole training
    from tensorflow.keras.optimizers.schedules import CosineDecay
    learning_rate_schedule = CosineDecay(initial_learning_rate=learning_rate, decay_steps=steps_per_epoch * epochs,
                                         alpha=arg_value('final_lr', 0.05))

# Create datasets

def create_dataset():
    # Yield whole batches from a fresh permutation each epoch. Feeding single
    # examples through a Python generator kept the GPU waiting on one CPU core.
    def batch_generator():
        while True:
            order = np.random.permutation(n_examples)
            for start in range(0, n_examples - batch_size + 1, batch_size):
                # Sorted indices make the gather from the big array much faster
                idx = np.sort(order[start:start + batch_size])
                yield X_train[idx], y_train[idx]

    output_signature = (
        tf.TensorSpec(shape=(batch_size,) + X_train.shape[1:], dtype=X_train.dtype),
        tf.TensorSpec(shape=(batch_size,) + y_train.shape[1:], dtype=y_train.dtype),
    )

    print("Batching input", output_signature)
    train_dataset = tf.data.Dataset.from_generator(batch_generator, output_signature=output_signature)

    return train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


print("Building model")
model  = build_model((None, n_ftrs), lstm_size, n_layers, n_bids)

# Define the path to save model weights
checkpoint_dir = "model"
os.makedirs(checkpoint_dir, exist_ok=True)

# Load the latest saved model if it exists
initial_epoch = 0

# Check for existing checkpoints and set the initial epoch
checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.keras') and system in f])
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    model_path = os.path.join(checkpoint_dir, latest_checkpoint)
    initial_epoch = int(latest_checkpoint.split('-E')[1].split('.')[0]) 
    print(f"Loading saved model from {model_path}, starting at epoch {initial_epoch+1}")
    model = load_model(model_path)
    epochs = epochs - initial_epoch

if not epochs > 0:
    print("Model training complete after", initial_epoch, "epochs")
    sys.exit(0)

initial_epoch += 1

# Train with the LSTM layers unrolled over the fixed number of bidding rounds and the training
# step compiled by XLA. On the GPU that step is several times faster than the cuDNN LSTM
# kernel (5 ms against 29 ms for a batch of 256 on an RTX 5080); unrolling without XLA is
# slower than cuDNN. Keras only compiles with XLA when no layer can use cuDNN, which it
# decides when the layer is created, so supports_jit is set here too. The weights are the
# same either way. The layers are switched back while saving, so a saved model still takes
# auctions of any length.
lstm_layers = [layer for layer in model.layers if isinstance(layer, layers.LSTM)]

def set_unroll(value):
    for layer in lstm_layers:
        layer.unroll = value
        layer.supports_jit = value

def save_model(save_path):
    set_unroll(False)
    try:
        model.save(save_path)
    finally:
        set_unroll(True)

set_unroll(True)
model.jit_compile = True
print("LSTM layers unrolled for training:", len(lstm_layers), "XLA:", model.jit_compile)

# Define callbacks
class ResourceMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: CPU usage: {psutil.cpu_percent()}%")
        print(f"Epoch {epoch + 1}: Memory usage: {psutil.virtual_memory().percent}%")

        # Get GPU stats
        gpus = GPUtil.getGPUs()
        if gpus:
            for gpu in gpus:
                print(f"Epoch {epoch + 1}: GPU {gpu.id} usage: {gpu.load * 100}%")
                print(f"Epoch {epoch + 1}: GPU {gpu.id} memory usage: { round(gpu.memoryUtil*100,1)}%")
        else:
            print(f"Epoch {epoch + 1}: No GPU detected.")
# Include this callback in the fit method
monitor = ResourceMonitor()

class CustomModelCheckpoint(Callback):
    def __init__(self, save_path, initial_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.save_path = save_path
        self.initial_epoch = initial_epoch
        self.t_start = None

    def on_epoch_begin(self, epoch, logs=None):
        # Start time of the epoch
        self.t_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        save_path = self.save_path.format(epoch=epoch + self.initial_epoch)
        print()
        print(f"Saving model to {save_path}")
        save_model(save_path)
        epoch_duration = time.time() - self.t_start
        print(f'Epoch took {epoch_duration:0.4f} seconds')

# Define the custom checkpoint callback
custom_checkpoint_callback = CustomModelCheckpoint(
    save_path=os.path.join(checkpoint_dir, f"{model_name}-E{{epoch:02d}}.keras"),
    initial_epoch=initial_epoch
)

early_stopping_callback = callbacks.EarlyStopping(monitor='loss', patience=patience, verbose=1, restore_best_weights=True)

print("Training started")
t_start = time.time()

training_callbacks = [custom_checkpoint_callback, early_stopping_callback, monitor]
if gpu_inputs is not None:
    # The one-hot targets go to the GPU as class indexes when they are exactly one-hot
    y_classes = gpu_epochs.compact_one_hot(y_train)
    if y_classes is not None:
        gpu_epochs.fit(model, gpu_inputs, y_classes, batch_size, epochs, steps_per_epoch,
                       callbacks=training_callbacks, n_bids=n_bids, one_hot_targets=(n_bids, y_train.dtype))
    else:
        gpu_epochs.fit(model, gpu_inputs, y_train, batch_size, epochs, steps_per_epoch,
                       callbacks=training_callbacks, n_bids=n_bids)
else:
    train_dataset = create_dataset()
    model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=training_callbacks)

