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
    print("Usage: python bidding_nn_keras.py inputdirectory system")
    sys.exit(1)

bin_dir = sys.argv[1]
print(sys.argv)

system = "bidding"
if len(sys.argv) > 2:
    system = sys.argv[2]

# Load training data
#X_train = np.load(os.path.join(bin_dir, 'x.npy'), mmap_mode='r')
#y_train = np.load(os.path.join(bin_dir, 'y.npy'), mmap_mode='r')
X_train = np.load(os.path.join(bin_dir, 'x.npy'))
y_train = np.load(os.path.join(bin_dir, 'y.npy'))

n_examples = X_train.shape[0]
n_sequence = X_train.shape[1]
n_ftrs = X_train.shape[2]
n_bids = y_train.shape[2]

n_cards = X_train.shape[2] - 169

batch_size = 512  
buffer_size = 100000
epochs = 30
learning_rate = 0.001
keep = 0.8
steps_per_epoch = n_examples // batch_size
# If no improvement in validation loss after 10 epochs, stop training
patience = 10

model_name = f'{system}_{datetime.datetime.now().strftime("%Y-%m-%d")}'

lstm_size = 128
n_layers = 3

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
print("buffer_size:             ", buffer_size)
print("steps_per_epoch          ", steps_per_epoch)
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
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

# Create datasets

def create_dataset():
    print("Reading input")
    def data_generator(X, y):
        for xi, yi in zip(X, y):
            yield xi, yi

    output_signature = (
        tf.TensorSpec(shape=X_train.shape[1:], dtype=X_train.dtype),
        tf.TensorSpec(shape=y_train.shape[1:], dtype=y_train.dtype),
    )

    print("Shuffling input", output_signature)
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(X_train, y_train),
        output_signature=output_signature
    )

    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat()

    return train_dataset


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
        self.model.save(save_path)
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

train_dataset = create_dataset()

# Training the model
model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch,
          callbacks=[custom_checkpoint_callback, early_stopping_callback, monitor])

