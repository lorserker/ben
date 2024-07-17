import sys
import datetime
import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, initializers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
import time
import psutil
import GPUtil

# Set logging level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_NUMA_ENABLED'] = '0'

# Redirect standard output and error
logging.getLogger('tensorflow').disabled = True

# Limit the number of CPU threads used
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
print("os.cpu_count()", os.cpu_count())
# TensorFlow thread settings
tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(32)

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
X_train = np.load(os.path.join(bin_dir, 'x.npy'), mmap_mode='r')
y_train = np.load(os.path.join(bin_dir, 'y.npy'), mmap_mode='r')
z_train = np.load(os.path.join(bin_dir, 'z.npy'), mmap_mode='r')

n_examples = X_train.shape[0]
n_sequence = X_train.shape[1]
n_ftrs = X_train.shape[2]
n_bids = y_train.shape[2]
n_alerts = z_train.shape[2]

batch_size = 128  
buffer_size = 12800
epochs = 25  
learning_rate = 0.0005
keep = 0.8
steps_per_epoch = n_examples // batch_size

model_name = f'{system}_{datetime.datetime.now().strftime("%Y-%m-%d")}'

print("-------------------------")
print("Examples for training:   ", n_examples)
print("Model path:              ", model_name )
print("-------------------------")
print("Size input hand:         ", n_ftrs)
print("Number of Sequences:     ", n_sequence)
print("Size output bid:         ", n_bids)
print("Size output alert:       ", n_alerts)
print("-------------------------")
print("dtype X_train:           ", X_train.dtype)
print("dtype y_train:           ", y_train.dtype)
print("dtype z_train:           ", z_train.dtype)
print("-------------------------")
print("Batch size:              ", batch_size)
print("buffer_size:             ", buffer_size)
print("steps_per_epoch          ", steps_per_epoch)
print("-------------------------")
print("Learning rate:           ", learning_rate)
print("Keep:                    ", keep)


lstm_size = 256
n_layers = 3

# Build the model

def build_model(input_shape, lstm_size, n_layers, n_bids, n_alerts):
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float16)
    x = inputs

    for _ in range(n_layers):
        x = layers.LSTM(lstm_size, return_sequences=True,  kernel_initializer=initializers.GlorotUniform(seed=1337))(x)
        x = layers.Dropout(1-keep)(x)
        x = layers.BatchNormalization()(x)

    bid_outputs = layers.TimeDistributed(layers.Dense(n_bids, activation='softmax'), name='bid_output')(x)
    alert_outputs = layers.TimeDistributed(layers.Dense(n_alerts, activation='sigmoid'), name='alert_output')(x)

    model = models.Model(inputs=inputs, outputs=[bid_outputs, alert_outputs])

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                loss={'bid_output': 'categorical_crossentropy', 'alert_output': 'binary_crossentropy'},
                metrics={'bid_output': 'accuracy', 'alert_output': 'accuracy'})

    print(model.summary())

# Create datasets
    print("Reading input")
    def data_generator(X, y, z):
        for xi, yi, zi in zip(X, y, z):
            yield xi, (yi, zi)

    output_signature = (
        tf.TensorSpec(shape=X_train.shape[1:], dtype=X_train.dtype),
        (
            tf.TensorSpec(shape=y_train.shape[1:], dtype=y_train.dtype),
            tf.TensorSpec(shape=z_train.shape[1:], dtype=z_train.dtype)
        )
    )

    print("Shuffling input")
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(X_train, y_train, z_train),
        output_signature=output_signature
    )

    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat()
    return model, train_dataset

print("Building model")
model, train_dataset = build_model((None, n_ftrs), lstm_size, n_layers, n_bids, n_alerts)

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
                print(f"Epoch {epoch + 1}: GPU {gpu.id} memory usage: {gpu.memoryUtil * 100}%")
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

early_stopping_callback = callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1)

print("Training started")
t_start = time.time()

# Training the model
model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch,
          callbacks=[custom_checkpoint_callback, early_stopping_callback, monitor])

# Save the final model with the last epoch number
final_epoch = initial_epoch + epochs -1
final_model_path = os.path.join(checkpoint_dir, f"{model_name}-E{(epochs ):02d}.keras")

model.save(final_model_path)
print("Saved model:", final_model_path)