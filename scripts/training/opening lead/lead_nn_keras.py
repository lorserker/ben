import sys
import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, initializers, losses
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from tensorflow.data import Dataset
import time 

# Set logging level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Limit the number of CPU threads used
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
    print("Usage: python lead_nn_keras.py inputdirectory name")
    sys.exit(1)

bin_dir = sys.argv[1]
print(sys.argv)

system = "OpeningLead"
if len(sys.argv) > 2:
    system = sys.argv[2]

X_train = np.load(os.path.join(bin_dir, 'x.npy'), mmap_mode='r')
B_train = np.load(os.path.join(bin_dir, 'B.npy'), mmap_mode='r')
y_train = np.load(os.path.join(bin_dir, 'y.npy'), mmap_mode='r')

n_examples = X_train.shape[0]
n_sequence = X_train.shape[1]
n_ftrs = X_train.shape[1]
n_cards = 32
n_bi = B_train.shape[1]

batch_size = 64  
buffer_size = 5000
epochs = 200  
learning_rate = 0.0005
keep = 0.6

steps_per_epoch = n_examples // batch_size

model_name = f'{system}_{datetime.datetime.now().strftime("%Y-%m-%d")}'

print("-------------------------")
print("Examples for training:   ", n_examples)
print("Model path:              ", model_name )
print("-------------------------")
print("Size input hand:         ", n_ftrs)
print("Size input bidding:      ", n_bi)
print("Number of Sequences:     ", n_sequence)
print("Size output card:        ", n_cards)
print("-------------------------")
print("dtype X_train:           ", X_train.dtype)
print("dtype y_train:           ", y_train.dtype)
print("dtype B_train:           ", B_train.dtype)
print("-------------------------")
print("Batch size:              ", batch_size)
print("buffer_size:             ", buffer_size)
print("steps_per_epoch          ", steps_per_epoch)
print("-------------------------")
print("Learning rate:           ", learning_rate)
print("Keep:                    ", keep)

n_hidden_units = 512
n_layers = 3

# Build the model

def build_model(input_shape, n_bi, n_hidden_units, n_cards):
    X_input = layers.Input(shape=(input_shape,), dtype=tf.float16, name='X_input')
    B_input = layers.Input(shape=(n_bi,), dtype=tf.float16, name='B_input')
    
    x = layers.Concatenate()([X_input, B_input])
    for _ in range(n_layers):
    
        x = layers.Dense(n_hidden_units, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1337))(x)
        x = layers.Dropout(1-keep)(x)
    
    # Define the output layer
    lead_logit = layers.Dense(n_cards, name='lead_logit')(x)
    lead_softmax =  layers.Softmax(name='lead_softmax')(lead_logit)
    
    model = models.Model(inputs=[X_input, B_input], outputs=lead_softmax)

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])

    print(model.summary())

# Create datasets
    train_dataset = Dataset.from_tensor_slices(({'X_input': X_train, 'B_input': B_train}, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat()
    return model, train_dataset

print("Building model")
model, train_dataset = build_model(n_ftrs, n_bi, n_hidden_units, n_cards)

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
    initial_epoch = int(latest_checkpoint.split('-E')[1].split('.')[0]) + 1
    print(f"Loading saved model from {model_path}, starting at epoch {initial_epoch+1}")
    model = load_model(model_path)
    epochs = epochs - initial_epoch

if not epochs > 0:
    print("Model training complete after", initial_epoch, "epochs")
    sys.exit(0)

initial_epoch += 1
# Define callbacks

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
    save_path=os.path.join(checkpoint_dir, f"{model_name}-E{{epoch:03d}}.keras"),
    initial_epoch=initial_epoch
)


early_stopping_callback = callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1)

print("Training started")
# Training the model
model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch,
          callbacks=[custom_checkpoint_callback, early_stopping_callback])

# Save the final model with the last epoch number
final_epoch = initial_epoch + epochs -1
final_model_path = os.path.join(checkpoint_dir, f"{model_name}-E{(final_epoch):02d}.keras")

model.save(final_model_path)
print("Saved model:", final_model_path)