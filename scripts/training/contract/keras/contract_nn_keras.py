import sys
import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, initializers, losses
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model

# Set logging level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    print("Usage: python contract_nn_keras.py inputdirectory system")
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
u_train = np.load(os.path.join(bin_dir, 'u.npy'), mmap_mode='r')

n_examples = X_train.shape[0]
n_contract = y_train.shape[1]
n_ftrs = X_train.shape[1]

output_dim_bool1 = z_train.shape[1]
output_dim_tricks = u_train.shape[1]
output_dim_contract = y_train.shape[1]

batch_size = 128  
buffer_size = 10000
epochs = 20  
learning_rate = 0.0005
keep = 0.8
steps_per_epoch = n_examples // batch_size

model_name = f'{system}_{datetime.datetime.now().strftime("%Y-%m-%d")}'

print("-------------------------")
print("Examples for training:   ", n_examples)
print("Model path:              ", model_name )
print("-------------------------")
print("Size input hand:         ", n_ftrs)
print("Size output doubled:     ", output_dim_bool1)
print("Size output tricks:      ", output_dim_tricks)
print("Size output contract:    ", output_dim_contract)
print("-------------------------")
print("dtype X_train:           ", X_train.dtype)
print("dtype y_train:           ", y_train.dtype)
print("dtype z_train:           ", z_train.dtype)
print("dtype u_train:           ", u_train.dtype)
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

def build_model(input_shape, lstm_size, n_layers):
    print("input_shape",input_shape)
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float16)
    x = layers.Dense(lstm_size, activation='tanh', kernel_initializer='truncated_normal')(inputs)
    x = layers.Dense(64, activation='tanh', kernel_initializer='truncated_normal')(x)
    
    contract_output = layers.Dense(output_dim_contract, activation='softmax', name='contract_output')(x)
    bool1_output = layers.Dense(output_dim_bool1, activation='sigmoid', name='bool1_output')(x)
    tricks_output = layers.Dense(output_dim_tricks, activation='sigmoid', name='tricks_output')(x)
    
    model = models.Model(inputs=inputs, outputs=[contract_output, bool1_output, tricks_output])
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss={
                      'contract_output': losses.CategoricalCrossentropy(),
                      'bool1_output': losses.BinaryCrossentropy(),
                      'tricks_output': losses.BinaryCrossentropy()
                  },
                  metrics={
                      'contract_output': 'accuracy',
                      'bool1_output': 'accuracy',
                      'tricks_output': 'accuracy',
                  })
    print(model.summary())

# Create datasets
    print("Reading input")
    def data_generator(X, y, z, u):
        for xi, yi, zi, ui in zip(X, y, z, u):
            yield (xi, {'contract_output': yi, 'bool1_output': zi, 'tricks_output': ui})

    output_signature = (
        tf.TensorSpec(shape=(n_ftrs,), dtype=X_train.dtype),
        {
            'contract_output': tf.TensorSpec(shape=(output_dim_contract,), dtype=y_train.dtype),
            'bool1_output': tf.TensorSpec(shape=(1,), dtype=z_train.dtype),
            'tricks_output': tf.TensorSpec(shape=(output_dim_tricks,), dtype=u_train.dtype)
        }
    )

    print("Shuffling input")
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(X_train, y_train, z_train, u_train),
        output_signature=output_signature
    )
    
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat()
    return model, train_dataset

print("Building model")
model, train_dataset = build_model(( n_ftrs,), lstm_size, n_layers)

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

class CustomModelCheckpoint(Callback):
    def __init__(self, save_path, initial_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.save_path = save_path
        self.initial_epoch = initial_epoch

    def on_epoch_end(self, epoch, logs=None):
        save_path = self.save_path.format(epoch=epoch + self.initial_epoch)
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

# Define the custom checkpoint callback
custom_checkpoint_callback = CustomModelCheckpoint(
    save_path=os.path.join(checkpoint_dir, f"{model_name}-E{{epoch:02d}}.keras"),
    initial_epoch=initial_epoch
)

early_stopping_callback = callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1)

print("Training started")
# Training the model
model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch,
          callbacks=[custom_checkpoint_callback, early_stopping_callback])

# Save the final model with the last epoch number
final_epoch = initial_epoch + epochs -1
final_model_path = os.path.join(checkpoint_dir, f"{model_name}-E{(epochs):02d}.keras")

model.save(final_model_path)
print("Saved model:", final_model_path)