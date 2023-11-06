import sys
import os
sys.path.append('../../../src')

import datetime
import numpy as np
import logging
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Dropout
from keras.optimizers import Adam
from batcher import Batcher

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Command line arguments
if len(sys.argv) < 2:
    print("Usage: python binfo_nn inputdirectory")
    sys.exit(1)

bin_dir = sys.argv[1]

model_path = './model/binfo.h5' 

batch_size = 64
display_step = 1000
epochs = 5

X_train = np.load(os.path.join(bin_dir, 'X.npy'))
HCP_train = np.load(os.path.join(bin_dir, 'HCP.npy'))
SHAPE_train = np.load(os.path.join(bin_dir, 'SHAPE.npy'))

n_examples = X_train.shape[0]
n_ftrs = X_train.shape[2]
n_dim_hcp = HCP_train.shape[2]
n_dim_shape = SHAPE_train.shape[2]

print("Size input hand:         ", n_ftrs)
print("Examples for training:   ", n_examples)
print("Batch size:              ", batch_size)
n_iterations = round(((n_examples / batch_size) * epochs) / display_step) * display_step
print("Iterations               ", n_iterations)
print("Model path:              ",model_path)

lstm_size = 128
n_layers = 3
keep_prob = 0.8  # Adjust the dropout rate as needed

# Define input layers
seq_in = Input(shape=(None, n_ftrs), name='seq_in')

# Define the LSTM layers
x = seq_in
for _ in range(n_layers):
    x = LSTM(lstm_size, return_sequences=True, dropout=1 - keep_prob)(x)

# Define separate output layers for hcp and shape
out_hcp_seq = Dense(n_dim_hcp, activation='linear', name='out_hcp_seq')(x)
out_shape_seq = Dense(n_dim_shape, activation='linear', name='out_shape_seq')(x)

# Create a model with multiple inputs and outputs
model = Model(inputs=[seq_in], outputs=[out_hcp_seq, out_shape_seq])

# Compile the model with separate losses for each output
model.compile(optimizer=Adam(learning_rate=0.001),
              loss={'out_hcp_seq': 'mean_absolute_error', 'out_shape_seq': 'mean_absolute_error'},
              loss_weights={'out_hcp_seq': 1.0, 'out_shape_seq': 1.0})

model.summary()

batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, batch_size)

# Train the model
for i in range(n_iterations):

    x_batch, hcp_batch, shape_batch = batch.next_batch([X_train, HCP_train, SHAPE_train])
    if (i != 0) and i % display_step == 0:
        x_cost, hcp_cost, shape_cost = cost_batch.next_batch([X_train, HCP_train, SHAPE_train])
        c_train = model.evaluate(x_cost, [hcp_cost, shape_cost], batch_size=32, verbose=0)
        print('{} {}. c_train={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i+1, c_train))

        p_hcp_seq, p_shape_seq = model.predict(x_cost, batch_size=32, verbose=0)
        
        hcp_diff = np.abs(hcp_cost - p_hcp_seq)
        shape_diff = np.abs(shape_cost - p_shape_seq)
        
        print(
            np.mean(hcp_diff[:, 0, :]),
            np.mean(hcp_diff[:, 1, :]),
            np.mean(hcp_diff[:, 2, :]),
            np.mean(hcp_diff[:, 3, :]),
            np.mean(hcp_diff[:, -1, :]))
        print(
            np.mean(shape_diff[:, 0, :]),
            np.mean(shape_diff[:, 1, :]),
            np.mean(shape_diff[:, 2, :]),
            np.mean(shape_diff[:, 3, :]),
            np.mean(shape_diff[:, -1, :]))

        sys.stdout.flush()
        model.save(model_path)

    model.fit(x_batch, {'out_hcp_seq': hcp_batch, 'out_shape_seq': shape_batch}, epochs, verbose=0)

# Save the model
model.save(model_path)
