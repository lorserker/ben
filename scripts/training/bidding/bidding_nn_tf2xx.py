import sys
import os
sys.path.append('../../../src')

import numpy as np
import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from batcher import Batcher

if len(sys.argv) < 2:
    print("Usage: python bidding_nn inputdirectory")
    sys.exit(1)

bin_dir = sys.argv[1]

model_path = './model/bidding.h5'

# Batch size is 1 so we can reset state between each sequence
batch_size = 32
display_step = 10000
epochs = 1

X_train = np.load(os.path.join(bin_dir, 'x.npy'))
y_train = np.load(os.path.join(bin_dir, 'y.npy'))

print("Input shape:  ", X_train.shape)
print("Output shape: ", y_train.shape)
n_examples = y_train.shape[0]
n_sequence_length = X_train.shape[1]
n_ftrs = X_train.shape[2]
n_bids = y_train.shape[2]

print("Size input hand:         ", n_ftrs)
print("Sequence length:         ", n_sequence_length)
print("Examples for training:   ", n_examples)
print("Batch size:              ", batch_size)
n_iterations = round(((n_examples / batch_size) * epochs) / 1000) * 1000
print("Iterations               ", n_iterations)
print("Model path:              ", model_path)

lstm_size = 128
n_layers = 3
keep_prob = 0.8  # Adjust the dropout rate as needed

input_shape = (n_sequence_length, n_ftrs)
# Create a Sequential model with stateful LSTM and specify batch_input_shape
model = Sequential()
model.add(LSTM(lstm_size, dropout=1 - keep_prob, stateful=True, return_sequences=True, batch_input_shape=(batch_size,) + input_shape))

for _ in range(n_layers - 1):  # Subtract 1 to account for the first LSTM layer
    model.add(LSTM(lstm_size, dropout=1 - keep_prob, stateful=True, return_sequences=True))

model.add(Dense(n_bids, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))

model.summary()

# Create Batcher instances
batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, batch_size)

for i in range(n_iterations):
    x_batch, y_batch = batch.next_batch([X_train, y_train])
    if (i != 0) and i % display_step == 0:
        # Evaluate the model on the test data
        x_cost, y_cost = cost_batch.next_batch([X_train, y_train])
        c_train = model.evaluate(x_cost, y_cost, verbose=0)
        print('{} {}. c_train={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, c_train))
        sys.stdout.flush()
        model.save(model_path, overwrite=True)  

    model.train_on_batch(x_batch, y_batch)

model.save(model_path, overwrite=True)
