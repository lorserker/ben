import sys
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

model_path = 'model/bidding.h5'

batch_size = 64
display_step = 1000
epochs = 1

X_train = np.load(os.path.join(bin_dir, 'x.npy'))
y_train = np.load(os.path.join(bin_dir, 'y.npy'))

print("Input shape:  ", X_train.shape)
print("Output shape: ", y_train.shape)
n_examples = y_train.shape[0]
n_ftrs = X_train.shape[1]
n_bids = y_train.shape[1]

print("Size input hand:         ", n_ftrs)
print("Examples for training:   ", n_examples)
print("Batch size:              ", batch_size)
n_iterations = round(((n_examples / batch_size) * epochs) / 1000) * 1000
print("Iterations               ", n_iterations)
print("Model path:              ", model_path)

lstm_size = 128
n_layers = 3

model = Sequential()

for _ in range(n_layers):
    model.add(LSTM(lstm_size, dropout=0.2, stateful=True))

model.add(Dense(n_bids, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))

# Create Batcher instances
batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, display_step)

for i in range(n_iterations):
    x_batch, y_batch = batch.next_batch([X_train, y_train])
    if (i != 0) and i % display_step == 0:
        x_cost, y_cost = cost_batch.next_batch([X_train, y_train])
        c_train = model.evaluate(x_cost, y_cost, verbose=0)
        print('{} {}. c_train={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, c_train))
        sys.stdout.flush()
        model.save(model_path, overwrite=True)  
        model.reset_states()

    model.fit(x_batch, y_batch, batch_size, epochs, verbose=0)

model.save(model_path, overwrite=True)
