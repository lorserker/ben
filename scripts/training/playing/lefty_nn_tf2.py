import sys
import os
import datetime
import numpy as np
import logging
sys.path.append('../../../src')
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from batcher import Batcher

model_path = './model/lefty.h5'

batch_size = 64
display_step = 10000
epochs = 5

X_train = np.load('./lefty_bin/X.npy')
Y_train = np.load('./lefty_bin/Y.npy')

n_examples = Y_train.shape[0]
n_ftrs = X_train.shape[2]
n_cards = Y_train.shape[2]

print("Size input hand:         ", n_ftrs)
print("Examples for training:   ", n_examples)
print("Batch size:              ", batch_size)
n_iterations = round(((n_examples / batch_size) * epochs) / 1000) * 1000
print("Iterations               ", n_iterations)
print("Model path:              ", model_path)

lstm_size = 128
n_layers = 3

# Input placeholders
x_ftrs = Input(shape=(None, n_ftrs), name='seq_in')

# Define the model
lstm_layer = LSTM(lstm_size, return_sequences=True, dropout=0.2)
output = lstm_layer(x_ftrs)
output = Dense(n_cards, activation='softmax')(output)

model = Model(inputs=x_ftrs, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0003), loss='categorical_crossentropy')


# Create Batcher instances
batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, batch_size)

# Training loop
for i in range(n_iterations):
    x_batch, y_batch = batch.next_batch([X_train, Y_train])

    if (i != 0) and i % display_step == 0:
        x_cost, y_cost = cost_batch.next_batch([X_train, Y_train])
        c_train = model.evaluate(x_cost, y_cost, batch_size=batch_size, verbose=2)
        l_train = model.predict(x_cost, batch_size=batch_size, verbose=0)
        accuracy = np.mean(np.argmax(l_train, axis=1) == np.argmax(y_cost, axis=1))
        print('{} {}. c_train={} cost={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, c_train, accuracy))
        sys.stdout.flush()
        model.save(model_path)

    model.fit(x_batch, y_batch, batch_size, epochs, verbose=0)

model.save(model_path)