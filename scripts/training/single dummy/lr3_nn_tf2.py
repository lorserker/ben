import sys
import numpy as np
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import datetime
from batcher import Batcher

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_path = './model/single_dummy.h5'

seed = 1337

batch_size = 64
display_step = 10000
epochs = 1

X_train = np.load('./lr3_bin/X.npy')
y_train = np.load('./lr3_bin/y.npy')

n_examples = X_train.shape[0]
n_ftrs = X_train.shape[1]
n_tricks = y_train.shape[1]

print("Size input hand:         ", n_ftrs)
print("Examples for training:   ", n_examples)
print("Batch size:              ", batch_size)
n_iterations = round(((n_examples / batch_size) * epochs) / 1000) * 1000
print("Iterations               ", n_iterations)
print("Model path:              ", model_path)

n_hidden_units = 512

model = Sequential()

model.add(Dense(n_hidden_units, input_dim=n_ftrs, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(n_hidden_units, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(n_hidden_units, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(n_hidden_units, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(n_tricks, kernel_initializer='glorot_uniform', activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

# Train the Keras model
batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, batch_size)

for i in range(n_iterations):
    x_batch, y_batch = batch.next_batch([X_train, y_train])
    if (i != 0) and i % display_step == 0:
        x_cost, y_cost = cost_batch.next_batch([X_train, y_train])
        c_train = model.evaluate(x_cost, y_cost, verbose=0)
        print('{} {}. c_train={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, c_train))
        y_pred = model.predict(x_cost, verbose=0)
        correct_predictions = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_cost, axis=1))
        off_by_one = np.mean(np.abs(np.argmax(y_pred, axis=1) - np.argmax(y_cost, axis=1)) > 1)
        avg_difference = np.mean(np.abs(np.argmax(y_pred, axis=1) - np.argmax(y_cost, axis=1)))
        print('Correct Predictions: ', correct_predictions)
        print('Off by One: ', off_by_one)
        print('Average Difference: ', avg_difference)
        sys.stdout.flush()
        model.save(model_path)  

    model.fit(x_batch, y_batch, batch_size, epochs, verbose=0)

model.save(model_path)
