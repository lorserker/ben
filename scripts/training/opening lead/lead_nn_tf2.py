import sys
sys.path.append('../../../src')
import datetime
import numpy as np
import os
import logging
from tensorflow import keras
from batcher import Batcher

model_path = './model/lead.h5' 

X_train = np.load('./lead_bin/X.npy')
B_train = np.load('./lead_bin/B.npy')
y_train = np.load('./lead_bin/y.npy')

batch_size = 1
display_step = 10000
epochs = 1

n_examples = X_train.shape[0]
n_ftrs = X_train.shape[1]
n_cards = 32
n_bi = B_train.shape[1]


print("Size input hand:         ", n_ftrs)
print("Size input distribution: ", n_bi)
print("Size of output;          ", n_cards)
print("Examples for training:   ", n_examples)
print("Batch size:              ", batch_size)
n_iterations = round(((n_examples / batch_size) * epochs) / 1000) * 1000
print("Iterations               ", n_iterations)
print("Model path:              ",model_path)

n_hidden_units = 512

# Define the input layers with specific shapes
x_ftrs = keras.layers.Input(shape=(n_ftrs,), name='X')
b_ftrs = keras.layers.Input(shape=(n_bi,), name='B')

# Concatenate x_ftrs and b_ftrs
XB = keras.layers.concatenate([x_ftrs, b_ftrs], name='XB')

# Define the rest of the model architecture based on your requirements
model = keras.Sequential()

model.add(keras.layers.Dense(n_hidden_units, activation='relu', kernel_initializer='glorot_uniform', name='w1'))
model.add(keras.layers.Dropout(0.6, name='a1'))
model.add(keras.layers.Dense(n_hidden_units, activation='relu', kernel_initializer='glorot_uniform', name='w2'))
model.add(keras.layers.Dropout(0.6, name='a2'))
model.add(keras.layers.Dense(n_hidden_units, activation='relu', kernel_initializer='glorot_uniform', name='w3'))
model.add(keras.layers.Dropout(0.6, name='a3'))
model.add(keras.layers.Dense(n_cards, kernel_initializer='glorot_uniform', name='w_out'))

# Define the output of the model
lead_softmax = model(XB)

# Compile the model
compiled_model = keras.Model(inputs=[x_ftrs, b_ftrs], outputs=lead_softmax)
compiled_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy')

compiled_model.summary()

batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, batch_size)
x_cost, b_cost, y_cost = cost_batch.next_batch([X_train, B_train, y_train])

for i in range(n_iterations):
    x_batch, b_batch, y_batch = batch.next_batch([X_train, B_train, y_train])
    if (i != 0) and i % display_step == 0:
        c_train = compiled_model.evaluate([x_cost, b_cost], y_cost, batch_size, verbose=2)
        l_train = compiled_model.predict([x_cost, b_cost], batch_size, verbose=0)
        cost = np.mean(np.argmax(l_train, axis=1) == np.argmax(y_cost, axis=1))
        print('{} {}. c_train={} cost={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),i, c_train, cost))

        sys.stdout.flush()
        compiled_model.save(model_path)  # Save the Keras model in HDF5 format

    compiled_model.fit([x_batch, b_batch], y_batch, batch_size, epochs, verbose=0)

compiled_model.save(model_path) 