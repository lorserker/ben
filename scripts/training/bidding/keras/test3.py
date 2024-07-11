import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

def hand_to_str(hand):
    x = hand.reshape((4, 8))
    symbols = 'AKQJT98x'
    suits = []
    for i in range(4):
        s = ''
        for j in range(8):
            if x[i,j] > 0:
                s += symbols[j] * int(x[i,j])
        suits.append(s)
    return '.'.join(suits)

def print_input(x, y):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            print(hand_to_str(x[i,j,7:39]),end=" ")
            for k in range(4):
                print(np.argmax(x[i,j,39+k*40:79+k*40]),end=" ")

            print("predict ",end="")                
            print(np.argmax(y[i,j]),end="")
            print()

np.set_printoptions(precision=2, suppress=True, linewidth=300,threshold=np.inf)
# Generate synthetic time sequence data
np.random.seed(42)

X = np.load('./bin2/X.npy')
y = np.load('./bin2/y.npy')
print(X.shape)
print(y.shape)
# Assuming X and y are your datasets
# X.shape = (4000, 8, 199)
# y.shape = (4000, 8, 40)

# Define the model
input_shape = (None, 199)
output_dim = 40
lstm_units = 128

model = Sequential([
    Input(shape=input_shape),
    LSTM(lstm_units, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(lstm_units, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(lstm_units, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(lstm_units, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(lstm_units, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),    TimeDistributed(Dense(output_dim, activation='softmax'))
])


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

print_input(X[1:2, :8, :], y[1:2, :8, :] )
# Making predictions
print(y[1:2, :, :])

# Predict with a single timestep
single_sequence = X[1:2, :1, :]  # Example: First sequence with only the first timestep
single_prediction = model.predict(single_sequence)
print("Single prediction shape:", single_prediction.shape)  # Should be (1, 1, 40)
print(single_prediction)

# Predict with multiple timesteps from the beginning
multiple_sequences = X[1:2, :5, :]  # Example: First sequence with the first 5 timesteps
multiple_predictions = model.predict(multiple_sequences)
print("Multiple predictions shape:", multiple_predictions.shape)  # Should be (1, 5, 40)
print(multiple_predictions)

# Predict with multiple timesteps from the beginning
multiple_sequences = X[1:2, :8, :]  # Example: First sequence with the first 5 timesteps
multiple_predictions = model.predict(multiple_sequences)
print("Multiple predictions shape:", multiple_predictions.shape)  # Should be (1, 5, 40)
print(multiple_predictions)

