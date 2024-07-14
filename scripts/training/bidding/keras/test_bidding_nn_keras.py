from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

np.set_printoptions(precision=2, suppress=True, linewidth=300,threshold=np.inf)

def pad_and_reshape_sequences(sequences):
    print("sequences.shape", sequences.shape)
    # Flatten the sequences
    flat_sequences = sequences.reshape(sequences.shape[0], -1)

    # Determine the maximum length for padding
    maxlen = 1 * 201

    # Create an array of zeros for padding
    padded_sequences = np.zeros((flat_sequences.shape[0], maxlen)) 

    # Copy the original sequences into the padded array
    for i, seq in enumerate(flat_sequences):
        padded_sequences[i, :len(seq)] = seq

    # Reshape the padded sequences back to the 3D shape
    padded_sequences_3d = padded_sequences.reshape(sequences.shape[0], 1, 201)
    
    return padded_sequences_3d

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

def print_input(x, y, z):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            print(hand_to_str(x[i,j,9:41]),end=" ")
            for k in range(4):
                print(np.argmax(x[i,j,41+k*40:81+k*40]),end=" ")

            print("predict ",end="")                
            print(np.argmax(y[i,j]),end="")
            if (z[i,j,0] != 0):             
                print("*",end="")
            print()

# Load the saved model
model_path = '../../../../models/TF2Models/GIB_2024-07-11-E20.keras'  # Replace with your actual model path
model_path = 'model/GIB_2024-07-12-E02.keras'
model = load_model(model_path)

X_train = np.load('./bidding_keras/X.npy')
y_train = np.load('./bidding_keras/y.npy')
z_train = np.load('./bidding_keras/z.npy')

print(X_train.shape)
print(y_train.shape)
print(z_train.shape)

# Take the first 8 elements from each array
X_train_first_8 = X_train[1:2]
Y_train_first_8 = y_train[1:2]
Z_train_first_8 = z_train[1:2]

#print("X_train_first_8")
#print(X_train_first_8[:, :8, :])

#print("Y_train_first_8")
#print(Y_train_first_8[:, :8, :])

#print("Z_train_first_8")
#print(Z_train_first_8[:, :8, :])

print_input(X_train_first_8[:, :2, :], Y_train_first_8[:, :2, :], Z_train_first_8[:, :2, :] )

predictions = model.predict(X_train_first_8[:, :2, :], verbose=0)
print("Input:", X_train_first_8[:, :2, :])
print(predictions)

print("-------------------------------------------------------")
