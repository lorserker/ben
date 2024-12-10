import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
# Set logging level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Limit the number of CPU threads used
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
import sys
sys.path.append('../../../../src')
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bidding import bidding

np.set_printoptions(precision=2, suppress=True, linewidth=300,threshold=np.inf)

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

            print("Expected ",end="")                
            c_hcp = (lambda x: 4 * x + 10)(y[i,j].copy())
            c_shp = (lambda x: 1.75 * x + 3.25)(z[i,j].copy())
            print(c_hcp,end="")
            print(c_shp,end="")
            print()

n_cards = 32
pips = 7
symbols = 'AKQJT98x'
n_cards = 24
pips = 9
symbols = 'AKQJTx'

def hand_to_str(hand):
    print(hand)
    x = hand.reshape((4, n_cards // 4))
    suits = []
    for i in range(4):
        s = ''
        for j in range( n_cards // 4):
            if x[i,j] > 0:
                s += symbols[j] * int(x[i,j])
        suits.append(s)
    return '.'.join(suits)


# Load the saved model
model_path = 'model/Tricks_2024-12-08-E50.keras'  # Replace with your actual model path
model = load_model(model_path)

bin_dir = 'keras_contract'
# Load training data
X_train = np.load(os.path.join(bin_dir, 'x.npy'))
y_train = np.load(os.path.join(bin_dir, 'y.npy'))
z_train = np.load(os.path.join(bin_dir, 'z.npy'))
u_train = np.load(os.path.join(bin_dir, 'u.npy'))


# Take the first 8 elements from each array
x = X_train[0:1]
y = y_train[0:1]
z = z_train[0:1]
u = u_train[0:1]
#x[0] = [1, 1, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 1, 0, 1, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2]
print("vul:", x[0,:2])
print(hand_to_str(x[0,2:26]))
print(hand_to_str(x[0,26:]))

contract = np.concatenate((x, y, z), axis=1)

tricks = model.predict(contract, verbose=1)


print("Trick probabilities:", tricks)
print(u)
print("Sum of contract probabilities:", np.sum(tricks, axis=1))


