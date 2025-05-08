import os
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# Load the saved model
model_path = 'model/GIB-Info_2024-07-18-E50.keras'  # Replace with your actual model path
model = load_model(model_path)

bin_dir = '../../bidding/keras/bidding_keras'
# Load training data
X_train = np.load(os.path.join(bin_dir, 'x.npy'), mmap_mode='r')
HCP_train = np.load(os.path.join(bin_dir, 'HCP.npy'), mmap_mode='r')
SHAPE_train = np.load(os.path.join(bin_dir, 'SHAPE.npy'), mmap_mode='r')

print(X_train.shape)
print(HCP_train.shape)
print(SHAPE_train.shape)

# Take the first 8 elements from each array
X_train_first_8 = X_train[1:2]
Y_train_first_8 = HCP_train[1:2]
Z_train_first_8 = SHAPE_train[1:2]

print_input(X_train_first_8[:, :1, :], Y_train_first_8[:, :1, :], Z_train_first_8[:, :1, :] )
x = X_train_first_8[:, :1, :]
print(x.shape)
p_hcp, p_shp = model.predict(x)
print(p_hcp, p_shp)

print("-------------------------------------------------------")

def f_trans_hcp(x): return 4 * x + 10
def f_trans_shp(x): return 1.75 * x + 3.25

p_hcp = f_trans_hcp(p_hcp)
p_shp = f_trans_shp(p_shp)

print(p_hcp, p_shp)
