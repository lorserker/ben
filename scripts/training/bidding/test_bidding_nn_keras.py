from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

np.set_printoptions(precision=2, suppress=True, linewidth=300,threshold=np.inf)

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

def print_input(x, y, z):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            print(hand_to_str(x[i,j,9:9+n_cards]),end=" ")
            for k in range(4):
                print(np.argmax(x[i,j,9 + n_cards + k*40:81+k*40]),end=" ")

            print("predict ",end="")                
            print(np.argmax(y[i,j]),end="")
            if (z[i,j,0] != 0):             
                print("*",end="")
            print()

# Load the saved model
model_path = '../../../../models/TF2Models/GIBOpening_2024-08-17-E100.keras'  # Replace with your actual model path
model_path = 'model/GIBOpening2D_2024-08-18-E35.keras'
model = load_model(model_path)

X_train = np.load('./2D/X.npy')
y_train = np.load('./2D/y.npy')
z_train = np.load('./2D/z.npy')

print(X_train.shape)
print(y_train.shape)
print(z_train.shape)

# Take the first 8 elements from each array
X_train_first_8 = X_train[0:1]
Y_train_first_8 = y_train[0:1]
Z_train_first_8 = z_train[0:1]

print("X_train_first_8")
print(X_train_first_8[:, :8, :])

#print("Y_train_first_8")
#print(Y_train_first_8[:, :8, :])

#print("Z_train_first_8")
#print(Z_train_first_8[:, :8, :])

print_input(X_train_first_8[:, :1, :], Y_train_first_8[:, :1, :], Z_train_first_8[:, :1, :] )

predictions = model.predict(X_train_first_8[:, :1, :], verbose=0)
#print("Input:", X_train_first_8[:, :2, :])
print(predictions)

print("-------------------------------------------------------")
