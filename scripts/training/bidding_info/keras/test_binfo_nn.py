from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
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
                print(f"{np.argmax(x[i,j,41+k*40:81+k*40]):02d}",end=" ")

            print("predict: ",end="")                
            print(y[i,j],end=" ")
            print(z[i,j])

# Load the saved model
model_path = 'model/NS-1EW-1-binfo_V2.keras'  # Replace with your actual model path
model = load_model(model_path)

X_train = np.load('../bidding/bin/X.npy')
y_train = np.load('../bidding/bin/HCP.npy')
z_train = np.load('../bidding/bin/SHAPE.npy')

print(X_train.shape)
print(y_train.shape)
print(z_train.shape)

# Take the first 8 elements from each array
X_train_first_8 = X_train[:1]
Y_train_first_8 = y_train[:1]
Z_train_first_8 = z_train[:1]

print("X_train_first_8")
print(X_train_first_8[:, :8, :])

print("Y_train_first_8")
print(Y_train_first_8[:, :8, :])

print("Z_train_first_8")
print(Z_train_first_8[:, :8, :])

print_input(X_train_first_8[:, :8, :], Y_train_first_8[:, :8, :], Z_train_first_8[:, :8, :] )

padded_squence = np.pad(X_train_first_8[:, :4, :],((0, 0), (0, 4), (0, 0)), constant_values=0) 
# Make predictions
predictions, a = model.predict(padded_squence)

print(predictions)
print(a)
