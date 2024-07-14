import tensorflow as tf
import numpy as np
import os

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

# Load the trained model
model_path =  'model/Lead-NT_2024-07-13-E01.keras'  # Replace with your model path
model = tf.keras.models.load_model(model_path)

bin_dir = 'lead_keras_nt'
X_test = np.load(os.path.join(bin_dir, 'x.npy'), mmap_mode='r')
B_test = np.load(os.path.join(bin_dir, 'B.npy'), mmap_mode='r')
y_test = np.load(os.path.join(bin_dir, 'y.npy'), mmap_mode='r')

# Get the first element from X and B
X_sample = X_test[0].reshape(1, -1)  # Reshape to keep the batch dimension
B_sample = B_test[0].reshape(1, -1)  # Reshape to keep the batch dimension

# Get the actual value from y
y_actual = y_test[0].reshape(1, -1)  # Reshape to keep the batch dimension

print(X_sample)
print(hand_to_str(X_sample[:,10:42]))
print(B_sample)
# Predict using the model
y_pred = model.predict([X_sample, B_sample])

# Compare the predicted value with the actual value
predicted_label = np.argmax(y_pred, axis=1)
actual_label = np.argmax(y_actual, axis=1)

print(f"Predicted label: {predicted_label[0]}")
print(f"Actual label: {actual_label[0]}")

# Print the softmax probabilities for the first element
print(f"Softmax probabilities: {y_pred[0]}")

# Print the actual probabilities from y
print(f"Actual probabilities: {y_actual[0]}")
