from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.set_printoptions(precision=2, suppress=True, linewidth=300,threshold=np.inf)

def pad_and_reshape_sequences(sequences):
    print("sequences.shape", sequences.shape)
    # Flatten the sequences
    flat_sequences = sequences.reshape(sequences.shape[0], -1)

    # Determine the maximum length for padding
    maxlen = 1 * 199

    # Create an array of zeros for padding
    padded_sequences = np.zeros((flat_sequences.shape[0], maxlen)) 

    # Copy the original sequences into the padded array
    for i, seq in enumerate(flat_sequences):
        padded_sequences[i, :len(seq)] = seq

    # Reshape the padded sequences back to the 3D shape
    padded_sequences_3d = padded_sequences.reshape(sequences.shape[0], 1, 199)
    
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

def print_input(x, y):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            print(hand_to_str(x[i,j,7:39]),end=" ")
            for k in range(4):
                print(np.argmax(x[i,j,39+k*40:79+k*40]),end=" ")

            print("predict ",end="")                
            print(np.argmax(y[i,j]),end="")
            print()

# Load the saved model
model_path = '../../../models/GIB/2024-07-06_bidding_V2-3114000'  # Replace with your actual model path
graph = tf.Graph()
sess = tf.compat.v1.Session(graph=graph)
with graph.as_default():
    saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)

#model = load_model(model_path)

X_train = np.load('./bidding_bin/X.npy')
y_train = np.load('./bidding_bin/y.npy')

print(X_train.shape)
print(y_train.shape)

# Take the first 8 elements from each array
X_train_first_8 = X_train[1:2]
Y_train_first_8 = y_train[1:2]

print("X_train_first_8")
print(X_train_first_8[:, :8, :])

print("Y_train_first_8")
print(Y_train_first_8[:, :8, :])

print_input(X_train_first_8[:, :8, :], Y_train_first_8[:, :8, :] )

print(X_train_first_8.shape)

output_softmax = tf.nn.softmax(graph.get_tensor_by_name('out_bid_logit:0'))

seq_in = graph.get_tensor_by_name('seq_in:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
# defining model
p_keep = 1.0

with graph.as_default():
    feed_dict = {
        keep_prob: p_keep,
        seq_in: X_train_first_8[:, :8, :],
    }
    predictions = sess.run(output_softmax, feed_dict=feed_dict)

print(predictions)

print("-------------------------------------------------------")