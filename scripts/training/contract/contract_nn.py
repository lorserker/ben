import sys
sys.path.append('../../../src')

import datetime
import os.path
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from batcher import Batcher

if len(sys.argv) < 2:
    print("Usage: python contract_nn.py inputdirectory")
    sys.exit(1)

bin_dir = sys.argv[1]
print(sys.argv)

batch_size = 64
display_step = 10000
epochs = 50
learning_rate = 0.0001

X_train = np.load(os.path.join(bin_dir, 'x.npy'))
y_train = np.load(os.path.join(bin_dir, 'y.npy'))
z_train = np.load(os.path.join(bin_dir, 'z.npy'))
u_train = np.load(os.path.join(bin_dir, 'u.npy'))

n_examples = y_train.shape[0]
n_ftrs = X_train.shape[1]
n_contract = y_train.shape[1]

model_path = 'model/contract'

print("Size input hand:         ", n_ftrs)
print("Examples for training:   ", n_examples)
print("Batch size:              ", batch_size)
n_iterations = round(((n_examples / batch_size) * epochs) / 100) * 100
print("Iterations               ", n_iterations)
print("Model path:              ",model_path)

# Define hyperparameters
output_dim_bool1 = 1
output_dim_tricks = 14
output_dim_contract = 40

# Define graph
graph = tf.Graph()
with graph.as_default():
    # Input and Output Placeholder Variables
    X = tf.placeholder(tf.float32, shape=(None, n_ftrs), name='X')
    labels_bool1 = tf.placeholder(tf.float32, shape=(None, 1), name='labels_bool1')
    labels_tricks = tf.placeholder(tf.float32, shape=(None, 14), name='labels_tricks')
    labels_contract = tf.placeholder(tf.float32, shape=(None, output_dim_contract), name='labels_contract')

    # Initializers
    weight_initializer = tf.truncated_normal_initializer(stddev=0.01)
    bias_initializer = tf.constant_initializer(value=0.0)

    # Hidden Layers
    w1 = tf.Variable(weight_initializer([n_ftrs, 128]), name='w1')
    b1 = tf.Variable(bias_initializer([128]), name='b1')
    h1 = tf.nn.tanh(tf.matmul(X, w1) + b1, name='h1')

    w2 = tf.Variable(weight_initializer([128, 64]), name='w2')
    b2 = tf.Variable(bias_initializer([64]), name='b2')
    h2 = tf.nn.tanh(tf.matmul(h1, w2) + b2, name='h2')

    # Boolean Output Layers
    w_bool1 = tf.Variable(weight_initializer([64, output_dim_bool1]), name='w_bool1')
    b_bool1 = tf.Variable(bias_initializer([output_dim_bool1]), name='b_bool1')
    bool1_logits = tf.add(tf.matmul(h2, w_bool1), b_bool1, name='bool1_logits')

    # One-Hot Encoded Output Layer
    w_tricks = tf.Variable(weight_initializer([64, output_dim_tricks]), name='w_tricks')
    b_tricks = tf.Variable(bias_initializer([output_dim_tricks]), name='b_tricks')
    tricks_logits = tf.add(tf.matmul(h2, w_tricks), b_tricks, name='tricks_logits')

    # One-Hot Encoded Output Layer
    w_oh = tf.Variable(weight_initializer([64, output_dim_contract]), name='w_oh')
    b_oh = tf.Variable(bias_initializer([output_dim_contract]), name='b_oh')
    oh_logits = tf.add(tf.matmul(h2, w_oh), b_oh, name='oh_logits')

    # Calculate Loss Functions
    bool1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_bool1, logits=bool1_logits))
    tricks_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_tricks, logits=tricks_logits))
    oh_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_contract, logits=oh_logits))
    total_loss = bool1_loss + tricks_loss + oh_loss

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(total_loss)

# Run session
with tf.Session(graph=graph) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for i in range(len(X_train)):
            x = X_train[i]
            b1 = z_train[i]
            b2 = u_train[i]
            o = y_train[i]
            feed_dict = {X: x.reshape((1, n_ftrs)),
                         labels_bool1: b1.reshape((1, 1)),
                         labels_tricks: b2.reshape((1, 14)),
                         labels_contract: [o]}

            _, loss_value = sess.run([train_op, total_loss], feed_dict=feed_dict)

            if (i+1) % 10000 == 0:
                print('Epoch {}, Step {}: Loss = {:.3f}'.format(epoch+1, i+1, loss_value))
    saver.save(sess, model_path, global_step=n_iterations)
