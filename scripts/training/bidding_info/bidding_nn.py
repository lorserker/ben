import sys
sys.path.append('D:/github/ben/src')

import datetime
import os.path
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from batcher import Batcher

if len(sys.argv) < 2:
    print("Usage: python bidding_nn inputdirectory")
    sys.exit(1)

bin_dir = sys.argv[1]
print(sys.argv)

# Test setting

batch_size = 64
display_step = 10000
epochs = 100
learning_rate = 0.001

X_train = np.load(os.path.join(bin_dir, 'x.npy'))
y_train = np.load(os.path.join(bin_dir, 'y.npy'))

n_examples = y_train.shape[0]
n_ftrs = X_train.shape[2]
n_bids = y_train.shape[2]

# If NS/EW cc included update name of model
if n_ftrs == 201:
    model_path = f'model/NS{int(X_train[0,0][0])}EW{int(X_train[0,0][1])}-bidding-V2'
else:
    if n_ftrs == 161:
        model_path = f'model/NS{int(X_train[0,0][0])}EW{int(X_train[0,0][1])}-bidding'
    else:
        model_path = 'model/bidding'


print("Size input hand:         ", n_ftrs)
print("Examples for training:   ", n_examples)
print("Batch size:              ", batch_size)
n_iterations = round(((n_examples / batch_size) * epochs) / 1000) * 1000
print("Iterations               ", n_iterations)
print("Model path:              ",model_path)

lstm_size = 128
n_layers = 3

keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

cells = []
for _ in range(n_layers):
    cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(lstm_size),
        output_keep_prob=keep_prob
    )
    cells.append(cell)
if n_ftrs < 200:
    state = []
    for i, cell_i in enumerate(cells):
        s_c = tf.compat.v1.placeholder(tf.float32, [1, lstm_size], name='state_c_{}'.format(i))
        s_h = tf.compat.v1.placeholder(tf.float32, [1, lstm_size], name='state_h_{}'.format(i))
        state.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c=s_c, h=s_h))
    state = tuple(state)

x_in = tf.compat.v1.placeholder(tf.float32, [1, n_ftrs], name='x_in')
    
lstm_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)

seq_in = tf.compat.v1.placeholder(tf.float32, [None, None, n_ftrs], 'seq_in')
seq_out = tf.compat.v1.placeholder(tf.float32, [None, None, n_bids], 'seq_out')

softmax_w = tf.compat.v1.get_variable('softmax_w', shape=[lstm_cell.output_size, n_bids], dtype=tf.float32, initializer=tf.compat.v1.initializers.glorot_uniform(seed=1337))

out_rnn, _ = tf.nn.dynamic_rnn(lstm_cell, seq_in, dtype=tf.float32)

out_bid_logit = tf.matmul(tf.reshape(out_rnn, [-1, lstm_size]), softmax_w, name='out_bid_logit')
out_bid_target = tf.reshape(seq_out, [-1, n_bids], name='out_bid_target')


if n_ftrs < 200:
    output, next_state = lstm_cell(x_in, state)
    for i, next_i in enumerate(next_state):
        tf.identity(next_i.c, name='next_c_{}'.format(i))
        tf.identity(next_i.h, name='next_h_{}'.format(i))
else:
    output, _ = tf.nn.dynamic_rnn(lstm_cell, seq_in, dtype=tf.float32)


out_bid = tf.nn.softmax(tf.matmul(output, softmax_w), name='out_bid')

cost = tf.compat.v1.losses.softmax_cross_entropy(out_bid_target, out_bid_logit)

train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, batch_size)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver(max_to_keep=1)

    for i in range(n_iterations):
        x_batch, y_batch = batch.next_batch([X_train, y_train])
        if (i != 0) and i % display_step == 0:
            x_cost, y_cost = cost_batch.next_batch([X_train, y_train])
            c_train = sess.run(cost, feed_dict={seq_in: x_cost, seq_out: y_cost, keep_prob: 1.0})
            print('{} {}. c_train={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),i, c_train))
            sys.stdout.flush()
            saver.save(sess, model_path, global_step=i)
        sess.run(train_step, feed_dict={seq_in: x_batch, seq_out: y_batch, keep_prob: 0.8})

    saver.save(sess, model_path, global_step=n_iterations)
