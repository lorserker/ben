import sys
sys.path.append('../../../src')

import datetime
import os.path
import numpy as np

import logging
# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from batcher import Batcher

if len(sys.argv) < 2:
    print("Usage: python binfo_and_info_nn.py inputdirectory ")
    sys.exit(1)

bin_dir = sys.argv[1]

batch_size = 64
display_step = 10000
epochs = 50
learning_rate = 0.001

X_train = np.load(os.path.join(bin_dir, 'x.npy'))
y_train = np.load(os.path.join(bin_dir, 'y.npy'))
HCP_train = np.load(os.path.join(bin_dir, 'HCP.npy'))
SHAPE_train = np.load(os.path.join(bin_dir, 'SHAPE.npy'))

n_examples = y_train.shape[0]
n_ftrs = X_train.shape[2]
n_bids = y_train.shape[2]
n_dim_hcp = HCP_train.shape[2]
n_dim_shape = SHAPE_train.shape[2]

# If NS/EW cc included update name of model
if n_ftrs == 161:
    model_path = f'model/NS{int(X_train[0,0][0])}EW{int(X_train[0,0][1])}-bidding_same'
    model_path_info = f'model/NS{int(X_train[0,0][0])}EW{int(X_train[0,0][1])}-binfo_same'
else:
    model_path = 'model/bidding'
    model_path_info = 'model/binfo'

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

state = []
for i, cell_i in enumerate(cells):
    s_c = tf.compat.v1.placeholder(tf.float32, [1, lstm_size], name='state_c_{}'.format(i))
    s_h = tf.compat.v1.placeholder(tf.float32, [1, lstm_size], name='state_h_{}'.format(i))
    state.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c=s_c, h=s_h))
state = tuple(state)

x_in = tf.placeholder(tf.float32, [1, n_ftrs], name='x_in')
    
lstm_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)

seq_in = tf.placeholder(tf.float32, [None, None, n_ftrs], 'seq_in')
output, next_state = lstm_cell(x_in, state)

## Define info model
seq_out_hcp = tf.placeholder(tf.float32, [None, None, n_dim_hcp], 'seq_out_hcp')
seq_out_shape = tf.placeholder(tf.float32, [None, None, n_dim_shape], 'seq_out_shape')

w_hcp = tf.get_variable('w_hcp', shape=[lstm_cell.output_size, n_dim_hcp], dtype=tf.float32, initializer=tf.compat.v1.initializers.glorot_uniform(seed=1337))
w_shape = tf.get_variable('w_shape', shape=[lstm_cell.output_size, n_dim_shape], dtype=tf.float32, initializer=tf.compat.v1.initializers.glorot_uniform(seed=1337))

out_rnn, _ = tf.nn.dynamic_rnn(lstm_cell, seq_in, dtype=tf.float32)

out_hcp_seq = tf.matmul(tf.reshape(out_rnn, [-1, lstm_size]), w_hcp, name='out_hcp_seq')
out_hcp_target_seq = tf.reshape(seq_out_hcp, [-1, n_dim_hcp], name='out_hcp_target_seq')
out_shape_seq = tf.matmul(tf.reshape(out_rnn, [-1, lstm_size]), w_shape, name='out_shape_seq')
out_shape_target_seq = tf.reshape(seq_out_shape, [-1, n_dim_shape], name='out_shape_target_seq')

out_hcp = tf.matmul(output, w_hcp, name='out_hcp')
out_shape = tf.matmul(output, w_shape, name='out_shape')

for i, next_i in enumerate(next_state):
    tf.identity(next_i.c, name='next_c_{}'.format(i))
    tf.identity(next_i.h, name='next_h_{}'.format(i))

cost_hcp = tf.losses.absolute_difference(out_hcp_target_seq, out_hcp_seq)
cost_shape = tf.losses.absolute_difference(out_shape_target_seq, out_shape_seq)
cost_info = cost_hcp + cost_shape

train_step_info = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_info)

## Define bidding model

seq_out = tf.compat.v1.placeholder(tf.float32, [None, None, n_bids], 'seq_out')

softmax_w = tf.compat.v1.get_variable('softmax_w', shape=[lstm_cell.output_size, n_bids], dtype=tf.float32, initializer=tf.compat.v1.initializers.glorot_uniform(seed=1337))

out_rnn, _ = tf.nn.dynamic_rnn(lstm_cell, seq_in, dtype=tf.float32)

out_bid_logit = tf.matmul(tf.reshape(out_rnn, [-1, lstm_size]), softmax_w, name='out_bid_logit')
out_bid_target = tf.reshape(seq_out, [-1, n_bids], name='out_bid_target')

out_bid = tf.nn.softmax(tf.matmul(output, softmax_w), name='out_bid')

cost = tf.compat.v1.losses.softmax_cross_entropy(out_bid_target, out_bid_logit)

train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, batch_size)

batch_info = Batcher(n_examples, batch_size)
cost_info_batch = Batcher(n_examples, batch_size)

with tf.compat.v1.Session() as sess, tf.compat.v1.Session() as sess_info:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess_info.run(tf.global_variables_initializer())

    saver = tf.compat.v1.train.Saver(max_to_keep=1)
    saver_info = tf.compat.v1.train.Saver(max_to_keep=1)

    for i in range(n_iterations):
        x_batch1, y_batch1 = batch.next_batch([X_train, y_train])
        if (i != 0) and i % display_step == 0:
            x_cost1, y_cost = cost_batch.next_batch([X_train, y_train])
            c_train1 = sess.run(cost, feed_dict={seq_in: x_cost1, seq_out: y_cost, keep_prob: 1.0})
            print('{} {}. c_train={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),i, c_train1))
            sys.stdout.flush()
            saver.save(sess, model_path, global_step=i)
        sess.run(train_step, feed_dict={seq_in: x_batch1, seq_out: y_batch1, keep_prob: 0.8})

        x_batch2, hcp_batch, shape_batch = batch_info.next_batch([X_train, HCP_train, SHAPE_train])
        if i > 0 and (i % display_step == 0):
            x_cost2, hcp_cost, shape_cost = cost_info_batch.next_batch([X_train, HCP_train, SHAPE_train])
            c_train2 = sess_info.run([cost_info, cost_hcp, cost_shape], feed_dict={seq_in: x_cost2, seq_out_hcp: hcp_cost, seq_out_shape: shape_cost, keep_prob: 1.0})

            print('{} {}. c_train={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),i ,c_train2))

            p_hcp_seq, p_shape_seq = sess_info.run([out_hcp_seq, out_shape_seq], feed_dict={seq_in: x_cost2, seq_out_hcp: hcp_cost, seq_out_shape: shape_cost, keep_prob: 1.0})
           
            sys.stdout.flush()

            saver_info.save(sess_info, model_path_info, global_step=i)
        sess_info.run(train_step_info, feed_dict={seq_in: x_batch2, seq_out_hcp: hcp_batch, seq_out_shape: shape_batch, keep_prob: 0.8})

    saver.save(sess, model_path, global_step=n_iterations)
    saver_info.save(sess_info, model_path_info, global_step=n_iterations)
