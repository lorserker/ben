import sys
sys.path.append('../../../src')

import datetime
import os.path
import numpy as np
import tensorflow as tf

    ##np.save(os.path.join(out_dir, 'X.npy'), X)

from batcher import Batcher

if len(sys.argv) < 2:
    print("Usage: python binfo_nn inputdirectory outputdirectory")
    sys.exit(1)

bin_dir = sys.argv[1]
modelpath = sys.argv[2]

model_path = os.path.join(modelpath, 'binfo')

batch_size = 64
n_iterations = 500000
display_step = 10000

X_train = np.load(os.path.join(bin_dir, 'X.npy'))
HCP_train = np.load(os.path.join(bin_dir, 'HCP.npy'))
SHAPE_train = np.load(os.path.join(bin_dir, 'SHAPE.npy'))

n_examples = X_train.shape[0]
n_ftrs = X_train.shape[2]
n_dim_hcp = HCP_train.shape[2]
n_dim_shape = SHAPE_train.shape[2]

lstm_size = 128
n_layers = 3

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

cells = []
for _ in range(n_layers):
    cell = tf.contrib.rnn.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(lstm_size),
        output_keep_prob=keep_prob
    )
    cells.append(cell)

state = []
for i, cell_i in enumerate(cells):
    s_c = tf.placeholder(tf.float32, [1, lstm_size], name='state_c_{}'.format(i))
    s_h = tf.placeholder(tf.float32, [1, lstm_size], name='state_h_{}'.format(i))
    state.append(tf.contrib.rnn.LSTMStateTuple(c=s_c, h=s_h))
state = tuple(state)

x_in = tf.placeholder(tf.float32, [1, n_ftrs], name='x_in')
    
lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)

seq_in = tf.placeholder(tf.float32, [None, None, n_ftrs], 'seq_in')
seq_out_hcp = tf.placeholder(tf.float32, [None, None, n_dim_hcp], 'seq_out_hcp')
seq_out_shape = tf.placeholder(tf.float32, [None, None, n_dim_shape], 'seq_out_shape')

w_hcp = tf.get_variable('w_hcp', shape=[lstm_cell.output_size, n_dim_hcp], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=1337))
w_shape = tf.get_variable('w_shape', shape=[lstm_cell.output_size, n_dim_shape], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=1337))

out_rnn, _ = tf.nn.dynamic_rnn(lstm_cell, seq_in, dtype=tf.float32)

out_hcp_seq = tf.matmul(tf.reshape(out_rnn, [-1, lstm_size]), w_hcp, name='out_hcp_seq')
out_hcp_target_seq = tf.reshape(seq_out_hcp, [-1, n_dim_hcp], name='out_hcp_target_seq')
out_shape_seq = tf.matmul(tf.reshape(out_rnn, [-1, lstm_size]), w_shape, name='out_shape_seq')
out_shape_target_seq = tf.reshape(seq_out_shape, [-1, n_dim_shape], name='out_shape_target_seq')

output, next_state = lstm_cell(x_in, state)

out_hcp = tf.matmul(output, w_hcp, name='out_hcp')
out_shape = tf.matmul(output, w_shape, name='out_shape')

for i, next_i in enumerate(next_state):
    tf.identity(next_i.c, name='next_c_{}'.format(i))
    tf.identity(next_i.h, name='next_h_{}'.format(i))

cost_hcp = tf.losses.absolute_difference(out_hcp_target_seq, out_hcp_seq)
cost_shape = tf.losses.absolute_difference(out_shape_target_seq, out_shape_seq)
cost = cost_hcp + cost_shape

train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, 10000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=20)

    for i in range(n_iterations):
        x_batch, hcp_batch, shape_batch = batch.next_batch([X_train, HCP_train, SHAPE_train])
        if i % display_step == 0:
            print(i)
            x_cost, hcp_cost, shape_cost = cost_batch.next_batch([X_train, HCP_train, SHAPE_train])
            c_train = \
                sess.run([cost, cost_hcp, cost_shape], 
                    feed_dict={seq_in: x_cost, seq_out_hcp: hcp_cost, seq_out_shape: shape_cost, keep_prob: 1.0})

            print('{} {}. c_train={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),i, c_train))

            p_hcp_seq, p_shape_seq = sess.run([out_hcp_seq, out_shape_seq], feed_dict={seq_in: x_cost, seq_out_hcp: hcp_cost, seq_out_shape: shape_cost, keep_prob: 1.0})

            print(
                np.mean(np.abs(hcp_cost[:,0,:] - p_hcp_seq.reshape(hcp_cost.shape)[:,0,:])),
                np.mean(np.abs(hcp_cost[:,1,:] - p_hcp_seq.reshape(hcp_cost.shape)[:,1,:])),
                np.mean(np.abs(hcp_cost[:,2,:] - p_hcp_seq.reshape(hcp_cost.shape)[:,2,:])),
                np.mean(np.abs(hcp_cost[:,3,:] - p_hcp_seq.reshape(hcp_cost.shape)[:,3,:])),
                np.mean(np.abs(hcp_cost[:,-1,:] - p_hcp_seq.reshape(hcp_cost.shape)[:,-1,:])))
            print(
                np.mean(np.abs(shape_cost[:,0,:] - p_shape_seq.reshape(shape_cost.shape)[:,0,:])),
                np.mean(np.abs(shape_cost[:,1,:] - p_shape_seq.reshape(shape_cost.shape)[:,1,:])),
                np.mean(np.abs(shape_cost[:,2,:] - p_shape_seq.reshape(shape_cost.shape)[:,2,:])),
                np.mean(np.abs(shape_cost[:,3,:] - p_shape_seq.reshape(shape_cost.shape)[:,3,:])),
                np.mean(np.abs(shape_cost[:,-1,:] - p_shape_seq.reshape(shape_cost.shape)[:,-1,:])))
            
            
            sys.stdout.flush()

            saver.save(sess, model_path, global_step=i)
        sess.run(train_step, feed_dict={seq_in: x_batch, seq_out_hcp: hcp_batch, seq_out_shape: shape_batch, keep_prob: 0.8})

    saver.save(sess, model_path, global_step=n_iterations)
    