import sys
sys.path.append('../../../src')

import datetime
import time
import os.path
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
print("os.cpu_count()",os.cpu_count())

from batcher import Batcher

if len(sys.argv) < 2:
    print("Usage: python bidding_nn_alert.py inputdirectory system")
    sys.exit(1)

bin_dir = sys.argv[1]
print(sys.argv)

system = "bidding"
if len(sys.argv) > 2:
    system = sys.argv[2]

batch_size = 64
display_step = 10000
epochs = 50
learning_rate = 0.0005

X_train = np.load(os.path.join(bin_dir, 'x.npy'))
y_train = np.load(os.path.join(bin_dir, 'y.npy'))

# Split y_train into bidding and alert components
bids = y_train[:, :, :-1]  # All but the last element
alerts = y_train[:, :, -1:]  # Only the last element

n_examples = X_train.shape[0]
n_ftrs = X_train.shape[2]
n_bids = bids.shape[2]

# If NS/EW cc included update name of model
if n_ftrs == 201:
    model_path = f'model/{datetime.datetime.now().strftime("%Y-%m-%d")}-NS{int(X_train[0,0][0])}EW{int(X_train[0,0][1])}-bidding_V2'
else:
    if n_ftrs == 199:
        model_path = f'model/{datetime.datetime.now().strftime("%Y-%m-%d")}_{system}_V2'
    else:
        print("Older versions not supported")

print("Size input hand:         ", n_ftrs)
print("Examples for training:   ", n_examples)
print("Batch size:              ", batch_size)
n_iterations = round(((n_examples / batch_size) * epochs) / 1000) * 1000
print("Iterations               ", n_iterations)
print("Model path:              ", model_path)
print("Learning rate:           ", learning_rate)

lstm_size = 256
n_layers = 3

keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

cells = []
for _ in range(n_layers):
    cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(lstm_size),
        output_keep_prob=keep_prob
    )
    cells.append(cell)

x_in = tf.compat.v1.placeholder(tf.float32, [1, n_ftrs], name='x_in')
    
lstm_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)

seq_in = tf.compat.v1.placeholder(tf.float32, [None, None, n_ftrs], 'seq_in')
seq_out_bid = tf.compat.v1.placeholder(tf.float32, [None, None, n_bids], 'seq_out_bid')
seq_out_alert = tf.compat.v1.placeholder(tf.float32, [None, None, 1], 'seq_out_alert')

softmax_w_bid = tf.compat.v1.get_variable('softmax_w_bid', shape=[lstm_cell.output_size, n_bids], dtype=tf.float32, initializer=tf.compat.v1.initializers.glorot_uniform(seed=1337))
softmax_w_alert = tf.compat.v1.get_variable('softmax_w_alert', shape=[lstm_cell.output_size, 1], dtype=tf.float32, initializer=tf.compat.v1.initializers.glorot_uniform(seed=1337))

out_rnn, _ = tf.nn.dynamic_rnn(lstm_cell, seq_in, dtype=tf.float32)

out_bid_logit = tf.matmul(tf.reshape(out_rnn, [-1, lstm_size]), softmax_w_bid, name='out_bid_logit')
out_alert_logit = tf.matmul(tf.reshape(out_rnn, [-1, lstm_size]), softmax_w_alert, name='out_alert_logit')

out_bid_target = tf.reshape(seq_out_bid, [-1, n_bids], name='out_bid_target')
out_alert_target = tf.reshape(seq_out_alert, [-1, 1], name='out_alert_target')

output, _ = tf.nn.dynamic_rnn(lstm_cell, seq_in, dtype=tf.float32)

out_bid = tf.nn.softmax(out_bid_logit, name='out_bid')
out_alert = tf.nn.sigmoid(out_alert_logit, name='out_alert')

loss_bid = tf.compat.v1.losses.softmax_cross_entropy(out_bid_target, out_bid_logit)
loss_alert = tf.compat.v1.losses.sigmoid_cross_entropy(out_alert_target, out_alert_logit)

cost = loss_bid + loss_alert

train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, batch_size)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver(max_to_keep=1)

    x_cost, y_cost_bid, y_cost_alert = cost_batch.next_batch([X_train, bids, alerts])

    t_start = time.time()
    for i in range(n_iterations):
        x_batch, y_batch_bid, y_batch_alert = batch.next_batch([X_train, bids, alerts])
        if (i != 0) and i % display_step == 0:
            c_train = sess.run(cost, feed_dict={seq_in: x_cost, seq_out_bid: y_cost_bid, seq_out_alert: y_cost_alert, keep_prob: 1.0})
            print('{} {}. c_train={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, c_train))
            sys.stdout.flush()
            saver.save(sess, model_path, global_step=i)
            print(f"{display_step} interations: {(time.time() - t_start):0.4f}")        
            t_start = time.time()
        sess.run(train_step, feed_dict={seq_in: x_batch, seq_out_bid: y_batch_bid, seq_out_alert: y_batch_alert, keep_prob: 0.8})

    saver.save(sess, model_path, global_step=n_iterations)
    print(f"{display_step} interations: {(time.time() - t_start):0.4f}")        
