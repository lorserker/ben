import sys
sys.path.append('../../../src')

import datetime
import os.path
import numpy as np
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
print("os.cpu_count()",os.cpu_count())

from batcher import Batcher

# Limit the number of CPU threads used
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
print("os.cpu_count()", os.cpu_count())

# Set TensorFlow to only allocate as much GPU memory as needed
physical_devices = tf.config.list_physical_devices('GPU')
print("physical_devices", physical_devices)
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        tf.config.set_visible_devices(physical_devices, 'GPU')
        print("Using GPU: ", physical_devices)
    except RuntimeError as e:
        print(e)
else:
    # Ensure TensorFlow uses only CPU
    tf.config.set_visible_devices([], 'GPU')
    print("Using CPU only")


if len(sys.argv) < 2:
    print("Usage: python bidding_nn.py inputdirectory")
    sys.exit(1)

bin_dir = sys.argv[1]
print(sys.argv)

system = "bidding"
if len(sys.argv) > 2:
    system = sys.argv[2]

# Load training data
X_train = np.load(os.path.join(bin_dir, 'x.npy'), mmap_mode='r')
y_train = np.load(os.path.join(bin_dir, 'y.npy'), mmap_mode='r')
z_train = np.load(os.path.join(bin_dir, 'z.npy'), mmap_mode='r')

n_examples = X_train.shape[0]
n_sequence = X_train.shape[1]
n_ftrs = X_train.shape[2]
n_bids = y_train.shape[2]
n_alerts = z_train.shape[2]

batch_size = 64
display_step = 10000
epochs = 10
learning_rate = 0.0005
keep = 0.8

# If NS/EW cc included update name of model
if n_ftrs == 201:
    model_name = f'model/NS{int(X_train[0,0][0])}EW{int(X_train[0,0][1])}-bidding_V2'
else:
    if n_ftrs == 199:
        model_name = 'model/bidding_V2'
    else:
        if n_ftrs == 161:
            model_name = f'model/NS{int(X_train[0,0][0])}EW{int(X_train[0,0][1])}-bidding_V1'
        else:
            model_name = 'model/bidding_V0'

lstm_size = 128
n_layers = 3
n_cards = 32
patience = "N/A"
steps_per_epoch = n_examples // batch_size
buffer_size = "N/A"

n_iterations = round(((n_examples / batch_size) * epochs) / 1000) * 1000
print("Iterations               ", n_iterations)

print("-------------------------")
print("Examples for training:   ", n_examples)
print("Model path:              ", model_name )
print("-------------------------")
print("Size input hand:         ", n_ftrs)
print("Number of Cards:         ", n_cards)
print("Number of Sequences:     ", n_sequence)
print("Size output bid:         ", n_bids)
print("Size output alert:       ", "N/A")
print("-------------------------")
print("dtype X_train:           ", X_train.dtype)
print("dtype y_train:           ", y_train.dtype)
print("dtype z_train:           ", "N/A")
print("-------------------------")
print("Batch size:              ", batch_size)
print("buffer_size:             ", buffer_size)
print("steps_per_epoch          ", steps_per_epoch)
print("patience                 ", patience)
print("-------------------------")
print("Learning rate:           ", learning_rate)
print("Keep:                    ", keep)
print("-------------------------")
print("lstm_size:               ", lstm_size)
print("n_layers:                ", n_layers)

seed_value = 42
np.random.seed(seed_value)
tf.random.set_random_seed(seed_value)

keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

cells = []
for _ in range(n_layers):
    cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(lstm_size),
        output_keep_prob=keep_prob
    )
    cells.append(cell)
if n_ftrs < 180:
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

if n_ftrs < 180:
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

    x_cost, y_cost = cost_batch.next_batch([X_train, y_train])

    t_start = time.time()
    for i in range(n_iterations):
        x_batch, y_batch = batch.next_batch([X_train, y_train])
        if (i != 0) and i % display_step == 0:
            c_train = sess.run(cost, feed_dict={seq_in: x_cost, seq_out: y_cost, keep_prob: 1.0})
            print('{} {}. c_train={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),i, c_train))
            sys.stdout.flush()
            saver.save(sess, model_name, global_step=i)
            print(f"{display_step} interations: {(time.time() - t_start):0.4f}")        
            t_start = time.time()
        sess.run(train_step, feed_dict={seq_in: x_batch, seq_out: y_batch, keep_prob: keep})

    saver.save(sess, model_name, global_step=n_iterations)
    print(f"{display_step} interations: {(time.time() - t_start):0.2f} seconds")        
    print(model_name)