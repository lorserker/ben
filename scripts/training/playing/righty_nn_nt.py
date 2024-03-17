import sys
import numpy as np
import datetime
sys.path.append('../../../src')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import logging
import shutil

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from batcher import Batcher

model_path = './righty_model/righty_nt'
saved_model_dir = './righty_saved_model' 
batch_size = 64
display_step = 10000
epochs = 50

X_train = np.load('./righty_bin_nt/X.npy')
Y_train = np.load('./righty_bin_nt/Y.npy')

n_examples = Y_train.shape[0]
n_ftrs = X_train.shape[2]
n_cards = Y_train.shape[2]

print("Size input hand:         ", n_ftrs)
print("Examples for training:   ", n_examples)
print("Batch size:              ", batch_size)
n_iterations = round(((n_examples / batch_size) * epochs) / 1000) * 1000
print("Iterations               ", n_iterations)
print("Model path:              ",model_path)

lstm_size = 128
n_layers = 3

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

cells = []
for _ in range(n_layers):
    cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(lstm_size),
        output_keep_prob=keep_prob
    )
    cells.append(cell)

state = []
for i, cell_i in enumerate(cells):
    s_c = tf.placeholder(tf.float32, [1, lstm_size], name='state_c_{}'.format(i))
    s_h = tf.placeholder(tf.float32, [1, lstm_size], name='state_h_{}'.format(i))
    state.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c=s_c, h=s_h))
state = tuple(state)

x_in = tf.placeholder(tf.float32, [1, n_ftrs], name='x_in')
    
lstm_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)

seq_in = tf.placeholder(tf.float32, [None, None, n_ftrs], 'seq_in')
seq_out = tf.placeholder(tf.float32, [None, None, n_cards], 'seq_out')

softmax_w = tf.get_variable('softmax_w', shape=[lstm_cell.output_size, n_cards], dtype=tf.float32, initializer=tf.compat.v1.initializers.glorot_uniform(seed=1337))

out_rnn, _ = tf.nn.dynamic_rnn(lstm_cell, seq_in, dtype=tf.float32)

out_card_logit = tf.matmul(tf.reshape(out_rnn, [-1, lstm_size]), softmax_w, name='out_card_logit')
out_card_target = tf.reshape(seq_out, [-1, n_cards], name='out_card_target')

output, next_state = lstm_cell(x_in, state)

out_card = tf.nn.softmax(tf.matmul(output, softmax_w), name='out_card')

for i, next_i in enumerate(next_state):
    tf.identity(next_i.c, name='next_c_{}'.format(i))
    tf.identity(next_i.h, name='next_h_{}'.format(i))

cost = tf.losses.softmax_cross_entropy(out_card_target, out_card_logit)

train_step = tf.train.AdamOptimizer(0.0003).minimize(cost)

batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=1)

    for i in range(n_iterations):
        x_batch, y_batch = batch.next_batch([X_train, Y_train])
        if i != 0 and (i % display_step) == 0:
            x_cost, y_cost = cost_batch.next_batch([X_train, Y_train])
            c_train = sess.run(cost, feed_dict={seq_in: x_cost, seq_out: y_cost, keep_prob: 1.0})
            print('{} {}. c_train={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),i, c_train))
            sys.stdout.flush()
            saver.save(sess, model_path, global_step=i)
        sess.run(train_step, feed_dict={seq_in: x_batch, seq_out: y_batch, keep_prob: 0.8})

    saver.save(sess, model_path, global_step=n_iterations)

    # Save the model in the SavedModel format
    saved_model_dir = model_path + '_saved_model'

    try:
        shutil.rmtree(model_path + '_saved_model')
        print("Directory removed successfully.")
    except FileNotFoundError:
        pass  # Ignore the "Directory not found" exception
    except Exception as e:
        print(f"An error occurred: {e}")


    # Save the model in the SavedModel format
    # tf.saved_model.simple_save(sess, saved_model_dir, inputs={'seq_in': seq_in, 'keep_prob': keep_prob}, outputs={'out_card': out_card})