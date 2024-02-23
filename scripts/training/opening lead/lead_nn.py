import sys
import datetime
import numpy as np
import sys
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

model_path = './model/lead'

seed = 1337

batch_size = 64
display_step = 10000
epochs = 50

X_train = np.load('./lead_bin/X.npy')
B_train = np.load('./lead_bin/B.npy')
y_train = np.load('./lead_bin/y.npy')

n_examples = X_train.shape[0]
n_ftrs = X_train.shape[1]
n_cards = 32
n_bi = B_train.shape[1]


print("Size input hand:         ", n_ftrs)
print("Examples for training:   ", n_examples)
print("Batch size:              ", batch_size)
n_iterations = round(((n_examples / batch_size) * epochs) / 1000) * 1000
print("Iterations               ", n_iterations)
print("Model path:              ",model_path)

n_hidden_units = 512

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

X = tf.placeholder(tf.float32, [None, n_ftrs], 'X')
B = tf.placeholder(tf.float32, [None, n_bi], 'B')
y = tf.placeholder(tf.float32, [None, n_cards], 'y')

XB = tf.concat([X, B], axis=1, name='XB')

w1 = tf.get_variable('w1', shape=[n_ftrs + n_bi, n_hidden_units], dtype=tf.float32, initializer=tf.compat.v1.initializers.glorot_uniform(seed=seed))
z1 = tf.matmul(XB, w1)
a1 = tf.nn.dropout(tf.nn.relu(z1), keep_prob=keep_prob)

w2 = tf.get_variable('w2', shape=[n_hidden_units, n_hidden_units], dtype=tf.float32, initializer=tf.compat.v1.initializers.glorot_uniform(seed=seed))
z2 = tf.matmul(a1, w2)
a2 = tf.nn.dropout(tf.nn.relu(z2), keep_prob=keep_prob)

w3 = tf.get_variable('w3', shape=[n_hidden_units, n_hidden_units], dtype=tf.float32, initializer=tf.compat.v1.initializers.glorot_uniform(seed=seed))
z3 = tf.matmul(a2, w3)
a3 = tf.nn.dropout(tf.nn.relu(z3), keep_prob=keep_prob)

w_out = tf.get_variable('w_out', shape=[n_hidden_units, 32], dtype=tf.float32, initializer=tf.compat.v1.initializers.glorot_uniform(seed=seed))
lead_logit = tf.matmul(a3, w_out, name='lead_logit')
lead_softmax = tf.nn.softmax(lead_logit, name='lead_softmax')

cost = tf.losses.softmax_cross_entropy(y, lead_logit)

weights = [w1, w2]

learning_rate = tf.placeholder(tf.float32, name='learning_rate')

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

batch = Batcher(n_examples, batch_size)
cost_batch = Batcher(n_examples, batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=1)

    for i in range(n_iterations):
        x_batch, b_batch, y_batch = batch.next_batch([X_train, B_train, y_train])
        if i != 0 and (i % display_step) == 0:
            x_cost, b_cost, y_cost = cost_batch.next_batch([X_train, B_train, y_train])
            c_train = sess.run(cost, feed_dict={X: x_cost, B: b_cost, y: y_cost, keep_prob: 1.0})
            l_train = sess.run(lead_softmax, feed_dict={X: x_cost, B: b_cost, y: y_cost, keep_prob: 1.0})
            print('{} {}. c_train={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),i, c_train))
            print(np.mean(np.argmax(l_train, axis=1) == np.argmax(y_cost, axis=1)))
                
            sys.stdout.flush()

            saver.save(sess, model_path, global_step=i)
        
        sess.run(train_step, feed_dict={X: x_batch, B: b_batch, y: y_batch, keep_prob: 0.6, learning_rate: 0.001 / (2**(i/5e5))})

    saver.save(sess, model_path, global_step=n_iterations)
