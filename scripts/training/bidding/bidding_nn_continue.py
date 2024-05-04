import sys
sys.path.append('../../../src')
import os

import datetime
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from batcher import Batcher

if len(sys.argv) < 2:
    print("Usage: python bidding_nn.py inputdirectory checkpoint_model")
    sys.exit(1)

bin_dir = sys.argv[1] # Location of data
checkpoint_model = sys.argv[2]  # pretrained model e.g './model/bidding-1000000'

model_path = checkpoint_model.split('-')[:-1][0]

batch_size = 64
start_iteration = int(checkpoint_model.split('-')[-1])
n_iterations = 1000000
display_step = 10000

bin_dir = sys.argv[1]

batch_size = 64
display_step = 10000
epochs = 50
learning_rate = 0.0005

X_train = np.load(os.path.join(bin_dir, 'x.npy'))
y_train = np.load(os.path.join(bin_dir, 'y.npy'))

n_examples = y_train.shape[0]
n_ftrs = X_train.shape[2]
n_bids = y_train.shape[2]

print("Size input hand:         ", n_ftrs)
print("Examples for training:   ", n_examples)
print("Batch size:              ", batch_size)
n_iterations = round(((n_examples / batch_size) * epochs) / 1000) * 1000
print("Iterations               ", n_iterations)
print("Start iterations         ", start_iteration)
print("Model path:              ", model_path)
print("Learning rate:           ", learning_rate)

lstm_size = 128
n_layers = 3

graph = tf.Graph()
sess = tf.Session(graph=graph)

with graph.as_default():
    saver = tf.train.import_meta_graph(f'{checkpoint_model}.meta')
    saver.restore(sess, checkpoint_model)

    seq_in = graph.get_tensor_by_name('seq_in:0')
    seq_out = graph.get_tensor_by_name('seq_out:0')

    out_bid_logit = graph.get_tensor_by_name('out_bid_logit:0')
    out_bid_target = graph.get_tensor_by_name('out_bid_target:0')

    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    cost = tf.losses.softmax_cross_entropy(out_bid_target, out_bid_logit)

    train_step = graph.get_operation_by_name('Adam')

    batch = Batcher(n_examples, batch_size)
    cost_batch = Batcher(n_examples, batch_size)

    saver = tf.train.Saver(max_to_keep=1)

    x_cost, y_cost = cost_batch.next_batch([X_train, y_train])

    for i in range(start_iteration, n_iterations):
        x_batch, y_batch = batch.next_batch([X_train, y_train])
        if (i != 0) and i % display_step == 0:
            c_train = sess.run(cost, feed_dict={seq_in: x_cost, seq_out: y_cost, keep_prob: 1.0})
            print('{} {}. c_train={}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, c_train))
            sys.stdout.flush()
            saver.save(sess, model_path, global_step=i)
        sess.run(train_step, feed_dict={seq_in: x_batch, seq_out: y_batch, keep_prob: 0.8})

    saver.save(sess, model_path, global_step=start_iteration + n_iterations)
