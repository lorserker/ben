import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from collections import namedtuple

State = namedtuple('State', ['c', 'h'])


class Bidder:
    
    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.load_model()
        self.output_softmax = tf.nn.softmax(self.graph.get_tensor_by_name('out_bid_logit:0'))
        self.graph.finalize()
        self.lstm_size = 128
        self.zero_state = (
            State(c=np.zeros((1, self.lstm_size)), h=np.zeros((1, self.lstm_size))),
            State(c=np.zeros((1, self.lstm_size)), h=np.zeros((1, self.lstm_size))),
            State(c=np.zeros((1, self.lstm_size)), h=np.zeros((1, self.lstm_size))),
        )
        self.nesw_initial = [self.zero_state, self.zero_state, self.zero_state, self.zero_state]
        self.model_seq, self.model  = self.init_model()
        
    def close(self):
        self.sess.close()

    def load_model(self):
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(self.model_path + '.meta')
            saver.restore(self.sess, self.model_path)

    def init_model(self):
        graph = self.sess.graph

        seq_in = graph.get_tensor_by_name('seq_in:0')
        seq_out = graph.get_tensor_by_name('seq_out:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')

        out_bid_logit = graph.get_tensor_by_name('out_bid_logit:0')
        out_bid_target = graph.get_tensor_by_name('out_bid_target:0')

        state_c_0 = graph.get_tensor_by_name('state_c_0:0')
        state_h_0 = graph.get_tensor_by_name('state_h_0:0')

        state_c_1 = graph.get_tensor_by_name('state_c_1:0')
        state_h_1 = graph.get_tensor_by_name('state_h_1:0')

        state_c_2 = graph.get_tensor_by_name('state_c_2:0')
        state_h_2 = graph.get_tensor_by_name('state_h_2:0')

        next_c_0 = graph.get_tensor_by_name('next_c_0:0')
        next_h_0 = graph.get_tensor_by_name('next_h_0:0')

        next_c_1 = graph.get_tensor_by_name('next_c_1:0')
        next_h_1 = graph.get_tensor_by_name('next_h_1:0')

        next_c_2 = graph.get_tensor_by_name('next_c_2:0')
        next_h_2 = graph.get_tensor_by_name('next_h_2:0')

        x_in = graph.get_tensor_by_name('x_in:0')
        out_bid = graph.get_tensor_by_name('out_bid:0')
        
        # defining model
        p_keep = 1.0
        
        def pred_fun(x, state_in):
            bids, next_state = None, None
            with self.graph.as_default():
                feed_dict = {
                    keep_prob: p_keep,
                    x_in: x,
                    state_c_0: state_in[0].c,
                    state_h_0: state_in[0].h,
                    state_c_1: state_in[1].c,
                    state_h_1: state_in[1].h,
                    state_c_2: state_in[2].c,
                    state_h_2: state_in[2].h,
                }
                bids = self.sess.run(out_bid, feed_dict=feed_dict)
                next_state = (
                    State(c=self.sess.run(next_c_0, feed_dict=feed_dict), h=self.sess.run(next_h_0, feed_dict=feed_dict)),
                    State(c=self.sess.run(next_c_1, feed_dict=feed_dict), h=self.sess.run(next_h_1, feed_dict=feed_dict)),
                    State(c=self.sess.run(next_c_2, feed_dict=feed_dict), h=self.sess.run(next_h_2, feed_dict=feed_dict)),
                )
            return bids, next_state
        
        def pred_fun_seq(x):
            result = None
            with self.graph.as_default():
                feed_dict = {
                    keep_prob: p_keep,
                    seq_in: x,
                }
                result = self.sess.run(self.output_softmax, feed_dict=feed_dict)
            return result
        
        return pred_fun_seq, pred_fun
