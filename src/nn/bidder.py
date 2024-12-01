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
        self.model  = self.init_model()
        
    def close(self):
        self.sess.close()

    def load_model(self):
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(self.model_path + '.meta')
            saver.restore(self.sess, self.model_path)

    def init_model(self):
        graph = self.sess.graph

        self.seq_in = graph.get_tensor_by_name('seq_in:0')
        self.seq_out = graph.get_tensor_by_name('seq_out:0')
        self.keep_prob = graph.get_tensor_by_name('keep_prob:0')

        self.out_bid_logit = graph.get_tensor_by_name('out_bid_logit:0')
        self.out_bid_target = graph.get_tensor_by_name('out_bid_target:0')

        self.state_c_0 = graph.get_tensor_by_name('state_c_0:0')
        self.state_h_0 = graph.get_tensor_by_name('state_h_0:0')

        self.state_c_1 = graph.get_tensor_by_name('state_c_1:0')
        self.state_h_1 = graph.get_tensor_by_name('state_h_1:0')

        self.state_c_2 = graph.get_tensor_by_name('state_c_2:0')
        self.state_h_2 = graph.get_tensor_by_name('state_h_2:0')

        self.next_c_0 = graph.get_tensor_by_name('next_c_0:0')
        self.next_h_0 = graph.get_tensor_by_name('next_h_0:0')

        self.next_c_1 = graph.get_tensor_by_name('next_c_1:0')
        self.next_h_1 = graph.get_tensor_by_name('next_h_1:0')

        self.next_c_2 = graph.get_tensor_by_name('next_c_2:0')
        self.next_h_2 = graph.get_tensor_by_name('next_h_2:0')

        self.x_in = graph.get_tensor_by_name('x_in:0')
        self.out_bid = graph.get_tensor_by_name('out_bid:0')
        
        # defining model
        self.p_keep = 1.0
        
    def pred_fun(self, x, state_in):
        bids, next_state = None, None
        with self.graph.as_default():
            feed_dict = {
                self.keep_prob: self.p_keep,
                self.x_in: x,
                self.state_c_0: state_in[0].c,
                self.state_h_0: state_in[0].h,
                self.state_c_1: state_in[1].c,
                self.state_h_1: state_in[1].h,
                self.state_c_2: state_in[2].c,
                self.state_h_2: state_in[2].h,
            }
            bids = self.sess.run(self.out_bid, feed_dict=feed_dict)
            next_state = (
                State(c=self.sess.run(self.next_c_0, feed_dict=feed_dict), h=self.sess.run(self.next_h_0, feed_dict=feed_dict)),
                State(c=self.sess.run(self.next_c_1, feed_dict=feed_dict), h=self.sess.run(self.next_h_1, feed_dict=feed_dict)),
                State(c=self.sess.run(self.next_c_2, feed_dict=feed_dict), h=self.sess.run(self.next_h_2, feed_dict=feed_dict)),
            )
        
        return bids, next_state
    
    # This should probably no longer be used as TF 2.x requires the entire sequence as input
    # Or more correct this is the only one to use
    def pred_fun_seq(self,x):
        result = None
        with self.graph.as_default():
            feed_dict = {
                self.keep_prob: self.p_keep,
                self.seq_in: x,
            }
            result = self.sess.run(self.output_softmax, feed_dict=feed_dict)
            
        return [result,0]
        
