import numpy as np
import tensorflow.compat.v1 as tf


class BidInfo:
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.load_model()
        self.model = self.init_model()

    def close(self):
        self.sess.close()
        
    def load_model(self):
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(self.model_path + '.meta')
            saver.restore(self.sess, self.model_path)
        
    def init_model(self):
        graph = self.sess.graph

        self.seq_in = graph.get_tensor_by_name('seq_in:0')
        self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
        self.out_hcp_seq = graph.get_tensor_by_name('out_hcp_seq:0')
        self.out_shape_seq = graph.get_tensor_by_name('out_shape_seq:0')

        self.p_keep = 1.0
        
    def pred_fun(self, x):
        result = None
        with self.graph.as_default():
            result = self.sess.run(
                [self.out_hcp_seq, self.out_shape_seq], 
                feed_dict={self.seq_in: x, self.keep_prob: self.p_keep}
            )
        return result

