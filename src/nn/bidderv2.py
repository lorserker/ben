import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Bidder:
    
    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.load_model()
        self.output_softmax = tf.nn.softmax(self.graph.get_tensor_by_name('out_bid_logit:0'))
        self.model_seq  = self.init_model()
        
    def close(self):
        self.sess.close()

    def load_model(self):
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(self.model_path + '.meta')
            saver.restore(self.sess, self.model_path)

    def init_model(self):
        graph = self.sess.graph

        seq_in = graph.get_tensor_by_name('seq_in:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')

        # defining model
        p_keep = 1.0
                
        # This should probably no longer be used as TF 2.x requires the entire sequence as input
        # Or more correct this is the only one to use
        def pred_fun_seq(x):
            result = None
            with self.graph.as_default():
                feed_dict = {
                    keep_prob: p_keep,
                    seq_in: x,
                }
                result = self.sess.run(self.output_softmax, feed_dict=feed_dict)
            return result
        
        return pred_fun_seq
