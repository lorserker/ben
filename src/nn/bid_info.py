import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

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

        seq_in = graph.get_tensor_by_name('seq_in:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        out_hcp_seq = graph.get_tensor_by_name('out_hcp_seq:0')
        out_shape_seq = graph.get_tensor_by_name('out_shape_seq:0')

        p_keep = 1.0
        
        def pred_fun(x):
            result = None
            with self.graph.as_default():
                result = self.sess.run(
                    [out_hcp_seq, out_shape_seq], 
                    feed_dict={seq_in: x, keep_prob: p_keep}
                )
            return result

        return pred_fun
