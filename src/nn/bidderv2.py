import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Bidder:
    
    def __init__(self, name, model_path, alert_supported):
        self.alert_supported = alert_supported
        self.name = name
        self.model_path = model_path
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.load_model()
        self.model_seq  = self.init_model()
        
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

        # defining model
        self.p_keep = 1.0

        # Get output logits
        self.bids = tf.nn.softmax(graph.get_tensor_by_name('out_bid_logit:0'))
        if self.alert_supported:
            alert_logits  = graph.get_tensor_by_name('out_alert_logit:0')
            self.alert = tf.nn.sigmoid(alert_logits)  # Apply sigmoid to convert logits to probabilities

    def pred_fun_seq(self,x):
        result = None
        with self.graph.as_default():
            feed_dict = {
                self.keep_prob: self.p_keep,
                self.seq_in: x,
            }
        if self.alert_supported:
            result = self.sess.run([self.bids, self.alert], feed_dict=feed_dict)
            bids_result, alert_result = result
        else:
            result = self.sess.run([self.bids], feed_dict=feed_dict)
            bids_result, alert_result = result[0], 0

        return bids_result, alert_result

