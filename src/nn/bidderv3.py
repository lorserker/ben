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

        # Get output logits
        bids = tf.nn.softmax(graph.get_tensor_by_name('out_bid_logit:0'))
        alert_logits  = graph.get_tensor_by_name('out_alert_logit:0')
        alert = tf.nn.sigmoid(alert_logits)  # Apply sigmoid to convert logits to probabilities
        def pred_fun_seq(x):
            result = None
            with self.graph.as_default():
                feed_dict = {
                    keep_prob: p_keep,
                    seq_in: x,
                }
                result = self.sess.run([bids, alert], feed_dict=feed_dict)
                # Print statements for debugging
                #print("Bids:", result[0])
                #print("Alerts (probabilities):", result[1])
                
                # Optionally interpret the alert probabilities
                #interpreted_alerts = [1 if a >= 0.5 else 0 for a in result[1]]
                #print("Interpreted Alerts:", interpreted_alerts)
                
                return result

        return pred_fun_seq
