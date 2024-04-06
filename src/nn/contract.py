import numpy as np
import tensorflow.compat.v1 as tf


class Contract:
    
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

        # Get placeholder variables
        X = graph.get_tensor_by_name('X:0')
        labels_bool1 = graph.get_tensor_by_name('labels_bool1:0')
        labels_tricks = graph.get_tensor_by_name('labels_tricks:0')
        labels_oh = graph.get_tensor_by_name('labels_contract:0')

        # Get output logits
        bool1_logits = graph.get_tensor_by_name('bool1_logits:0')
        tricks_logits = graph.get_tensor_by_name('tricks_logits:0')
        contract_logits = graph.get_tensor_by_name('oh_logits:0')

        def pred_fun(x):
            feed_dict = {X: x, labels_bool1: np.zeros((x.shape[0], 1)), 
                         labels_tricks: np.zeros((x.shape[0], 14)), labels_oh: np.zeros((x.shape[0], 40))}
            bool1, tricks, contract = self.sess.run([bool1_logits, tricks_logits, contract_logits], feed_dict=feed_dict)
            doubled = bool1[0] > 0
            tricks = int(np.argmax(tricks, axis=1)[0])
            contract_id = np.argmax(contract, axis=1)[0]
            return contract_id, doubled[0], tricks

        def get_top_k_tricks(x, k=3):
            feed_dict = {X: x, labels_bool1: np.zeros((x.shape[0], 1)),
                        labels_tricks: np.zeros((x.shape[0], 14)), labels_oh: np.zeros((x.shape[0], 40))}
            probs, indices = self.sess.run([tf.nn.softmax(tricks_logits), tf.argsort(tricks_logits, axis=1, direction="DESCENDING")], feed_dict=feed_dict)
            top_k_indices = indices[:, :k]
            top_k_probs = probs[np.arange(probs.shape[0])[:, np.newaxis], top_k_indices]
            return top_k_indices, top_k_probs
        
        def get_top_k_oh( x, k=3):
            feed_dict = {X: x, labels_bool1: np.zeros((x.shape[0], 1)),
                        labels_tricks: np.zeros((x.shape[0], 14)), labels_oh: np.zeros((x.shape[0], 40))}
            probs, indices = self.sess.run([tf.nn.softmax(contract_logits), tf.argsort(contract_logits, axis=1, direction='DESCENDING')], feed_dict=feed_dict)
            top_k_indices = indices[:, :k]
            top_k_probs = probs[np.arange(probs.shape[0])[:, np.newaxis], top_k_indices]
            return top_k_indices, top_k_probs         
        
        return pred_fun, get_top_k_tricks, get_top_k_oh
