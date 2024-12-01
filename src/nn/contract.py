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
        self.X = graph.get_tensor_by_name('X:0')
        self.labels_bool1 = graph.get_tensor_by_name('labels_bool1:0')
        self.labels_tricks = graph.get_tensor_by_name('labels_tricks:0')
        self.labels_oh = graph.get_tensor_by_name('labels_contract:0')

        # Get output logits
        self.bool1_logits = graph.get_tensor_by_name('bool1_logits:0')
        self.tricks_logits = graph.get_tensor_by_name('tricks_logits:0')
        self.contract_logits = graph.get_tensor_by_name('oh_logits:0')

    def pred_fun(self,x):
        feed_dict = {self.X: x, self.labels_bool1: np.zeros((x.shape[0], 1)), 
                        self.labels_tricks: np.zeros((x.shape[0], 14)), self.labels_oh: np.zeros((x.shape[0], 40))}
        bool1, tricks, contract = self.sess.run([self.bool1_logits, self.tricks_logits, self.contract_logits], feed_dict=feed_dict)

# Apply softmax to get probabilities for contract logits in the same session
        contract_probs = self.sess.run(tf.nn.softmax(self.contract_logits), feed_dict=feed_dict)

        doubled = bool1[0] > 0
        tricks = int(np.argmax(tricks, axis=1)[0])
        contract_id = np.argmax(contract, axis=1)[0]
        score = contract_probs[0][contract_id]
        return contract_id, doubled[0], tricks, score

    def get_top_k_tricks(self, x, k=3):
        feed_dict = {self.X: x, self.labels_bool1: np.zeros((x.shape[0], 1)),
                    self.labels_tricks: np.zeros((x.shape[0], 14)), self.labels_oh: np.zeros((x.shape[0], 40))}
        probs, indices = self.sess.run([tf.nn.softmax(self.tricks_logits), tf.argsort(self.tricks_logits, axis=1, direction="DESCENDING")], feed_dict=feed_dict)
        top_k_indices = indices[:, :k]
        top_k_probs = probs[np.arange(probs.shape[0])[:, np.newaxis], top_k_indices]
        return top_k_indices, top_k_probs
    
    def get_top_k_oh(self, x, k=3):
        feed_dict = {self.X: x, self.labels_bool1: np.zeros((x.shape[0], 1)),
                    self.labels_tricks: np.zeros((x.shape[0], 14)), self.labels_oh: np.zeros((x.shape[0], 40))}
        probs, indices = self.sess.run([tf.nn.softmax(self.contract_logits), tf.argsort(self.contract_logits, axis=1, direction='DESCENDING')], feed_dict=feed_dict)
        top_k_indices = indices[:, :k]
        top_k_probs = probs[np.arange(probs.shape[0])[:, np.newaxis], top_k_indices]
        return top_k_indices, top_k_probs         
        
