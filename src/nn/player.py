import numpy as np
import tensorflow.compat.v1 as tf

from scipy.special import softmax


class BatchPlayer:

    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.load_model()
        self.graph.finalize()
        self.model = self.init_model()

    def close(self):
        self.sess.close()

    def load_model(self):
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(self.model_path + '.meta')
            saver.restore(self.sess, self.model_path)

    def init_model(self):
        graph = self.sess.graph

        seq_in = graph.get_tensor_by_name('seq_in:0')  #  we always give the whole sequence from the beginning. shape = (batch_size, n_tricks, n_features)
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        out_card_logit = graph.get_tensor_by_name('out_card_logit:0')  #  shows which card it would play at each trick. (but we only care about the card for last trick)

        p_keep = 1.0

        def pred_fun(x):
            result = None
            with self.graph.as_default():
                card_logit = self.sess.run(out_card_logit, feed_dict={seq_in: x, keep_prob: p_keep})
                result = self.reshape_card_logit(card_logit, x)
            return result

        return pred_fun

    def reshape_card_logit(self, card_logit, x):
        return softmax(card_logit.reshape((x.shape[0], x.shape[1], 32)), axis=2)

    def next_cards_softmax(self, x):
        result = self.model(x)[:,-1,:]
        return result


class BatchPlayerLefty(BatchPlayer):

    def reshape_card_logit(self, card_logit, x):
        softmax_card_logit =  softmax(card_logit.reshape((x.shape[0], x.shape[1] - 1, 32)), axis=2)
        return softmax_card_logit



