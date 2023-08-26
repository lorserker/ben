import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class LeadSingleDummy:

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

        x_in = graph.get_tensor_by_name('X:0')
        tricks_softmax = graph.get_tensor_by_name('tricks_softmax:0')

        keep_prob = graph.get_tensor_by_name('keep_prob:0')

        def pred_fun(x):
            result = None
            with self.graph.as_default():
                feed_dict = {
                    keep_prob: 1.0,
                    x_in: x
                }
                result = self.sess.run(tricks_softmax, feed_dict=feed_dict)

            return result

        return pred_fun
