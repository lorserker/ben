import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class BatchPlayer:

    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        return load_model(self.model_path, compile=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 298], dtype=tf.float16)])
    def pred_fun_tf(self, x):
        return self.model(x, training=False)

    def pred_fun(self, x):
        card_logit = self.pred_fun_tf(x)
        return card_logit.numpy()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 298], dtype=tf.float16)])
    def next_cards_softmax_tf(self, x):
        result = self.model(x)
        return result

    def next_cards_softmax(self, x):
        result = self.next_cards_softmax_tf(x)[:,-1,:]
        return result.numpy()
