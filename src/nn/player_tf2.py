import sys
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
    def pred_fun(self, x):
        card_logit = self.model(x, training=False)
        return card_logit

    def next_cards_softmax(self, x):
        result = self.model(x)[:,-1,:]
        return result
