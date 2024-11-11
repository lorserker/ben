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
        model = load_model(self.model_path)
        return model

    def init_model(self):
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 298], dtype=tf.float16)])
        def pred_fun(x):
            try:
                input_tensor = tf.cast(x, dtype=tf.float16)
            except:
                input_tensor = tf.cast(x, dtype=tf.float32)
            #card_logit = model.predict(input_tensor,verbose=0)
            card_logit = self.model(input_tensor, training=False)
            return card_logit

        return pred_fun

    def next_cards_softmax(self, x):
        result = self.model(x)[:,-1,:]
        return result
