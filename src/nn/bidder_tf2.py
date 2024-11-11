import numpy as np
import tensorflow as tf
from collections import namedtuple
from tensorflow.keras.models import load_model
import time

class Bidder:
    
    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.model = self.load_model()
        self.model_seq = self.init_model_seq()


    def load_model(self):
        return load_model(self.model_path)
    
    def init_model_seq(self):
        # Wrapping the function with @tf.function to optimize for graph execution
        # @tf.function
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float16)])
        def pred_fun_seq(x):
            # Ensure that x is a tensor
            try:
                input_tensor = tf.cast(x, dtype=tf.float16)
            except:
                input_tensor = tf.cast(x, dtype=tf.float32)
            
            # Perform the model prediction (returns tensors)
            bids, alerts = self.model(input_tensor, training=False)  # Use model call instead of predict
            return bids, alerts

        return pred_fun_seq

