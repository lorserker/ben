import numpy as np
import tensorflow as tf
from collections import namedtuple
from tensorflow.keras.models import load_model
import time

class Bidder:
    
    def __init__(self, name, model_path, alert_supported):
        self.alert_supported = alert_supported
        self.name = name
        self.model_path = model_path
        self.model = self.load_model()


    def load_model(self):
        return load_model(self.model_path)
    
    # Wrapping the function with @tf.function to optimize for graph execution
    # @tf.function
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float16)])
    def pred_fun_seq(self, x):
        # Ensure that x is a tensor
        try:
            input_tensor = tf.cast(x, dtype=tf.float16)
        except:
            input_tensor = tf.cast(x, dtype=tf.float32)
        if self.alert_supported:
            # Perform the model prediction (returns tensors)
            bids, alerts = self.model(input_tensor, training=False)  # Use model call instead of predict
        else:
            # Perform the model prediction (returns tensors)
            bids = self.model(input_tensor, training=False)  # Use model call instead of predict
            alerts = 0
        return bids, alerts


