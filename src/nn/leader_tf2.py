import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class Leader:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        return load_model(self.model_path, compile=False)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 42], dtype=tf.float16),  # shape of x
        tf.TensorSpec(shape=[None, 15], dtype=tf.float16)    # shape of b (adjust shape accordingly)
    ])
    def pred_fun(self, x, b):
        # Forward x and b as a dictionary to the model
        result = self.model({'X_input': x, 'B_input': b}, training=False)  # Using model() instead of model.predict()
        return result
