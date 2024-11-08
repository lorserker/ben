import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class Leader:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.init_model()

    def load_model(self):
        model = load_model(self.model_path)
        return model

    def init_model(self):
        model = self.load_model()

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, 42], dtype=tf.float16),  # shape of x
            tf.TensorSpec(shape=[None, 15], dtype=tf.float16)    # shape of b (adjust shape accordingly)
        ])
        def pred_fun(x, b):
            # Forward x and b as a dictionary to the model
            hand = tf.convert_to_tensor(x, dtype=tf.float16)
            shape = tf.convert_to_tensor(b, dtype=tf.float16)

            result = model({'X_input': hand, 'B_input': shape}, training=False)  # Using model() instead of model.predict()
            return result
        return pred_fun
