import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class Leader:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        model = load_model(self.model_path)
        return model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 42], dtype=tf.float16),  # shape of x
        tf.TensorSpec(shape=[None, 15], dtype=tf.float16)    # shape of b (adjust shape accordingly)
    ])
    def pred_fun(self, x, b):
        # Forward x and b as a dictionary to the model
        hand = tf.convert_to_tensor(x, dtype=tf.float16)
        shape = tf.convert_to_tensor(b, dtype=tf.float16)
        try:
            hand = tf.cast(x, dtype=tf.float16)
            shape = tf.cast(b, dtype=tf.float16)
        except:
            hand = tf.cast(x, dtype=tf.float32)
            shape = tf.cast(b, dtype=tf.float32)

        result = self.model({'X_input': hand, 'B_input': shape}, training=False)  # Using model() instead of model.predict()
        return result
