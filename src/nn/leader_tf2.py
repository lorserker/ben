import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras

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
        # # Forward x and b as a dictionary to the model
        # for inp in self.model.inputs:
        #     print(inp.name)

        # #tf.print(x.shape, b.shape)
        # tf.print("Shape of x:", x.shape)
        # tf.print("Shape of b:", b.shape)
        # tf.print("Shape of x:", keras.ops.shape(x))
        # tf.print("Shape of b:", keras.ops.shape(b))
        # tf.print(self.model.input_shape) # Forward x and b as a dictionary to the model
        result = self.model({'X_input': x, 'B_input': b}, training=False) # Using model() instead of model.predict()
        # tf.print("Prediction result:", result)
        # tf.print("Shape:", result.shape)
        return result
