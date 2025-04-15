import tensorflow as tf
from tensorflow.keras.models import load_model

class Trick:
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        return load_model(self.model_path, compile=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 55], dtype=tf.float16)])
    def pred_fun(self,x):
        input_tensor = tf.cast(x, dtype=tf.float16)
        # Forward pass through the model (note that we now use the TensorFlow model)
        tricks = self.model(input_tensor, training=False)  # Using the model directly (no `predict` method)
       
        return tricks
