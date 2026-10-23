import tensorflow as tf
from tensorflow.keras.models import load_model
from nn.timing import ModelTimer
from nn.tf_guard import tf_lock
from nn.batch_pad import padded_size, pad, unpad

class Contract:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        return load_model(self.model_path, compile=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 50], dtype=tf.float16)])
    def _pred_fun_tf(self, x):
        input_tensor = tf.cast(x, dtype=tf.float16)
        # Forward pass through the model (note that we now use the TensorFlow model)
        contract = self.model(input_tensor, training=False)  # Using the model directly (no `predict` method)
        return contract

    def pred_fun(self, x):
        n = len(x)
        with tf_lock:
            with ModelTimer.time_call('contract'):
                result = unpad(self._pred_fun_tf(pad(x, padded_size(n))), n)
        return result
