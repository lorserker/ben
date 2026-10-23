import tensorflow as tf
from tensorflow.keras.models import load_model
from nn.timing import ModelTimer
from nn.tf_guard import tf_lock
from nn.batch_pad import padded_size, pad, unpad

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
    def pred_fun_tf(self, x, b):
        # Pass the inputs as a list, not a dict: Keras <= 3.6 and >= 3.7 order dict inputs
        # differently, see scripts/training/opening lead/fix_lead_input_order.py
        result = self.model([x, b], training=False) # Using model() instead of model.predict()
        return result

    def pred_fun(self, x, b):
        n = len(x)
        size = padded_size(n)
        with tf_lock:
            with ModelTimer.time_call('leader'):
                result = unpad(self.pred_fun_tf(pad(x, size), pad(b, size)), n)
            return result.numpy()
