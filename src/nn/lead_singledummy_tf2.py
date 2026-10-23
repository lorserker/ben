import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from nn.timing import ModelTimer
from nn.tf_guard import tf_lock
from nn.batch_pad import padded_size, pad, unpad

class LeadSingleDummy:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        return load_model(self.model_path, compile=False)

    # Rank 2, any width: the SD model takes 165 features, the no-lead (RPDD) model 133.
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def pred_fun_tf(self, x):
        result = self.model(x, training=False)
        return result


    def pred_fun(self, x):
        n = len(x)
        with tf_lock:
            with ModelTimer.time_call('single_dummy'):
                result = unpad(self.pred_fun_tf(pad(np.asarray(x, dtype=np.float32), padded_size(n))), n)
            return result.numpy()

