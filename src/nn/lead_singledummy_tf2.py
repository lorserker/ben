import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class LeadSingleDummy:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        return load_model(self.model_path, compile=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 298], dtype=tf.float16)])
    def pred_fun_tf(self, x):
        result = self.model.predict(x,verbose=0)
        return result


    def pred_fun(self, x):
        result = self.pred_fun_tf(x)
        return result.numpy()

