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

        def pred_fun(x, b):
            x = np.array(x)  # Ensure that input data is in the right format
            b = np.array(b)
            result = model.predict([x, b],verbose=0)
            result_with_softmax = tf.nn.softmax(result, axis=-1).numpy()  # Apply softmax activation
            return result_with_softmax
        return pred_fun
