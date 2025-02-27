import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class BidInfo:
    

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        return load_model(self.model_path, compile=False)
    

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float16)])
    def pred_fun_tf(self,x):
        # Ensure that x is a tensor
        input_tensor = tf.convert_to_tensor(x, dtype=tf.float16)
        
        # Perform the model prediction (returns tensors)
        return self.model(input_tensor, training=False)  # Use model call instead of predict

    def pred_fun(self,x):
        out_hcp_seq, out_shape_seq = self.pred_fun_tf(x)  # Call the tf.function
        return out_hcp_seq.numpy(), out_shape_seq.numpy()  # Convert in this function
