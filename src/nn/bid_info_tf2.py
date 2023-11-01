import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class BidInfo:
    

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.init_model()

    def load_model(self):
        model = load_model(self.model_path)
        return model

        
    def init_model(self):
        model = self.load_model()

        def pred_fun(x):
            out_hcp_seq, out_shape_seq = model.predict(x,verbose=0) 
            return out_hcp_seq, out_shape_seq

        return pred_fun
