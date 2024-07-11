import numpy as np
import tensorflow as tf
from collections import namedtuple
from tensorflow.keras.models import load_model


class Bidder:
    
    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.model = self.load_model()
        self.model_seq = self.init_model_seq()


    def load_model(self):
        return load_model(self.model_path)

    
    def init_model_seq(self):
        model = self.load_model()
        def pred_fun_seq( x):
            bids, alerts = self.model.predict(x, verbose=0)
            return bids, alerts


        return pred_fun_seq


