import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Softmax
from scipy.special import softmax

class BatchPlayer:

    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.model = self.init_model()

    def load_model(self):
        model = load_model(self.model_path)
        return model

    def init_model(self):
        model = self.load_model()
        def pred_fun(x):

            card_logit = model.predict(x,verbose=0)
            return card_logit

        return pred_fun

    def next_cards_softmax(self, x):
        result = self.model(x)[:,-1,:]
        return result
