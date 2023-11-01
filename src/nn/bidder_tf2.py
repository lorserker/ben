import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging
import os

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.ERROR)
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Bidder:
    
    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.model = self.init_model()
        self.model_seq = self.init_model_seq()
        self.zero_state = (0,None)
        self.model = self.init_model()

    def load_model(self):
        model = load_model(self.model_path)
        return model

    def init_model(self):
        model = self.load_model()
        def pred_fun(x, state):
            if state[0] == 0:
                filled_x = np.zeros((1,8,159))
            else:
                filled_x = state[1]
            # Now we need to update the sequence input                
            filled_x[:, state[0]:state[0] + 1, :x.shape[1]] = x
            bids = model.predict(filled_x, verbose=0)
            state = (state[0]+1,filled_x)
            return [bids[0, state[0]-1, :]], state
        return pred_fun

    def init_model_seq(self):
        model = self.load_model()
        def pred_fun_seq(x):
            filled_x = np.zeros((x.shape[0],8,159))
            filled_x[:, :x.shape[1], :] = x[:, :x.shape[1], :]
            #sequence_length = 8
            #x_reshaped = np.repeat(x, sequence_length, axis=1)
            #print(x_reshaped.shape)
            bids = model.predict(filled_x, verbose=0)
            #print("pred_fun_seq: ",bids.shape)
            return bids[:, :x.shape[1], :]

        return pred_fun_seq

