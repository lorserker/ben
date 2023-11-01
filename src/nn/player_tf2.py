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
            # Keres expect all 11 rows in the sequence as input
            if x.shape[1] != 11:
                # We have only 11 tricks in the neural network, but is trying to find a card for trick 12
                if x.shape[1] > 11:
                    x = x[:, :11, :]
                # Keres expect all 11 rows in the sequence as input
                y = np.zeros((1, 11, 298))
                y[:, :x.shape[1], :] = x
            else:
                y = x
            card_logit = model.predict(y,verbose=0)
            result = self.reshape_card_logit(card_logit, x)

            # I need to figure out, why this sometimes is a tensor and other timnes just a numpy array
            if not isinstance(result, np.ndarray):
                result = result.numpy()
            return result

        return pred_fun

    def reshape_card_logit(self, card_logit, x):
        return softmax(card_logit.reshape((x.shape[0], 11, 32)), axis=2)

    def next_cards_softmax(self, x):
        result = self.model(x)[:,-1,:]
        return result

class BatchPlayerLefty(BatchPlayer):        

    def reshape_card_logit(self, card_logit, x):
        reshaped_card_logit = tf.keras.layers.Reshape((-1, 32), input_shape=(1, -1, 298))(card_logit)
        softmax_card_logit = Softmax(axis=-1)(reshaped_card_logit)
        return softmax_card_logit
