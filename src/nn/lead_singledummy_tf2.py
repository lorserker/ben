import numpy as np
from tensorflow import keras

class LeadSingleDummy:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        model = keras.models.load_model(self.model_path)
        return model

    def pred_fun(self, x):
        result = self.model.predict(x,verbose=0)
        return result

