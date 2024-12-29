import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model

class LeadSingleDummy:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        return load_model(self.model_path, compile=False)

    def pred_fun(self, x):
        result = self.model.predict(x,verbose=0)
        return result

