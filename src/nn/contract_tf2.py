import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class Contract:
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.init_model()

    def load_model(self):
        # Load the TensorFlow 2.x saved model
        model = load_model(self.model_path)
        return model

    def init_model(self):
        model = self.load_model()

        def pred_fun(x):
            # Perform inference
            predictions = model.predict(x, verbose=0)
            contract_logits, bool1_logits, tricks_logits = predictions

            doubled = bool1_logits[0] > 0
            tricks = int(np.argmax(tricks_logits, axis=1)[0])
            contract_id = np.argmax(contract_logits, axis=1)[0]
            return contract_id, doubled, tricks

        def get_top_k_tricks(x, k=3):
            # Perform inference
            predictions = model.predict(x, verbose=0)
            tricks_logits = predictions[2]

            probs = tf.nn.softmax(tricks_logits)
            indices = tf.argsort(tricks_logits, axis=1, direction='DESCENDING')

            top_k_indices = indices[:, :k].numpy()
            top_k_probs = probs.numpy()[np.arange(probs.shape[0])[:, np.newaxis], top_k_indices]

            return top_k_indices, top_k_probs
        
        def get_top_k_oh(x, k=3):
            # Perform inference
            predictions = model.predict(x, verbose=0)
            contract_logits = predictions[0]

            probs = tf.nn.softmax(contract_logits)
            indices = tf.argsort(contract_logits, axis=1, direction='DESCENDING')

            top_k_indices = indices[:, :k].numpy()
            top_k_probs = probs.numpy()[np.arange(probs.shape[0])[:, np.newaxis], top_k_indices]

            return top_k_indices, top_k_probs         
        
        return pred_fun, get_top_k_tricks, get_top_k_oh