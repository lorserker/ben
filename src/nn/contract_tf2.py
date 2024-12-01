import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class Contract:
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        # Load the TensorFlow 2.x saved model
        model = load_model(self.model_path)
        return model

    def pred_funold(self, x):
        # Perform inference
        predictions = self.model.predict(x, verbose=0)
        contract_logits, bool1_logits, tricks_logits = predictions

        doubled = bool1_logits[0] > 0
        tricks = int(np.argmax(tricks_logits, axis=1)[0])
        contract_id = np.argmax(contract_logits, axis=1)[0]
        return contract_id, doubled, tricks

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float16)])
    def pred_fun(self,x):
        input_tensor = tf.cast(x, dtype=tf.float16)
        # Forward pass through the model (note that we now use the TensorFlow model)
        contract_logits, bool1_logits, tricks_logits = self.model(input_tensor, training=False)  # Using the model directly (no `predict` method)

        # Apply softmax to contract_logits
        contract_probs = tf.nn.softmax(contract_logits)
        
        # Print the softmax probabilities for contract_logits
        # tf.print("contract_probs (softmax of contract_logits):", contract_probs)
        doubled = bool1_logits[0] > 0
        tricks = tf.argmax(tricks_logits, axis=1)[0]
        contract_id = tf.argmax(contract_logits, axis=1)[0]
        score = contract_probs[0][contract_id]
        return contract_id, doubled, tricks, score

    def get_top_k_tricks(self, x, k=3):
        # Perform inference
        predictions = self.model.predict(x, verbose=0)
        tricks_logits = predictions[2]

        probs = tf.nn.softmax(tricks_logits)
        indices = tf.argsort(tricks_logits, axis=1, direction='DESCENDING')

        top_k_indices = indices[:, :k].numpy()
        top_k_probs = probs.numpy()[np.arange(probs.shape[0])[:, np.newaxis], top_k_indices]

        return top_k_indices, top_k_probs
    
    def get_top_k_oh(self, x, k=3):
        # Perform inference
        predictions = self.model.predict(x, verbose=0)
        contract_logits = predictions[0]

        probs = tf.nn.softmax(contract_logits)
        indices = tf.argsort(contract_logits, axis=1, direction='DESCENDING')

        top_k_indices = indices[:, :k].numpy()
        top_k_probs = probs.numpy()[np.arange(probs.shape[0])[:, np.newaxis], top_k_indices]

        return top_k_indices, top_k_probs         
        
