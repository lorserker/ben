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

    @staticmethod
    @tf.keras.utils.register_keras_serializable()
    def masked_categorical_crossentropy(y_true, y_pred):
        mask = tf.not_equal(tf.argmax(y_true, axis=-1), 1)
        mask = tf.reshape(mask, [-1])

        y_true_flat = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])

        y_true_masked = tf.boolean_mask(y_true_flat, mask)
        y_pred_masked = tf.boolean_mask(y_pred_flat, mask)

        return tf.keras.losses.categorical_crossentropy(y_true_masked, y_pred_masked)

    def load_model(self):
        return load_model(self.model_path, custom_objects={
            'masked_categorical_crossentropy': Bidder.masked_categorical_crossentropy
        })
    
    def init_model_seq(self):
        model = self.load_model()
        def pred_fun_seq( x):
            bids, alerts = self.model.predict(x, verbose=0)
            return bids, alerts


        return pred_fun_seq


