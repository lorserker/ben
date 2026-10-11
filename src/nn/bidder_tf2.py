import tensorflow as tf
from tensorflow.keras.models import load_model
from nn.timing import ModelTimer


class Bidder:

    def __init__(self, name, model_path, alert_supported):
        self.alert_supported = alert_supported
        self.name = name
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        return load_model(self.model_path, compile=False)

    def pred_fun_tf(self, x):
        # Run inference eagerly.
        # This avoids TensorFlow trying to capture model tensors
        # inside a tf.function graph.
        try:
            input_tensor = tf.cast(x, dtype=tf.float16)
        except Exception:
            input_tensor = tf.cast(x, dtype=tf.float32)

        if self.alert_supported:
            bids, alerts = self.model(input_tensor, training=False)
        else:
            bids = self.model(input_tensor, training=False)
            alerts = tf.zeros((1,), dtype=tf.float32)

        return bids, alerts

    def pred_fun_seq(self, x):
        with ModelTimer.time_call('bidder', rows=x):
            bids, alerts = self.pred_fun_tf(x)

        return bids.numpy(), alerts.numpy()
