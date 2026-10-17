"""
Keras version workarounds for loading BEN's models.

TimeDistributed in Keras 3.15 (graph mode, unknown number of timesteps) loops over
the batch dimension instead of the time dimension: it takes ops.shape(inputs)[0]
before transposing batch and time. Called through a tf.function with an
input_signature of [None, None, n] - which is how BEN calls the bidder and bid info
models - it returns one output step per board in the batch instead of one per bid.

The TimeDistributed layers in BEN's models wrap a Dense layer, and Dense already
works on the last axis of a 3D input, so applying the Dense layer to the whole
sequence gives identical results in every Keras version.
"""
import keras
from tensorflow.keras.models import load_model as _load_model


def _sequence_call(dense):
    def call(inputs, training=None, mask=None):
        return dense(inputs)
    return call


def load_model(model_path, compile=False):
    model = _load_model(model_path, compile=compile)
    for layer in model.layers:
        if isinstance(layer, keras.layers.TimeDistributed) and isinstance(layer.layer, keras.layers.Dense):
            layer.call = _sequence_call(layer.layer)
    return model
