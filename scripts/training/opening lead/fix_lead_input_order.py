"""
Re-save a lead model so its input order no longer depends on the Keras version.

The lead models have two inputs, X_input (42 hand features) and B_input (15 bidding
features), which the network concatenates. They were trained with Keras <= 3.6 and
fed as a dict ({'X_input': x, 'B_input': b}). Those Keras versions flattened the
dict in sorted key order, so the network was actually trained on [B, X]. Keras 3.7
fixed that and concatenates [X, B], which silently ruins the predictions
(see https://github.com/keras-team/keras/issues/20686).

This script rebuilds the model with the same weights but an explicit
Concatenate([B, X]), so calling it with a list [x, b] gives the trained behaviour
in every Keras version. Run it with Keras 3.5/3.6 (the version the model was
trained with), so the saved file loads in old and new Keras alike.

Usage: python fix_lead_input_order.py input.keras output.keras
"""
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import numpy as np
from keras import layers

if len(sys.argv) != 3:
    print("Usage: python fix_lead_input_order.py input.keras output.keras")
    sys.exit(1)

src, dst = sys.argv[1], sys.argv[2]
old = keras.models.load_model(src, compile=False)

concat_i = next(i for i, l in enumerate(old.layers) if isinstance(l, layers.Concatenate))
x_in, b_in = old.inputs
X = keras.Input(shape=x_in.shape[1:], dtype=x_in.dtype, name='X_input')
B = keras.Input(shape=b_in.shape[1:], dtype=b_in.dtype, name='B_input')
h = layers.Concatenate(name='concatenate_b_x')([B, X])
for layer in old.layers[concat_i + 1:]:
    h = layer(h)
new = keras.Model(inputs=[X, B], outputs=h, name=old.name)

# The rebuilt model called with a list must match the old one called with a dict
# in the Keras version that trained it.
rng = np.random.default_rng(0)
x = (rng.random((256,) + tuple(x_in.shape[1:])) < 0.15).astype(np.float16)
b = rng.random((256,) + tuple(b_in.shape[1:])).astype(np.float16)
ref = np.asarray(old({'X_input': x, 'B_input': b}, training=False))
out = np.asarray(new([x, b], training=False))
diff = float(np.abs(ref - out).max())
print(f"Keras {keras.__version__}: max diff vs original dict call {diff:.2e}")
if diff > 1e-5:
    print("Mismatch - this Keras version does not reproduce the trained input order, nothing saved")
    sys.exit(1)

new.save(dst)
print("Saved", dst)
