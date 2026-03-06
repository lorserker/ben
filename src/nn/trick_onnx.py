"""
ONNX Runtime based Trick model.
Drop-in replacement for trick_tf2.py with faster inference.
"""

import numpy as np
from nn.timing import ModelTimer
from nn.onnx_config import create_session


class Trick:
    """ONNX-based trick model - API compatible with trick_tf2.Trick"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.session = create_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def pred_fun(self, x):
        """
        Run inference on the trick model.

        Args:
            x: numpy array of shape [batch, 55]

        Returns:
            trick prediction
        """
        with ModelTimer.time_call('trick'):
            x_input = np.asarray(x, dtype=np.float32)
            result = self.session.run(None, {self.input_name: x_input})
        return result[0]
