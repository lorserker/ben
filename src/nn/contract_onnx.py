"""
ONNX Runtime based Contract model.
Drop-in replacement for contract_tf2.py with faster inference.
"""

import numpy as np
from nn.timing import ModelTimer
from nn.onnx_config import create_session


class Contract:
    """ONNX-based contract model - API compatible with contract_tf2.Contract"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.session = create_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def pred_fun(self, x):
        """
        Run inference on the contract model.

        Args:
            x: numpy array of shape [batch, 50]

        Returns:
            contract prediction
        """
        with ModelTimer.time_call('contract'):
            x_input = np.asarray(x, dtype=np.float32)
            result = self.session.run(None, {self.input_name: x_input})
        return result[0]
