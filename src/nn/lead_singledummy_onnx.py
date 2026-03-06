"""
ONNX Runtime based LeadSingleDummy model.
Drop-in replacement for lead_singledummy_tf2.py with faster inference.
"""

import numpy as np
from nn.timing import ModelTimer
from nn.onnx_config import create_session


class LeadSingleDummy:
    """ONNX-based single dummy estimator - API compatible with lead_singledummy_tf2.LeadSingleDummy"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.session = create_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def pred_fun(self, x):
        """
        Run inference on the single dummy model.

        Args:
            x: numpy array of shape [batch, 165] for SD or [batch, 133] for RPDD

        Returns:
            result: numpy array of shape [batch, 14]
        """
        with ModelTimer.time_call('single_dummy'):
            x_input = np.asarray(x, dtype=np.float32)
            result = self.session.run(None, {self.input_name: x_input})
        return result[0]
