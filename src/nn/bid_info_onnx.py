"""
ONNX Runtime based BidInfo model.
Drop-in replacement for bid_info_tf2.py with faster inference.
"""

import numpy as np
from nn.timing import ModelTimer
from nn.onnx_config import create_session


class BidInfo:
    """ONNX-based bidding info model - API compatible with bid_info_tf2.BidInfo"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.session = create_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def pred_fun(self, x):
        """
        Run inference on the bid info model.

        Args:
            x: numpy array of shape [batch, seq_len, 193]

        Returns:
            out_hcp_seq: numpy array of shape [batch, seq_len, 3]
            out_shape_seq: numpy array of shape [batch, seq_len, 12]
        """
        with ModelTimer.time_call('bidinfo'):
            x_input = np.asarray(x, dtype=np.float32)
            results = self.session.run(None, {self.input_name: x_input})
        return results[0], results[1]
