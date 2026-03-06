"""
ONNX Runtime based Bidder model.
Drop-in replacement for bidder_tf2.py with faster inference.
"""

import numpy as np
from nn.timing import ModelTimer
from nn.onnx_config import create_session


class Bidder:
    """ONNX-based bidding model - API compatible with bidder_tf2.Bidder"""

    def __init__(self, name, model_path, alert_supported):
        self.alert_supported = alert_supported
        self.name = name
        self.model_path = model_path
        self.session = create_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def pred_fun_seq(self, x):
        """
        Run inference on the bidder model.

        Args:
            x: numpy array of shape [batch, seq_len, 193]

        Returns:
            bids: numpy array of shape [batch, seq_len, 40]
            alerts: 0 (alert not supported in ONNX version)
        """
        with ModelTimer.time_call('bidder'):
            x_input = np.asarray(x, dtype=np.float32)
            result = self.session.run(None, {self.input_name: x_input})
            bids = result[0]
        # Alert model not included in ONNX conversion
        alerts = np.zeros_like(bids[:, :, :1]) if self.alert_supported else 0
        return bids, alerts
