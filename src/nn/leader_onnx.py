"""
ONNX Runtime based Leader model for opening lead prediction.
Drop-in replacement for leader_tf2.py with ~60x faster inference.

Usage:
    from nn.leader_onnx import Leader
    leader = Leader("models/onnx/lead_suit.onnx")
    result = leader.pred_fun(x, b)
"""

import numpy as np
from nn.timing import ModelTimer
from nn.onnx_config import create_session


class Leader:
    """ONNX-based opening lead model - API compatible with leader_tf2.Leader"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.session = create_session(model_path)

        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def pred_fun(self, x, b):
        """
        Run inference on the lead model.

        Args:
            x: numpy array of shape [batch, 42] - level, strain, doubled, vuln, hand
            b: numpy array of shape [batch, 15] - bidding info (HCP and shape predictions)

        Returns:
            numpy array of shape [batch, 32] - probability for each card
        """
        with ModelTimer.time_call('leader'):
            # Ensure float32 (ONNX model expects float32)
            x_input = np.asarray(x, dtype=np.float32)
            b_input = np.asarray(b, dtype=np.float32)

            # Run inference
            result = self.session.run(
                self.output_names,
                {'X_input': x_input, 'B_input': b_input}
            )

        return result[0]


# For backwards compatibility, also support the tf.function-style call
Leader.pred_fun_tf = Leader.pred_fun
