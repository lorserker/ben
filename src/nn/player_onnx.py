"""
ONNX Runtime based BatchPlayer model.
Drop-in replacement for player_tf2.py with faster inference.
"""

import numpy as np
from nn.timing import ModelTimer
from nn.onnx_config import create_session


class BatchPlayer:
    """ONNX-based card player model - API compatible with player_tf2.BatchPlayer"""

    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.session = create_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def pred_fun(self, x):
        """
        Run inference on the player model.

        Args:
            x: numpy array of shape [batch, seq_len, 298]

        Returns:
            card_logit: numpy array of shape [batch, seq_len, 32]
        """
        with ModelTimer.time_call(f'player_{self.name}'):
            x_input = np.asarray(x, dtype=np.float32)
            result = self.session.run(None, {self.input_name: x_input})
        return result[0]

    def next_cards_softmax(self, x):
        """
        Get softmax for next card prediction.

        Args:
            x: numpy array of shape [batch, seq_len, 298]

        Returns:
            result: numpy array of shape [batch, 32] (last sequence position)
        """
        with ModelTimer.time_call(f'player_{self.name}'):
            x_input = np.asarray(x, dtype=np.float32)
            result = self.session.run(None, {self.input_name: x_input})
        return result[0][:, -1, :]
