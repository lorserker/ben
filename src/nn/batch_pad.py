"""Pad model inputs to a few batch sizes, so oneDNN sees a bounded set of shapes.

oneDNN, TensorFlow's CPU kernel library, keeps native memory for every input
shape it has run. BEN calls the models with a new number of samples almost
every time, so a long-running process grew by several GB (2.6 GB to 5.0 GB over
1,000 bids, reported by Jay). Capping the oneDNN primitive cache and the MKL
allocator settings did not help; TF_ENABLE_ONEDNN_OPTS=0 did, but bidding was
13 % slower. Padding the batch up to the next of 4, 5, 6 or 7 times a power of
two keeps memory flat at full speed, and wastes at most 25 % of a call.

The padding rows are zeros and every row is predicted on its own, so the real
rows get exactly the same result.
"""
import numpy as np
import tensorflow as tf


def padded_size(n):
    """Smallest of 1..7 or 4, 5, 6, 7 times a power of two that is >= n."""
    if n <= 7:
        return n
    step = 1 << (n.bit_length() - 3)
    return -(-n // step) * step


def pad(x, size):
    """x with zero rows added up to size rows."""
    x = np.asarray(x)
    n = x.shape[0]
    if n == size:
        return x
    return np.concatenate([x, np.zeros((size - n,) + x.shape[1:], dtype=x.dtype)])


def unpad(result, n):
    """The first n rows of every output (a tensor, an array or a tuple/list of them).
    A scalar output, like the bidder's alerts without alert support, is kept as it is."""
    return tf.nest.map_structure(lambda out: out[:n] if len(out.shape) else out, result)
