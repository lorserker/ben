"""Training epochs with the data on the GPU, for the bidding and bidding-info trainers.

model.fit() with a generator spends most of a step outside the GPU: one program launch and one
batch transfer per step (about 6 ms per step on an RTX 5080, GPU under 40% busy). Here the whole
training set is put on the GPU once and several hundred steps run as one XLA program, which
brings the same step, at the same batch size, to 1 to 2 ms.

The training set fits because the input array is highly redundant: of the features of one
bidding round, all but the last 4 * n_bids are the same in every round of a hand, and those
last ones are four one-hot encoded bids. compact_inputs() stores them as (static, bid indices)
and checks that this loses nothing; if the array is not of that form it returns None and the
caller trains with model.fit() as before.

Each step is model.train_step(), so loss, metrics, optimizer and learning rate schedule are
exactly those of model.fit().
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks as keras_callbacks

ONE_F16 = 0x3C00   # bit pattern of float16 1.0


def compact_inputs(x, n_bids=40, chunk=50_000, threads=8):
    """x: (examples, rounds, features) float16, possibly memory mapped.

    Returns (static, bids): static (examples, features - 4 * n_bids) float16 with the features
    that do not change between rounds, bids (examples, rounds, 4) uint8 with the index of each
    one-hot bid (n_bids for an all-zero block). Returns None when x cannot be stored that way
    without loss.
    """
    n, n_rounds, n_ftrs = x.shape
    n_static = n_ftrs - 4 * n_bids
    if x.dtype != np.float16 or n_static <= 0:
        return None
    static = np.empty((n, n_static), dtype=np.float16)
    bids = np.empty((n, n_rounds, 4), dtype=np.uint8)

    def do(start):
        c = np.asarray(x[start:start + chunk])
        m = len(c)
        s = c[:, 0, :n_static]
        if not (c[:, :, :n_static].view(np.uint16) == s.view(np.uint16)[:, None, :]).all():
            return False
        blocks = np.ascontiguousarray(c[:, :, n_static:]).view(np.uint16).reshape(m, n_rounds, 4, n_bids)
        count = np.count_nonzero(blocks, axis=-1)
        if count.max() > 1 or not (blocks.max(axis=-1) == np.where(count == 1, ONE_F16, 0)).all():
            return False
        static[start:start + m] = s
        bids[start:start + m] = np.where(count == 1, blocks.argmax(axis=-1), n_bids)
        return True

    with ThreadPoolExecutor(threads) as pool:
        if not all(pool.map(do, range(0, n, chunk))):
            return None
    return static, bids


def compact_one_hot(y, chunk=500_000):
    """y: (examples, rounds, classes) one-hot integers. Returns the class indexes (examples,
    rounds) uint8, or None when some row is not exactly one-hot."""
    if y.dtype.kind not in 'ui' or y.shape[-1] > 255:
        return None
    idx = np.empty(y.shape[:-1], dtype=np.uint8)
    for a in range(0, len(y), chunk):
        c = np.asarray(y[a:a + chunk])
        if not ((c.max(axis=-1) == 1) & (c.sum(axis=-1, dtype=np.int32) == 1)).all():
            return None
        idx[a:a + chunk] = c.argmax(axis=-1)
    return idx


def compact_rounds(t):
    """t: (examples, rounds, values). Returns (examples, values) when every round of an example
    has the same values (as the HCP and shape targets do), else None."""
    first = np.ascontiguousarray(t[:, 0])
    bits = np.dtype(f'u{t.dtype.itemsize}')
    for r in range(1, t.shape[1]):
        if not (np.ascontiguousarray(t[:, r]).view(bits) == first.view(bits)).all():
            return None
    return first


def expand_inputs(static, bids, n_bids=40):
    """The inverse of compact_inputs, in numpy (for checks)."""
    n, n_rounds, _ = bids.shape
    one_hot = np.concatenate([np.eye(n_bids, dtype=np.float16), np.zeros((1, n_bids), np.float16)])[bids]
    return np.concatenate([np.repeat(static[:, None, :], n_rounds, axis=1),
                           one_hot.reshape(n, n_rounds, 4 * n_bids)], axis=2)


def fit(model, inputs, targets, batch_size, epochs, steps_per_epoch, callbacks=(), n_bids=40,
        steps_per_call=512, one_hot_targets=None, repeat_targets=False):
    """Train like model.fit(generator of shuffled whole batches, epochs, steps_per_epoch, callbacks).

    inputs: (static, bids) from compact_inputs. targets: one array or a tuple of arrays, one
    per model output, first axis examples. With one_hot_targets=(classes, dtype) the targets
    are class indexes from compact_one_hot and the model gets them one-hot encoded, as the
    original array was. With repeat_targets=True the targets are (examples, values) arrays from
    compact_rounds and the model gets them repeated for every round. Every epoch uses a fresh
    permutation and drops the
    last partial batch. The callbacks get on_train_begin, on_epoch_begin, on_epoch_end with
    the same logs as from model.fit(), and on_train_end; model.stop_training is honoured.
    """
    static, bids = inputs
    n_examples, n_rounds, _ = bids.shape
    steps_per_call = next((k for k in sorted(range(128, 1025), key=lambda k: abs(k - steps_per_call))
                           if steps_per_epoch % k == 0), steps_per_call)
    single_target = not isinstance(targets, (tuple, list))
    targets = (targets,) if single_target else tuple(targets)

    t_start = time.time()
    try:
        with tf.device('/GPU:0'):
            g_static = tf.constant(static)
            g_bids = tf.constant(bids)
            g_targets = tuple(tf.constant(t) for t in targets)
    except tf.errors.ResourceExhaustedError:
        raise SystemExit("The training set does not fit on the GPU. Run the trainer with gpu_data=0.")

    def batch(idx):
        s = tf.gather(g_static, idx)
        b = tf.one_hot(tf.gather(g_bids, idx), n_bids, dtype=tf.float16)
        x = tf.concat([tf.repeat(s[:, None, :], n_rounds, axis=1),
                       tf.reshape(b, (batch_size, n_rounds, 4 * n_bids))], axis=2)
        y = tuple(tf.gather(t, idx) for t in g_targets)
        if one_hot_targets:
            y = tuple(tf.one_hot(t, one_hot_targets[0], dtype=one_hot_targets[1]) for t in y)
        if repeat_targets:
            y = tuple(tf.repeat(t[:, None, :], n_rounds, axis=1) for t in y)
        return x, (y[0] if single_target else y)

    # Create the loss, metric and optimizer variables before the compiled loop is traced
    x0, y0 = batch(tf.range(batch_size))
    y_pred = model(x0, training=False)
    model.compute_loss(x=x0, y=y0, y_pred=y_pred)
    model.compute_metrics(x0, y0, y_pred)
    model.optimizer.build(model.trainable_variables)
    model.reset_metrics()
    print(f"Training data on the GPU and model ready in {time.time() - t_start:.1f} s", flush=True)
    first_call = True

    # A call with a different number of steps compiles this program a second time (n_steps
    # becomes the loop's maximum_iterations, a compile-time constant), about 13 s. So the
    # steps of an epoch are split into equal calls when a suitable divisor exists. An explicit
    # tf.while_loop or a fixed count with tf.cond would also avoid the second compilation, but
    # made every step 5 to 30% slower (measured 2026-11-17).
    @tf.function(jit_compile=True)
    def run_steps(order, n_steps):
        for i in tf.range(n_steps):
            model.train_step(batch(order[i]))

    cbs = keras_callbacks.CallbackList(list(callbacks), add_history=False, add_progbar=False, model=model)
    model.stop_training = False
    cbs.on_train_begin()
    logs = {}
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}", flush=True)
        cbs.on_epoch_begin(epoch)
        model.reset_metrics()
        order = np.random.permutation(n_examples)[:steps_per_epoch * batch_size]
        order = order.reshape(steps_per_epoch, batch_size).astype(np.int32)
        t_epoch = time.time()
        done = 0
        next_report = 0
        while done < steps_per_epoch:
            n_steps = min(steps_per_call, steps_per_epoch - done)
            part = order[done:done + n_steps]
            if n_steps < steps_per_call:     # same array shape for every call
                part = np.concatenate([part, np.zeros((steps_per_call - n_steps, batch_size), np.int32)])
            t_call = time.time()
            run_steps(tf.constant(part), tf.constant(n_steps))
            if first_call:
                first_call = False
                print(f"First {n_steps} steps, with compiling the loop: {time.time() - t_call:.1f} s", flush=True)
            done += n_steps
            if done >= next_report or done == steps_per_epoch:
                next_report = done + steps_per_epoch // 10
                logs = {k: float(v) for k, v in model.get_metrics_result().items()}
                elapsed = time.time() - t_epoch
                print(f"{done}/{steps_per_epoch} - {elapsed:.0f}s {1000 * elapsed / done:.2f}ms/step - "
                      + " - ".join(f"{k}: {v:.4f}" for k, v in sorted(logs.items())), flush=True)
        cbs.on_epoch_end(epoch, logs)
        if model.stop_training:
            break
    cbs.on_train_end(logs)
    return logs
