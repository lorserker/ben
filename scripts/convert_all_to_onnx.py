"""
Convert all BEN neural network models from Keras to ONNX format.

Usage:
    python convert_all_to_onnx.py

This will create ONNX versions of all models in models/onnx/
"""

import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_model_dtype(model):
    """Get the dtype of the model's first input."""
    if model.inputs:
        return model.inputs[0].dtype
    return tf.float16


def convert_single_input_model(keras_path, onnx_path, input_shape, test_input_fn, model_name):
    """
    Convert a single-input Keras model to ONNX.

    Args:
        keras_path: Path to .keras file
        onnx_path: Path for output .onnx file
        input_shape: TensorSpec shape (e.g., [None, 50])
        test_input_fn: Function that returns test input array
        model_name: Name for logging
    """
    print(f"\n{'='*60}")
    print(f"Converting {model_name}")
    print(f"  From: {keras_path}")
    print(f"  To: {onnx_path}")

    if not os.path.exists(keras_path):
        print(f"  ERROR: Model not found: {keras_path}")
        return False

    model = load_model(keras_path, compile=False)
    model_dtype = get_model_dtype(model)
    print(f"  Model input dtype: {model_dtype}")

    # Import tf2onnx
    import tf2onnx

    # Create wrapper for float16 models
    class WrapperModel(tf.keras.Model):
        def __init__(self, base_model, use_fp16):
            super().__init__()
            self.base_model = base_model
            self.use_fp16 = use_fp16

        def call(self, x):
            if self.use_fp16:
                x = tf.cast(x, tf.float16)
            result = self.base_model(x, training=False)
            return tf.cast(result, tf.float32)

    use_fp16 = (model_dtype == tf.float16)
    wrapper = WrapperModel(model, use_fp16)

    # Create concrete function
    @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='input')])
    def inference(x):
        return wrapper(x)

    # Test the wrapper
    test_input = test_input_fn()
    _ = inference(tf.constant(test_input, dtype=tf.float32))
    print(f"  Wrapper test successful")

    # Create output directory
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # Convert to ONNX
    model_proto, _ = tf2onnx.convert.from_function(
        inference,
        input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='input')],
        output_path=onnx_path,
        opset=13
    )
    print(f"  Saved ONNX model")

    # Verify conversion
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)

    # Get input name
    input_name = session.get_inputs()[0].name

    # Run TF inference
    tf_result = inference(tf.constant(test_input, dtype=tf.float32))
    if isinstance(tf_result, (list, tuple)):
        tf_output = tf_result[0].numpy()
    else:
        tf_output = tf_result.numpy()

    # Run ONNX inference
    onnx_result = session.run(None, {input_name: test_input})
    onnx_output = onnx_result[0]

    # Compare
    max_diff = np.max(np.abs(tf_output - onnx_output))
    print(f"  Verification: max_diff = {max_diff:.6f}")

    if max_diff < 0.01:
        print(f"  [OK] Conversion successful!")
        return True
    else:
        print(f"  [FAIL] WARNING: Outputs differ significantly!")
        return False


def convert_multi_output_model(keras_path, onnx_path, input_shape, test_input_fn, model_name, num_outputs=2):
    """
    Convert a Keras model with multiple outputs to ONNX.
    """
    print(f"\n{'='*60}")
    print(f"Converting {model_name} (multi-output)")
    print(f"  From: {keras_path}")
    print(f"  To: {onnx_path}")

    if not os.path.exists(keras_path):
        print(f"  ERROR: Model not found: {keras_path}")
        return False

    model = load_model(keras_path, compile=False)
    model_dtype = get_model_dtype(model)
    print(f"  Model input dtype: {model_dtype}")

    import tf2onnx

    use_fp16 = (model_dtype == tf.float16)

    # Create concrete function that handles multi-output
    @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='input')])
    def inference(x):
        if use_fp16:
            x = tf.cast(x, tf.float16)
        results = model(x, training=False)
        if isinstance(results, (list, tuple)):
            return tuple(tf.cast(r, tf.float32) for r in results)
        return tf.cast(results, tf.float32)

    # Test
    test_input = test_input_fn()
    _ = inference(tf.constant(test_input, dtype=tf.float32))
    print(f"  Wrapper test successful")

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    model_proto, _ = tf2onnx.convert.from_function(
        inference,
        input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='input')],
        output_path=onnx_path,
        opset=13
    )
    print(f"  Saved ONNX model")

    # Verify
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    tf_results = inference(tf.constant(test_input, dtype=tf.float32))
    onnx_results = session.run(None, {input_name: test_input})

    all_match = True
    if isinstance(tf_results, (list, tuple)):
        for i, (tf_out, onnx_out) in enumerate(zip(tf_results, onnx_results)):
            max_diff = np.max(np.abs(tf_out.numpy() - onnx_out))
            print(f"  Output {i}: max_diff = {max_diff:.6f}")
            if max_diff >= 0.01:
                all_match = False
    else:
        max_diff = np.max(np.abs(tf_results.numpy() - onnx_results[0]))
        print(f"  Verification: max_diff = {max_diff:.6f}")
        all_match = max_diff < 0.01

    if all_match:
        print(f"  [OK] Conversion successful!")
        return True
    else:
        print(f"  [FAIL] WARNING: Outputs differ significantly!")
        return False


def convert_dual_input_model(keras_path, onnx_path, x_shape, b_shape, test_input_fn, model_name):
    """
    Convert a dual-input Keras model (like Lead models) to ONNX.
    """
    print(f"\n{'='*60}")
    print(f"Converting {model_name} (dual-input)")
    print(f"  From: {keras_path}")
    print(f"  To: {onnx_path}")

    if not os.path.exists(keras_path):
        print(f"  ERROR: Model not found: {keras_path}")
        return False

    model = load_model(keras_path, compile=False)
    model_dtype = get_model_dtype(model)
    print(f"  Model input dtype: {model_dtype}")

    import tf2onnx

    use_fp16 = (model_dtype == tf.float16)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=x_shape, dtype=tf.float32, name='X_input'),
        tf.TensorSpec(shape=b_shape, dtype=tf.float32, name='B_input')
    ])
    def inference(x, b):
        if use_fp16:
            x = tf.cast(x, tf.float16)
            b = tf.cast(b, tf.float16)
        result = model({'X_input': x, 'B_input': b}, training=False)
        return tf.cast(result, tf.float32)

    test_x, test_b = test_input_fn()
    _ = inference(tf.constant(test_x, dtype=tf.float32), tf.constant(test_b, dtype=tf.float32))
    print(f"  Wrapper test successful")

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    model_proto, _ = tf2onnx.convert.from_function(
        inference,
        input_signature=[
            tf.TensorSpec(shape=x_shape, dtype=tf.float32, name='X_input'),
            tf.TensorSpec(shape=b_shape, dtype=tf.float32, name='B_input')
        ],
        output_path=onnx_path,
        opset=13
    )
    print(f"  Saved ONNX model")

    # Verify
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)

    tf_result = inference(tf.constant(test_x, dtype=tf.float32), tf.constant(test_b, dtype=tf.float32))
    onnx_result = session.run(None, {'X_input': test_x, 'B_input': test_b})

    max_diff = np.max(np.abs(tf_result.numpy() - onnx_result[0]))
    print(f"  Verification: max_diff = {max_diff:.6f}")

    if max_diff < 0.01:
        print(f"  [OK] Conversion successful!")
        return True
    else:
        print(f"  [FAIL] WARNING: Outputs differ significantly!")
        return False


def main():
    base_path = os.path.dirname(os.path.dirname(__file__))
    onnx_dir = os.path.join(base_path, 'models', 'onnx')
    tf2_dir = os.path.join(base_path, 'models', 'TF2models')

    results = {}

    # ========== Contract Model ==========
    results['contract'] = convert_single_input_model(
        keras_path=os.path.join(tf2_dir, 'Contract_2024-12-09-E50.keras'),
        onnx_path=os.path.join(onnx_dir, 'contract.onnx'),
        input_shape=[None, 50],
        test_input_fn=lambda: np.random.randn(1, 50).astype(np.float32),
        model_name='Contract'
    )

    # ========== Trick Model ==========
    results['trick'] = convert_single_input_model(
        keras_path=os.path.join(tf2_dir, 'Tricks_2024-12-09-E50.keras'),
        onnx_path=os.path.join(onnx_dir, 'trick.onnx'),
        input_shape=[None, 55],
        test_input_fn=lambda: np.random.randn(1, 55).astype(np.float32),
        model_name='Trick'
    )

    # ========== Bidder Model ==========
    # Bidder: input [None, None, 193] dtype=float16 → output [None, None, 40]
    results['bidder'] = convert_single_input_model(
        keras_path=os.path.join(tf2_dir, 'GIB-BBO-8730_2025-04-19-E30.keras'),
        onnx_path=os.path.join(onnx_dir, 'bidder.onnx'),
        input_shape=[None, None, 193],
        test_input_fn=lambda: np.random.randn(1, 10, 193).astype(np.float32),
        model_name='Bidder'
    )

    # ========== BidInfo Model ==========
    # BidInfo: input [None, None, 193] dtype=float16 → outputs [None, None, 3], [None, None, 12]
    results['bidinfo'] = convert_multi_output_model(
        keras_path=os.path.join(tf2_dir, 'GIB-BBOInfo-8730_2025-04-19-E30.keras'),
        onnx_path=os.path.join(onnx_dir, 'bidinfo.onnx'),
        input_shape=[None, None, 193],
        test_input_fn=lambda: np.random.randn(1, 10, 193).astype(np.float32),
        model_name='BidInfo',
        num_outputs=2
    )

    # ========== Single Dummy Estimator ==========
    # SD: input [None, 165] dtype=float32 → output [None, 14]
    results['sd'] = convert_single_input_model(
        keras_path=os.path.join(tf2_dir, 'SD_2024-07-08-E20.keras'),
        onnx_path=os.path.join(onnx_dir, 'single_dummy.onnx'),
        input_shape=[None, 165],
        test_input_fn=lambda: np.random.randn(1, 165).astype(np.float32),
        model_name='SingleDummy'
    )

    # ========== Double Dummy Estimator (RPDD) ==========
    # RPDD: input [None, 133] dtype=float32 → output [None, 14]
    results['rpdd'] = convert_single_input_model(
        keras_path=os.path.join(tf2_dir, 'RPDD_2024-07-08-E02.keras'),
        onnx_path=os.path.join(onnx_dir, 'double_dummy.onnx'),
        input_shape=[None, 133],
        test_input_fn=lambda: np.random.randn(1, 133).astype(np.float32),
        model_name='DoubleDummy (RPDD)'
    )

    # ========== Player Models (8 total) ==========
    player_models = [
        ('lefty_nt', 'lefty_nt_2024-07-08-E20.keras'),
        ('dummy_nt', 'dummy_nt_2024-07-08-E20.keras'),
        ('righty_nt', 'righty_nt_2024-07-16-E20.keras'),
        ('decl_nt', 'decl_nt_2024-07-08-E20.keras'),
        ('lefty_suit', 'lefty_suit_2024-07-08-E20.keras'),
        ('dummy_suit', 'dummy_suit_2024-07-08-E20.keras'),
        ('righty_suit', 'righty_suit_2024-07-16-E20.keras'),
        ('decl_suit', 'decl_suit_2024-07-08-E20.keras'),
    ]

    for name, keras_file in player_models:
        results[name] = convert_single_input_model(
            keras_path=os.path.join(tf2_dir, keras_file),
            onnx_path=os.path.join(onnx_dir, f'{name}.onnx'),
            input_shape=[None, None, 298],
            test_input_fn=lambda: np.random.randn(1, 13, 298).astype(np.float32),
            model_name=f'Player ({name})'
        )

    # ========== Lead Models (already done, but include for completeness) ==========
    results['lead_suit'] = convert_dual_input_model(
        keras_path=os.path.join(tf2_dir, 'Lead-Suit_2024-11-04-E200.keras'),
        onnx_path=os.path.join(onnx_dir, 'lead_suit.onnx'),
        x_shape=[None, 42],
        b_shape=[None, 15],
        test_input_fn=lambda: (np.random.randn(1, 42).astype(np.float32),
                               np.random.randn(1, 15).astype(np.float32)),
        model_name='Lead Suit'
    )

    results['lead_nt'] = convert_dual_input_model(
        keras_path=os.path.join(tf2_dir, 'Lead-NT_2024-11-04-E200.keras'),
        onnx_path=os.path.join(onnx_dir, 'lead_nt.onnx'),
        x_shape=[None, 42],
        b_shape=[None, 15],
        test_input_fn=lambda: (np.random.randn(1, 42).astype(np.float32),
                               np.random.randn(1, 15).astype(np.float32)),
        model_name='Lead NT'
    )

    # ========== Summary ==========
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {success_count}/{total_count} models converted successfully")

    if success_count == total_count:
        print("\nAll models converted! You can now update your config to use ONNX models.")
    else:
        print("\nSome models failed conversion. Check the errors above.")


if __name__ == "__main__":
    main()
