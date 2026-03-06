"""
Shared ONNX Runtime configuration for all models.

Centralizes session options to prevent thread contention.
"""

import onnxruntime as ort

# Global session options - shared by all ONNX models
def get_session_options():
    """Get optimized session options for ONNX Runtime."""
    sess_options = ort.SessionOptions()

    # Memory optimization
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True

    # Thread settings - limit to prevent contention
    # Use 1 thread per model to avoid contention with multiple models
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    # Execution mode - sequential is often faster for small batches
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Graph optimization - use extended for balance of speed vs memory
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    return sess_options


def create_session(model_path):
    """Create an ONNX inference session with optimized settings."""
    return ort.InferenceSession(model_path, get_session_options())
