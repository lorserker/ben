#!/bin/bash

# this is all in one wrapper script mainly for container
# Suppress TensorFlow CUDA/XLA warnings (no GPU in container)
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

python3 gameserver.py 2>&1 | grep -v "cuda\|cuDNN\|cuBLAS\|cuFFT\|cuInit\|CUDA\|absl::InitializeLog" & # listen on 4443 for websocket

python3 gameapi.py --host 0.0.0.0 2>&1 | grep -v "cuda\|cuDNN\|cuBLAS\|cuFFT\|cuInit\|CUDA\|absl::InitializeLog" & # listen on 8085 for REST API

cd "$(dirname "$0")"/frontend
python3 appserver.py --host 0.0.0.0 &  # listen on 8080 for browser

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?