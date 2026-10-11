#!/bin/bash

# this is all in one wrapper script mainly for container
# Suppress TensorFlow CUDA/XLA warnings (no GPU in container)
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

python3 gameserver.py 2>&1 | grep -v "cuda\|cuDNN\|cuBLAS\|cuFFT\|cuInit\|CUDA\|absl::InitializeLog" & # listen on 4443 for websocket

# Output through process substitution rather than a pipe, so $! is gameapi's own pid
python3 gameapi.py --host 0.0.0.0 > >(grep -v "cuda\|cuDNN\|cuBLAS\|cuFFT\|cuInit\|CUDA\|absl::InitializeLog") 2>&1 & # listen on 8085 for REST API
API_PID=$!

# On docker stop, pass the signal to gameapi, which finishes the requests in progress
STOPPING=""
trap 'STOPPING=1; kill -TERM "$API_PID" 2>/dev/null' TERM INT

cd "$(dirname "$0")"/frontend
python3 appserver.py --host 0.0.0.0 &  # listen on 8080 for browser

# Wait for any process to exit
wait -n
STATUS=$?

if [ -n "$STOPPING" ]; then
    # The signal interrupted the wait; give gameapi the time it needs to finish
    while kill -0 "$API_PID" 2>/dev/null; do
        wait "$API_PID" 2>/dev/null
    done
    exit 0
fi

# Exit with status of process that exited first
exit $STATUS