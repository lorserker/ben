#!/bin/bash

# this is all in one wrapper script mainly for container
# Suppress TensorFlow CUDA/XLA warnings (no GPU in container)
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

python3 gameserver.py 2>&1 | grep -v "cuda\|cuDNN\|cuBLAS\|cuFFT\|cuInit\|CUDA\|absl::InitializeLog" & # listen on 4443 for websocket

# gameapi.py is started so that $! is PYTHON's own process id - with `python ... | grep &`
# it would be grep's - because the stop signal has to be handed to it by name (below).
python3 gameapi.py --host 0.0.0.0 > >(grep -v "cuda\|cuDNN\|cuBLAS\|cuFFT\|cuInit\|CUDA\|absl::InitializeLog") 2>&1 & # listen on 8085 for REST API
API_PID=$!

# THE STOP SIGNAL IS PASSED ON TO gameapi.py, and this script then waits for it to leave.
# gameapi.py finishes the requests it is in the middle of before it goes (see A STOP THAT
# LETS THE REQUESTS IN FLIGHT FINISH there). Without this the script, as the container's
# first process, took the signal itself and the API was simply killed some seconds later.
STOPPING=""
trap 'STOPPING=1; kill -TERM "$API_PID" 2>/dev/null' TERM INT

cd "$(dirname "$0")"/frontend
python3 appserver.py --host 0.0.0.0 &  # listen on 8080 for browser

# Wait for any process to exit
wait -n
STATUS=$?

# A stop signal interrupts the wait above; give the API the time it asked for.
if [ -n "$STOPPING" ]; then
    while kill -0 "$API_PID" 2>/dev/null; do
        wait "$API_PID" 2>/dev/null
    done
    exit 0
fi

# Exit with status of process that exited first
exit $STATUS