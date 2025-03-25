#!/bin/bash

set -e  # Exit immediately on error
set -o pipefail

export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH="models/olmo"
SCRIPT_PATH="baselines/olmo.py"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Start vLLM server in the background
echo "Starting vLLM server..."
vllm serve "$MODEL_PATH" \
   --chat-template "scripts/chat_template.jinja" \
   --tensor-parallel-size 1 \
   --dtype auto \
   --disable-log-requests \
   --host 0.0.0.0 \
   --trust-remote-code \
   --max-model-len 4096 \
   --max-seq-len-to-capture 4096 \
   --max-num-seqs 90 \
   --gpu-memory-utilization 0.8 \
   --port 8018 > "$LOG_DIR/vllm_server.log" 2>&1 &

# Get PID to kill later if needed
VLLM_PID=$!

# Ensure the server gets killed on script exit or error
trap "kill $VLLM_PID" EXIT

# Health check loop
echo "Waiting for vLLM server to be ready..."
sleep 10  # Initial wait

until curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8018/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_PATH"'",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Are you up?"}]}
        ]
    }' | grep -q "200"; do
    echo "Server not ready, retrying in 10 seconds..."
    sleep 10
done

echo "vLLM server is ready."

# Run inference script
echo "Running Python script..."
python "$SCRIPT_PATH" > "$LOG_DIR/result_olmo_log.txt" 2>&1
echo "Inference complete. Logs saved to $LOG_DIR"