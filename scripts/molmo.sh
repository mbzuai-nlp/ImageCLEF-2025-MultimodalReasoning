#!/bin/bash

set -e  # Exit immediately on error
set -o pipefail

export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH="models/molmo"
SCRIPT_PATH="baselines/molmo.py"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Start vLLM server in the background 
echo "Starting vLLM server..."
vllm serve "$MODEL_PATH" \
   --tensor-parallel-size 1 \
   --dtype auto \
   --disable-log-requests \
   --host 0.0.0.0 \
   --trust-remote-code \
   --max-model-len 2048 \
   --max-seq-len-to-capture 2048 \
   --max-num-seqs 50 \
   --gpu-memory-utilization 0.8 \
   --port 8022 > "$LOG_DIR/vllm_server.log" 2>&1 &

# Get PID to kill later if needed
VLLM_PID=$!

# Ensure the server gets killed on script exit or error
trap "kill $VLLM_PID" EXIT

# Health check loop
echo "Waiting for vLLM server to be ready..."
sleep 10  # Initial wait

until curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8022/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_PATH"'",
        "messages": [
			{
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": "Describe this image in one sentence."
					},
					{
						"type": "image_url",
						"image_url": {
							"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
						}
					}
				]
			}
		]
    }' | grep -q "200"; do
    echo "Server not ready, retrying in 10 seconds..."
    sleep 10
done

echo "vLLM server is ready."

# Run inference script
echo "Running Python script..."
python "$SCRIPT_PATH" > "$LOG_DIR/result_molmo_log.txt" 2>&1
echo "Inference complete. Logs saved to $LOG_DIR"
