#!/bin/bash
set -e

# Configuration
VENV_DIR=".venv"
PYTHON_VERSION="3.12"
REQUIREMENTS_FILE="requirements.txt"

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Qwen3-VL Training Pipeline...${NC}"

# 1. Environment Setup
echo -e "${GREEN}[1/4] Setting up environment...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    uv venv --python $PYTHON_VERSION
else
    echo "Virtual environment exists."
fi

source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
uv pip install -r $REQUIREMENTS_FILE

# 2. Data Preparation
echo -e "${GREEN}[2/4] preparing dataset...${NC}"
python download_data.py

# 3. Training
echo -e "${GREEN}[3/4] Starting training...${NC}"

# Check for wandb API key
if [ -z "${WANDB_API_KEY}" ]; then
    echo "Error: WANDB_API_KEY environment variable is not set."
    echo "Please set it with: export WANDB_API_KEY=<your_api_key>"
    exit 1
fi

# Check if GPU is available (basic check)
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. Ensure you have GPUs allocated."
else
    nvidia-smi
fi

# We assume resources are allocated (e.g., via srun) as per user instruction.
echo "Running training script using torchrun..."
NUM_GPUS=1
torchrun --nproc_per_node=$NUM_GPUS train.py

# 4. Inference
echo -e "${GREEN}[4/4] Running inference...${NC}"
python inference.py

echo -e "${GREEN}Pipeline completed successfully!${NC}"
