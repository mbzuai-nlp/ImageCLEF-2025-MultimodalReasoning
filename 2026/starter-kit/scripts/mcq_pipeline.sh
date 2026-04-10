#!/bin/bash
set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_VENV_DIR="$PROJECT_ROOT/.venv-train"
SGLANG_VENV_DIR="$PROJECT_ROOT/.venv-sglang"

PYTHON_VERSION="3.11"
MODEL_ID="unsloth/Qwen3.5-9B-GGUF"
DATASET_NAME="MBZUAI/EXAMS-V"

TRAIN_REQUIREMENTS_FILE="$PROJECT_ROOT/mcq_visual/requirements-train.txt"
SGLANG_REQUIREMENTS_FILE="$PROJECT_ROOT/mcq_visual/requirements-sglang.txt"
DATASET_FILEPATH="$PROJECT_ROOT/datasets/EXAMS-V"
HF_MODEL_FILEPATH="$PROJECT_ROOT/unsloth_models/Qwen3.5-9B-GGUF"
SFT_MODEL_FILEPATH="$PROJECT_ROOT/unsloth_models/Qwen3.5-9B-GGUF-mcq-Visual-SFT"
MERGED_MODEL_FILEPATH="$PROJECT_ROOT/unsloth_models/Qwen3.5-9B-GGUF-mcq-Visual-SFT-merged"
PREDICTIONS_DIR="$PROJECT_ROOT/predictions"
LOG_DIR="$PROJECT_ROOT/logs"
GROUNDTRUTH_FILE="mcq_visual_groundtruth.json"
PRETRAIN_PREDICTIONS_FILE="mcq_visual_pretrain_predictions.json"
POSTTRAIN_PREDICTIONS_FILE="mcq_visual_posttrain_predictions.json"
MCQ_EVAL_SPLIT="validation"

mkdir -p "$PREDICTIONS_DIR" "$LOG_DIR"

source "$PROJECT_ROOT/scripts/sglang_utils.sh"

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Qwen3-VL Training Pipeline...${NC}"

trap cleanup_sglang_server EXIT

# 1. Environment Setup
echo -e "${GREEN}[1/6] Setting up environment...${NC}"
if [ ! -d "$TRAIN_VENV_DIR" ]; then
    echo "Creating training virtual environment..."
    uv venv --python "$PYTHON_VERSION" "$TRAIN_VENV_DIR"
else
    echo "Training virtual environment exists."
fi

if [ ! -d "$SGLANG_VENV_DIR" ]; then
    echo "Creating SGLang virtual environment..."
    uv venv --python "$PYTHON_VERSION" "$SGLANG_VENV_DIR"
else
    echo "SGLang virtual environment exists."
fi

TRAIN_PYTHON="$TRAIN_VENV_DIR/bin/python"
TRAIN_TORCHRUN="$TRAIN_VENV_DIR/bin/torchrun"
SGLANG_PYTHON="$SGLANG_VENV_DIR/bin/python"

echo "Installing training dependencies into $TRAIN_VENV_DIR ..."
source $TRAIN_VENV_DIR/bin/activate
uv pip install -r "$TRAIN_REQUIREMENTS_FILE"
# uv pip install "flash-attn" --no-build-isolation
 
echo "Installing SGLang dependencies into $SGLANG_VENV_DIR ..."
source $SGLANG_VENV_DIR/bin/activate
uv pip install -r "$SGLANG_REQUIREMENTS_FILE"
uv pip install "nvidia-cudnn-cu12==9.16.0.29"

# 2. Model Download
echo -e "${GREEN}[2/6] Downloading Qwen3-VL-8B-Instruct to ../models...${NC}"
mkdir -p "$(dirname "$HF_MODEL_FILEPATH")"
"$TRAIN_VENV_DIR/bin/hf" download "$MODEL_ID" --local-dir "$HF_MODEL_FILEPATH"

# 3. Data Preparation
echo -e "${GREEN}[3/6] Preparing dataset...${NC}"
mkdir -p "$(dirname "$DATASET_FILEPATH")"
"$TRAIN_VENV_DIR/bin/hf" download --repo-type dataset "$DATASET_NAME" --local-dir "$DATASET_FILEPATH"

# 4. Inference Before Training
echo -e "${GREEN}[4/6] Running inference before training...${NC}"
source $SGLANG_VENV_DIR/bin/activate

start_sglang_server \
    "$HF_MODEL_FILEPATH" \
    "$SGLANG_MODEL_NAME" \
    "$LOG_DIR/sglang-pretrain.log"

python3 mcq_visual/inference.py \
    --model-id-or-path "$SGLANG_MODEL_NAME" \
    --dataset "$DATASET_FILEPATH" \
    --api-base "$SGLANG_API_BASE" \
    --max-image-long-side 2048 \
    --max-image-pixels 2000000 \
    --jpeg-quality 90 \
    --output-dir "$PREDICTIONS_DIR" \
    --output-name "$PRETRAIN_PREDICTIONS_FILE"

echo -e "${GREEN}Generating pre-train MCQ gold file...${NC}"
python3 "$PROJECT_ROOT/evaluation/generate_mcq_gold.py" \
        --dataset "$DATASET_FILEPATH" \
        --split "$MCQ_EVAL_SPLIT" \
        --output-file "$PREDICTIONS_DIR/$GROUNDTRUTH_FILE"

echo -e "${GREEN}Pre-train MCQ evaluation:${NC}"
python3 evaluation/evaluate_mcq.py \
    --pred_file "$PREDICTIONS_DIR/$PRETRAIN_PREDICTIONS_FILE" \
    --gold_file "$PREDICTIONS_DIR/$GROUNDTRUTH_FILE" \
    --print_score True

cleanup_sglang_server

# 5. Training
echo -e "${GREEN}[5/6] Starting training...${NC}"
source $TRAIN_VENV_DIR/bin/activate

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
echo "Running training script using torchrun from the training environment..."
NUM_GPUS=4
"$TRAIN_TORCHRUN" --nproc_per_node=$NUM_GPUS mcq_visual/train.py \
    --model-id-or-path "$HF_MODEL_FILEPATH" \
    --output-model-path "$SFT_MODEL_FILEPATH"

# 6. Inference After Training
echo -e "${GREEN}[6/6] Running inference after training...${NC}"
source $TRAIN_VENV_DIR/bin/activate

# SGLang's current Qwen3-VL LoRA path only supports text-layer LoRA modules,
# while the training adapter also contains visual-layer deltas. Merge first so
# inference uses the full finetuned checkpoint instead of a partially applied LoRA.
if [ ! -f "$MERGED_MODEL_FILEPATH/config.json" ] || \
   [ "$SFT_MODEL_FILEPATH/adapter_model.safetensors" -nt "$MERGED_MODEL_FILEPATH/config.json" ] || \
   [ "$SFT_MODEL_FILEPATH/adapter_config.json" -nt "$MERGED_MODEL_FILEPATH/config.json" ]; then
    echo "Merging adapter into base model for SGLang-compatible inference..."
    python3 mcq_visual/merge_adapter.py \
        --base-model-path "$HF_MODEL_FILEPATH" \
        --adapter-model-path "$SFT_MODEL_FILEPATH" \
        --output-model-path "$MERGED_MODEL_FILEPATH"
else
    echo "Merged model is up to date."
fi

source $SGLANG_VENV_DIR/bin/activate

start_sglang_server \
    "$MERGED_MODEL_FILEPATH" \
    "$SGLANG_MODEL_NAME" \
    "$LOG_DIR/sglang-posttrain.log"

python3 mcq_visual/inference.py \
    --model-id-or-path "$SGLANG_MODEL_NAME" \
    --dataset "$DATASET_FILEPATH" \
    --api-base "$SGLANG_API_BASE" \
    --output-dir "$PREDICTIONS_DIR" \
    --output-name "$POSTTRAIN_PREDICTIONS_FILE"

echo -e "${GREEN}Post-train MCQ evaluation:${NC}"
python3 evaluation/evaluate_mcq.py \
    --pred_file "$PREDICTIONS_DIR/$POSTTRAIN_PREDICTIONS_FILE" \
    --gold_file "$PREDICTIONS_DIR/$GROUNDTRUTH_FILE" \
    --print_score True

cleanup_sglang_server

echo -e "${GREEN}Pipeline completed successfully!${NC}"
