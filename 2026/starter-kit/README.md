# Multimodal Reasoning with Qwen3-VL

This project implements a fine-tuning pipeline for the Qwen3-VL-4B-Instruct model using the EXAMS-V dataset.

## Design Choices

### 🚀 Unsloth (Faster Training)
We use [Unsloth](https://github.com/unslothai/unsloth) for fine-tuning. Unsloth provides optimized kernels for LLM fine-tuning, significantly reducing memory usage and increasing training speed (up to 2x faster) compared to standard Hugging Face implementations. It specifically supports `FastVisionModel` and `SFTTrainer` for efficient vision-language model training.

### 📊 Weights & Biases (Logging)
Training metrics, including loss and accuracy, are logged to [Weights & Biases (Wandb)](https://wandb.ai/). logging to Wandb allows for real-time monitoring of experiments and easy comparison across different runs.

## Dataset

This project uses the **MBZUAI/EXAMS-V** dataset from Hugging Face.
- **Train Split**: Used for fine-tuning the model.
- **Validation Split**: Used for evaluating model performance (accuracy) during training.

The dataset contains multiple-choice questions based on images, requiring multimodal reasoning capabilities.

## Execution Instructions

The training pipeline is automated via a shell script.

### Prerequisites
- NVIDIA GPU with CUDA support.
- `uv` installed for fast package management.

### Running the Pipeline
Simply execute the `run_pipeline.sh` script:

```bash
bash run_pipeline.sh
```

### What the script does:
1.  **Environment Setup**: Creates a Python 3.12 virtual environment using `uv` and installs dependencies.
2.  **Data Preparation**: Downloads the necessary datasets.
3.  **Training**: 
    - Checks for the `WANDB_API_KEY` environment variable.
    - Launches the `train.py` script.
    - Model checkpoints and LoRA adapters are saved in `./outputs` and `lora_model`.

> [!IMPORTANT]
> You must set your Wandb API key before running the pipeline:
> `export WANDB_API_KEY=your_key_here`
