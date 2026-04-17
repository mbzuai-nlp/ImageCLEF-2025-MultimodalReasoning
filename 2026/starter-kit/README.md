# Multimodal Reasoning Pipelines

This repository contains two end-to-end shell pipelines for multimodal reasoning experiments with Qwen-based models:

- `scripts/mcq_pipeline.sh` runs the multiple-choice workflow on `MBZUAI/EXAMS-V`.
- `scripts/openqa_pipeline.sh` runs the OpenQA workflow on `SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual`.

Both pipelines set up isolated environments, download datasets and model artifacts, run inference with SGLang, fine-tune with Unsloth, and save predictions and model outputs locally.

## Design Choices

### Unsloth

The training scripts use [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning. This keeps the training code lightweight while still supporting vision-language SFT workflows.

### Weights & Biases

Training runs report metrics to [Weights & Biases](https://wandb.ai/). Both pipeline scripts check that `WANDB_API_KEY` is set before launching training.

## Prerequisites

- NVIDIA GPU with CUDA support.
- `uv` installed for environment management and package installation.
- A valid Wandb API key exported before training:

```bash
export WANDB_API_KEY=your_key_here
```

Both scripts currently launch training with `torchrun --nproc_per_node=4`, so they assume a 4-GPU setup unless you edit that value.

## MCQ Pipeline

Run:

```bash
bash scripts/mcq_pipeline.sh
```

What `scripts/mcq_pipeline.sh` does:

1. Creates `.venv-train` and `.venv-sglang` with Python 3.11 and installs the MCQ training and SGLang requirements.
2. Downloads the base model `unsloth/Qwen3.5-9B-GGUF` into `unsloth_models/Qwen3.5-9B-GGUF`.
3. Downloads the `MBZUAI/EXAMS-V` dataset into `datasets/EXAMS-V`.
4. Starts an SGLang server for the base model and runs pre-training inference with `mcq_visual/inference.py` on the validation split.
5. Generates the MCQ gold file with `evaluation/generate_mcq_gold.py` and evaluates the baseline predictions with `evaluation/evaluate_mcq.py`.
6. Fine-tunes the model with `mcq_visual/train.py`.
7. Merges the LoRA adapter into a full checkpoint with `mcq_visual/merge_adapter.py` so SGLang can serve the full finetuned model.
8. Starts SGLang again with the merged checkpoint, reruns inference, and evaluates the post-training predictions.

In practice, this script gives you a before-vs-after MCQ evaluation loop, with artifacts written under `predictions/`, `logs/`, and `unsloth_models/`.

## OpenQA Pipeline

Run:

```bash
bash scripts/openqa_pipeline.sh
```

What `scripts/openqa_pipeline.sh` does:

1. Creates `.venv-train` and `.venv-sglang` with Python 3.11 and installs the OpenQA training and SGLang requirements.
2. Downloads the base model `unsloth/Qwen3-VL-8B-Instruct` into `unsloth_models/Qwen3-VL-8B-Instruct`.
3. Downloads the `SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual` dataset into `datasets/ImageCLEF-MR2026-OpenQA-Visual`.
4. Starts SGLang with the question-extraction model at `unsloth_models/Qwen3.5-35B-A3B` and runs `openqa/extract_questions.py` to build `extracted_questions.jsonl` for the `train` and `dev` splits.
5. Starts SGLang with the base OpenQA model and runs `openqa/inference.py` on the `dev` split to produce pre-training predictions.
6. Fine-tunes the model with `openqa/train.py` using the downloaded dataset and extracted question file.
7. Merges the LoRA adapter into a full checkpoint with `openqa/merge_adapter.py`.
8. Starts SGLang with the merged checkpoint and reruns `openqa/inference.py` on `dev` to produce post-training predictions.

This pipeline is centered around question extraction plus answer generation. Unlike the MCQ script, it currently stops after producing the pre-training and post-training prediction files; it does not call `evaluation/evaluate_qa.py` inside the shell script.

## Switching OpenQA from Visual to Textual

To switch `scripts/openqa_pipeline.sh` from the visual dataset to the textual dataset, update the configuration block at the top of the script:

```bash
DATASET_NAME="SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Textual"
DATASET_FILEPATH="$PROJECT_ROOT/datasets/ImageCLEF-MR2026-OpenQA-Textual"
```

That is usually enough for the main dataset swap, because `QUESTION_FILEPATH` is derived from `DATASET_FILEPATH`:

```bash
QUESTION_FILEPATH="$DATASET_FILEPATH/extracted_questions.jsonl"
```

If you also want the saved artifacts to stay clearly separated from the visual run, it is a good idea to rename these variables from `visual` to `textual` as well:

- `SFT_MODEL_FILEPATH`
- `MERGED_MODEL_FILEPATH`
- `PRETRAIN_PREDICTIONS_FILE`
- `POSTTRAIN_PREDICTIONS_FILE`

With those small changes, the same OpenQA pipeline can be reused for `SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Textual` without changing the training or inference logic.
