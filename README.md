# ImageCLEF 2025 - Multimodal Reasoning Baseline

This repository provides baseline implementations for the **ImageCLEF 2025 Multimodal Reasoning - Visual Question Answering (VQA)** task. The baselines use vision-language models (VLMs) and large language models (LLMs) to solve image-based multiple-choice questions in a **zero-shot** setting.

---

## 📘 Task Overview

Given an image of a multiple-choice question (MCQ), the task is to:

- Identify the question and all answer options from the image or extracted caption.
- Understand any relevant visual content (e.g., graphs, tables).
- Predict the correct answer **only based on the provided image or caption**.

Each sample consists of:

- `id`: Unique image ID
- `image`: Corresponding PNG image file
- `answer_key`: Ground truth label (e.g., "A")
- `language`: Image language metadata

---

## 🏗️ Baseline Models

We provide two types of baselines:

### Vision-Language Models (VLM)

Use the image directly for reasoning in a zero-shot setting.

#### 1. `molmo.py`

- Model path: `models/molmo`
- Script: `baselines/molmo.py`
- Prompt: direct prompt + image

#### 2. `smolvlm.py`

- Model path: `models/smolvlm`
- Script: `baselines/smolvlm.py`
- Prompt: Same as above

### Large Language Models with Captions (LLM)

Use precomputed image captions for reasoning.

#### 3. `olmo.py`

- Model path: `models/olmo2-1124-7b-instruct`
- Captions: `captions/Llama-3.2-11B-Vision/` or `captions/SmolVLM/`
- Script: `baselines/olmo.py`

#### 4. `smollm.py`

- Model path: `models/smollm`
- Captions: `captions/Llama-3.2-11B-Vision/` or `captions/SmolVLM/`
- Script: `baselines/smollm.py`
- Server port: `8018`

All models are evaluated in a **zero-shot** setting with no fine-tuning.

---

## 📁 File Structure

```
ImageCLEF-2025-MultimodalReasoning/
├── baselines/
│   ├── molmo.py       # VLM (image input)
│   ├── smolvlm.py     # VLM (image input)
│   ├── olmo.py        # LLM (caption input)
│   └── smollm.py      # LLM (caption input)
├── scripts/
│   ├── molmo.sh       # Launch molmo baseline
│   ├── smolvlm.sh     # Launch smolvlm baseline
│   ├── olmo.sh        # Launch olmo baseline
│   └── smollm.sh      # Launch smollm baseline
├── captions/
│   ├── Llama-3.2-11B-Vision/   # Precomputed captions for olmo.py
│   └── SmolVLM/                # Precomputed captions for smollm.py
├── data/
│   ├── images/                 # MCQ images (.png)
│   └── metadata_labeled.json  # Ground truth JSON
├── models/                    # Downloaded model folders
├── logs/                      # All log and result outputs
└── run.sh                     # Entry point for selected baseline
```

---

## 🚀 How to Run

### Setup

Install required Python dependencies:

```bash
pip install -r requirements.txt
```

Ensure the `models/` folder contains your downloaded vLLM-compatible models.

#### Vision-Language Models (VLM)

- [`Molmo`](https://huggingface.co/allenai/Molmo-7B-O-0924) → place in `models/molmo`
- [`SmolVLM`](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) → place in `models/smolvlm`

#### Language-Only Models (LLM with captions)

- [`OLMo`](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct) → place in `models/olmo`
- [`SmolLM`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) → place in `models/smollm`

Unzip the caption files in `captions/`:

```bash
unzip captions/Llama-3.2-11B-Vision.zip -d captions/
unzip captions/SmolVLM.zip -d captions/
```

### Run Baselines

#### Run VLM Baseline (e.g., molmo)

```bash
bash scripts/molmo.sh
```

#### Run LLM Baseline (e.g., olmo)

```bash
bash scripts/olmo.sh
```

You can also run `run.sh` to default to a specific script (e.g., olmo).

---

## 📊 Output Format

Each evaluation script produces a JSON file:

```json
[
  {
    "id": "image_001",
    "language": "en",
    "answer_key": "C"
  },
  ...
]
```

The final accuracy is printed and logged in `logs/result_<model>_log.txt`.

---

## 📝 Notes

- The `OpenAI` interface is used for vLLM-compatible chat API calls.
- All predictions use a **guided choice** mechanism: ["A", "B", "C", "D", "E"]
- Prompt 1 is enabled by default (short reasoning prompt). You may uncomment Prompt 2 for more detailed chain-of-thought-style prompting.
- Captions are extracted from vision models and serve as proxies for visual content in LLM pipelines.
