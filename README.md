# ImageCLEF 2025 - Multimodal Reasoning Baseline

This repository provides baseline implementations for the **ImageCLEF 2025 Multimodal Reasoning - Visual Question Answering (VQA)** task. The baselines use vision-language models (VLMs) and large language models (LLMs) to solve image-based multiple-choice questions in a **zero-shot** setting.

---

## 📘 Task Overview

Given an image of a multiple-choice question (MCQ), the task is to:

- Identify the question and all answer options from the image or extracted caption.
- Understand any relevant visual content (e.g., graphs, tables).
- Predict the correct answer **only based on the provided image or caption**.

---

## Prediction format

The submission file **MUST** have the following format:

- `id`: Unique Identifier (matching to a sample from the Test set). 
- `answer_key`: Ground truth label (one of "A", "B", "C", "D", or "E")
- `language`: Question language  

Additional formatting rules:
- Submission **MUST** be the same size as the Test set. *For single-language submissions, we expect the size to match the respective test data for that language.*
- Submission **MUST NOT** contain duplicates (There will be an evaluation Error!)
- `answer_key` must be **EXACTLY ONE** of "A", "B", "C", "D", or "E". 

Correct submission file example:

```
[
  {
      "id": "5e9sf6b9-3338-4e97-ba6b-762e24a07e69",
      "answer_key": "A",
      "language": "English"
  },
  {
      "id": "08fjguy8-4e97-12s4-bt65-385f09dsk5df",
      "answer_key": "C",
      "language": "English"
  },
  ...
]

```

## Evaluation

The evaluation metric for the task is **accuracy**: *correct* / *total_questions*.

We provide an evaluation script, that you can use locally, located in `evaluation/evaluate.py`. 

*Example usage:*

```
python evaluate.py --pred_file="./pred.json" --gold_file="./gold.json" --print_score="True"
```

---

## 🏗️ Baseline Models

We provide two types of baselines:

### Vision-Language Models (VLM)

Use the image directly for reasoning in a zero-shot setting.

#### 1. `molmo.py`

- Model path: `models/molmo`
- Script: `baselines/molmo.py`

#### 2. `smolvlm.py`

- Model path: `models/smolvlm`
- Script: `baselines/smolvlm.py`

### Large Language Models with Captions (LLM)

Use precomputed image captions for reasoning.

#### 3. `olmo.py`

- Model path: `models/olmo`
- Captions: `captions/Llama-3.2-11B-Vision/` or `captions/SmolVLM/`
- Script: `baselines/olmo.py`

#### 4. `smollm.py`

- Model path: `models/smollm`
- Captions: `captions/Llama-3.2-11B-Vision/` or `captions/SmolVLM/`
- Script: `baselines/smollm.py`

All models are evaluated in a **zero-shot** setting with no fine-tuning.

---

## Prompts

Each baseline uses a specific zero-shot prompt to guide reasoning:

- **Prompt 1**: A short, direct instruction for selecting the correct answer based on image or caption content only.
- **Prompt 2**: A step-by-step reasoning prompt encouraging deeper analysis of textual and visual cues (including multilingual content).

### Vision-Language Models (VLM)

These models use the image as input. Examples include `molmo.py` and `smolvlm.py`.

**Prompt 1**:

> Analyze the image of a multiple-choice question. Identify the question, all answer options (even if there are more than four), and any relevant visuals like graphs or tables. Choose the correct answer based only on the image. Reply with just the letter of the correct option, no explanation.

**Prompt 2**:

> You are a sophisticated Vision-Language Model (VLM) capable of analyzing images containing multiple-choice questions, regardless of language. To guide your analysis, you may adopt the following process:
>
> 1. Examine the image carefully for all textual and visual information.
> 2. Identify the question text, even if it's in a different language.
> 3. Extract all answer options (note: there may be more than four).
> 4. Look for additional visual elements such as tables, diagrams, charts, or graphs.
> 5. Ensure to consider any multilingual content present in the image.
> 6. Analyze the complete context and data provided.
> 7. Select the correct answer(s) based solely on your analysis.
> 8. Respond by outputting only the corresponding letter(s) without any extra explanation.

To query a Vision-Language Model (VLM) with an image, follow these steps:

1. **Convert the image to base64 format**:

   - Open the image file (e.g., `.png`) in binary mode.
   - Encode the binary data using `base64.b64encode(...)`.
   - Prefix the encoded string with `data:image/png;base64,` to make it web-compatible.

2. **Format the input as an OpenAI-compatible chat message**:

```python
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "<insert_prompt_text_here (prompt1 or prompt 2)>"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,<base64_encoded_image>"
      }
    }
  ]
}

```

### Language-Only Models (LLM)

These models use precomputed captions as input. Examples include `olmo.py` and `smollm.py`.
**Prompt 1**:

> You are given a multiple-choice question extracted from an exam. The question is: `{caption}` Identify the question and all answer options (even if there are more than four), and any relevant data related to graphs or tables. Choose the correct answer and reply with just the letter of the correct option, no explanation.

**Prompt 2**:

> You are given a multiple-choice question extracted from an exam.  
> The question is: `{caption}`  
> Please follow the steps below to determine the correct answer:
>
> 1. Carefully read and interpret the full question text.
> 2. Identify the main question, even if it is in a different language.
> 3. Extract all available answer options (note: there may be more than four).
> 4. Pay attention to any references to data, including tables, diagrams, charts, or graphs mentioned in the text.
> 5. Take into account any multilingual elements present in the question.
> 6. Analyze all information in context, both textual and inferred data.
> 7. Select the correct answer based solely on your analysis.
> 8. Respond by outputting only the letter(s) of the correct answer option, with no additional explanation.

**Format the input as an OpenAI-compatible chat message**:

```python
{
  "role": "user",
  "content": prompt_text
}
```

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
    "language": "English",
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
