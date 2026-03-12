# ImageCLEF 2026 - Multimodal Reasoning

This repository provides baseline implementations and supporting materials for the ImageCLEF 2026 Multimodal Reasoning competition, including run scripts, evaluation utilities, and example files to help participants get started. The competition evaluates multimodal models on challenging vision-based exam problems through two distinct tasks:

1. **Multiple-Choice Question Answering (MCQ)** – Classification
2. **Open Question Answering (OpenQA)** – Generative

The baselines use Vision-Language Models (VLMs) in zero-shot or few-shot settings.

# 🏆 Competition Tasks

## 1️⃣ Multiple-Choice Question Answering (MCQ)

**Task Type:** Classification

### 📘 Task Overview

Given an image of a question containing **three to five answer options**, the system must:

- Identify the question and answer options from the image.
- Understand relevant visual content (graphs, charts, tables, diagrams, etc.).
- Select the **single correct answer choice**.

The model must choose exactly one option from the predefined choices.

---

### 📄 MCQ Submission Format

The submission file **MUST** follow this JSON format:

- `id`: Unique identifier (matching a sample from the Test set)
- `prediction`: Predicted answer label — one of `"A"`, `"B"`, `"C"`, `"D"`, or `"E"`

#### 🔒 Rules

- Submission size **MUST match** the Test set size.
- No duplicate IDs.
- `prediction` must be **EXACTLY ONE** of `"A"`, `"B"`, `"C"`, `"D"`, or `"E"`.
- File must be valid JSON.

#### ✅ Example (MCQ)

```json
[
  {
    "id": "5e9sf6b9-3338-4e97-ba6b-762e24a07e69",
    "prediction": "A"
  },
  {
    "id": "08fjguy8-4e97-12s4-bt65-385f09dsk5df",
    "prediction": "C"
  }
]
```

## 2️⃣ Open Question Answering (OpenQA)

**Task Type:** Generative

### 📘 Task Overview

Given an image of a question without predefined answer options, the system must:

- Extract and understand the question from the image.
- Reason over both textual and visual content.
- Generate a free-form textual answer.

Unlike MCQ, there are no fixed answer choices — the model must generate the correct response.

### 📄 OpenQA Submission Format

The submission file MUST follow this JSON format:

- `id`: Unique identifier (matching a sample from the Test set)
- `prediction`: Generated textual answer
- `language`: Question language

### 🔒 Rules

- Submission size MUST match the Test set size.
- No duplicate IDs.
- The `prediction` field must contain only the generated answer (no explanations unless explicitly allowed in official guidelines).
- File must be valid JSON.

### ✅ Example (OpenQA)

```json
[
  {
    "id": "3ac9d21e-1ab3-4f21-92fa-1f2390abc123",
    "prediction": "Photosynthesis",
    "language": "English"
  },
  {
    "id": "9fd21c44-77d2-4cdd-81d3-812fbc991111",
    "prediction": "42",
    "language": "English"
  }
]
```

## Evaluation

This repository provides separate evaluation scripts for **MCQ** and **OpenQA**.

### 1) MCQ Evaluation (Accuracy)

Use `src/evaluation/evaluate_mcq.py` to compute accuracy:

```bash
python src/evaluation/evaluate_mcq.py \
  --pred_file ./pred.json \
  --gold_file ./gold.json \
  --print_score True
```

**Expected fields in both files:**

- `id`
- `answer_key` (must be one of `A`, `B`, `C`, `D`, `E`)

**What it checks before scoring:**

- Prediction size matches gold size
- IDs in prediction exist in gold
- Answer keys are valid

### 2) OpenQA Evaluation (Automatic Metrics)

Use `src/evaluation/evaluate_qa.py` to compute text-generation metrics:

```bash
python src/evaluation/evaluate_qa.py \
  --pred_file ./pred.json \
  --gold_file ./gold.json
```

Optional COMET batch size:

````bash
python src/evaluation/evaluate_qa.py \
  --pred_file ./pred.json \
  --gold_file ./gold.json \
  --batch_size_comet 64

Optional output path:

```bash
python src/evaluation/evaluate_qa.py \
  --pred_file ./pred.json \
  --gold_file ./gold.json \
  --out_file ./metrics_summary.json
````

```

**Expected fields:**

- Gold file: `id`, `question`, `answer` (`image_id` is optional)
- Prediction file: `id`, `prediction`

The script computes and reports task-level averages for:

- `bleu_scores` (`bleu-1` to `bleu-4`) and `bleu_avg`
- `rouge_scores` (`rouge-1`, `rouge-2`, `rouge-l`)
- `meteor`
- `comet`

Output is printed to stdout and saved as a single summary JSON (default path):

- `2026/src/evaluation/automatic_metrics/metrics.json`

---

## 📁 File Structure

```

ImageCLEF-MultimodalReasoning-2026/
├── README.md
├── requirements.txt
├── run.sh
└── src/
└── evaluation/
├── evaluate_mcq.py
├── evaluate_qa.py
├── example_maths_english.json
└── automatic_metrics/
└── metrics.json

```

## 📌 Official Resources

For complete task descriptions, datasets, evaluation scripts, and submission guidelines, refer to the official task website:

👉 https://mbzuai-nlp.github.io/ImageCLEF-MultimodalReasoning/2026/
```
