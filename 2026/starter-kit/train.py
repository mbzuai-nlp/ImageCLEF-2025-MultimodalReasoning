import numpy as np
import re
from unsloth import FastVisionModel, is_bfloat16_supported, UnslothVisionDataCollator
from unsloth.trainer import SFTTrainer
from transformers import TrainingArguments, EvalPrediction
from datasets import load_dataset, concatenate_datasets
from datetime import datetime

# Function to format the dataset for Qwen-VL (Batched version)
def format_data(examples):
    question = (
        "You are a sophisticated Vision-Language Model (VLM) capable of analyzing images containing multiple-choice questions, regardless of language. To guide your analysis, you may adopt the following process:",
        "1. Examine the image carefully for all textual and visual information.",
        "2. Identify the question text, even if it's in a different language.",
        "3. Extract all answer options (note: there may be more than four).",
        "4. Look for additional visual elements such as tables, diagrams, charts, or graphs.",
        "5. Ensure to consider any multilingual content present in the image.",
        "6. Analyze the complete context and data provided.",
        "7. Select the correct answer(s) based solely on your analysis.",
        "8. Respond by outputting only the corresponding letter(s) without any extra explanation."
    )
    
    questions = ["\n".join(question)] * len(examples['image'])
    
    all_messages = []
    for image, answer, question in zip(examples['image'], examples['answer_key'], questions):
        answer_text = f"The answer is {str(answer).strip()}."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer_text}
                ]
            }
        ]
        all_messages.append(conversation)
    return {"messages": all_messages}


_ANSWER_RE = re.compile(r"the answer is\s*([A-Za-z0-9]+)", re.IGNORECASE)


def _extract_answer(text: str):
    match = _ANSWER_RE.search(text)
    if not match:
        return None
    return match.group(1).strip().upper()


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred: EvalPrediction):
        predictions, labels = eval_pred
        # In eval, predictions are preprocessed token IDs (argmax over vocab).
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        total, correct = 0, 0
        for pred_ids, label_ids in zip(predictions, labels):
            label_mask = label_ids != -100
            if not np.any(label_mask):
                continue

            pred_text = tokenizer.decode(pred_ids[label_mask], skip_special_tokens=True)
            label_text = tokenizer.decode(label_ids[label_mask], skip_special_tokens=True)

            pred_answer = _extract_answer(pred_text)
            gt_answer = _extract_answer(label_text)
            if gt_answer is None:
                continue

            total += 1
            if pred_answer == gt_answer:
                correct += 1

        return {"accuracy": float(correct / total) if total > 0 else 0.0}

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    # Keep only token IDs so Trainer does not gather full-vocab logits in eval.
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def main():
    print("Loading model...")
    model_id = "unsloth/Qwen3-VL-8B-Instruct" # Using Qwen3 as requested
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        use_gradient_checkpointing = "unsloth",
    )
    
    # Configure LoRA
    model = FastVisionModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None
    )

    print("Loading dataset...")
    import multiprocessing
    num_proc = min(multiprocessing.cpu_count(), 32)

    # Load and format train split
    full_train_dataset = load_dataset("MBZUAI/EXAMS-V", split="train")
    
    # Sample 64 records per language — single pass over the dataset
    # Build lang -> [indices] map without repeated full-dataset .filter() calls
    from collections import defaultdict
    import random
    rng = random.Random(3407)

    lang_to_indices = defaultdict(list)
    for idx, lang in enumerate(full_train_dataset['language']):
        lang_to_indices[lang].append(idx)

    unique_languages = list(lang_to_indices.keys())
    print(f"Sampling 512 records from each of the {len(unique_languages)} languages...")
    subsets = []
    for lang, indices in lang_to_indices.items():
        rng.shuffle(indices)
        selected = indices[:min(512, len(indices))]
        sampled_subset = full_train_dataset.select(selected)
        subsets.append(sampled_subset)
        print(f"  - {lang}: {len(sampled_subset)} records")
    
    train_dataset = concatenate_datasets(subsets).shuffle(seed=3407)
    print(f"Total training records: {len(train_dataset)}")

    # Load validation split
    val_dataset = load_dataset("MBZUAI/EXAMS-V", split="validation")

    print("Formatting dataset...")
    formatted_dataset = train_dataset.map(format_data, batched=True, num_proc=num_proc)
    formatted_val_dataset = val_dataset.map(format_data, batched=True, num_proc=num_proc)
    
    print("Setting up trainer...")
    training_args = TrainingArguments(
        output_dir = "./outputs",
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 16,
        eval_accumulation_steps = 1,
        gradient_accumulation_steps = 1, 
        warmup_ratio = 0.1, 
        num_train_epochs = 5,
        # learning_rate = 5e-5, # Original learning rate
        learning_rate = 1e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1, 
        optim = "adamw_torch",
        seed = 3407,
        run_name = f"Qwen3-VL-8B-Instruct-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        metric_for_best_model = "eval_accuracy",
        save_strategy = "epoch", 
        eval_strategy = "epoch", 
        report_to = "wandb",
        remove_unused_columns = False,
        eval_on_start=True,
        average_tokens_across_devices = False,
        dataloader_num_workers = 4,
        dataloader_pin_memory = True,
        dataloader_persistent_workers = True,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer, resize=512),
        train_dataset = formatted_dataset,
        eval_dataset = formatted_val_dataset,
        compute_metrics = build_compute_metrics(tokenizer),
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        max_seq_length = 512,
        dataset_num_proc = 4,
        args = training_args,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained("qwen3_8b_lora_model")
    tokenizer.save_pretrained("qwen3_8b_lora_model")
    print("Done.")

if __name__ == "__main__":
    main()
