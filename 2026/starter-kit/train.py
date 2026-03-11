import numpy as np
from unsloth import FastVisionModel, is_bfloat16_supported, UnslothVisionDataCollator
from unsloth.trainer import SFTTrainer
from transformers import TrainingArguments, EvalPrediction
from datasets import load_dataset, concatenate_datasets
from datetime import datetime

# Function to format the dataset for Qwen-VL (Batched version)
def format_data(examples):
    questions = ["Answer the multiple choice question presented in the image. Return the letter corresponding to the correct answer."] * len(examples['image'])
    
    all_messages = []
    for image, answer, question in zip(examples['image'], examples['answer_key'], questions):
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
                    {"type": "text", "text": answer}
                ]
            }
        ]
        all_messages.append(conversation)
    return {"messages": all_messages}


def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # Only evaluate on real (non-padding) label tokens
    mask = labels != -100
    accuracy = float((predictions[mask] == labels[mask]).sum() / mask.sum())
    return {"accuracy": accuracy}


def main():
    print("Loading model...")
    model_id = "unsloth/Qwen3-VL-4B-Instruct" # Using Qwen3 as requested
    
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
    
    # Sample 64 records per language
    unique_languages = set(full_train_dataset['language'])
    subsets = []
    print(f"Sampling 64 records from each of the {len(unique_languages)} languages...")
    for lang in unique_languages:
        lang_subset = full_train_dataset.filter(lambda x: x['language'] == lang, num_proc=num_proc)
        # Randomly sample 64 records
        sampled_subset = lang_subset.shuffle(seed=3407).select(range(min(64, len(lang_subset))))
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
        gradient_accumulation_steps = 4, 
        warmup_ratio = 0.1, 
        num_train_epochs = 10,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1, 
        optim = "adamw_torch",
        seed = 3407,
        run_name = f"Qwen3-VL-4B-Instruct-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        metric_for_best_model = "eval_accuracy",
        save_strategy = "epoch", 
        evaluation_strategy = "epoch", 
        report_to = "wandb",
        remove_unused_columns = False,
        average_tokens_across_devices = False,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer, resize=512),
        train_dataset = formatted_dataset,
        eval_dataset = formatted_val_dataset,
        compute_metrics = compute_metrics,
        max_seq_length = 512,
        dataset_num_proc = 4,
        args = training_args,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    print("Done.")

if __name__ == "__main__":
    main()
