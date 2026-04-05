from unsloth import FastVisionModel, UnslothVisionDataCollator, is_bfloat16_supported
from unsloth.trainer import SFTTrainer
from datasets import concatenate_datasets, load_dataset
from transformers import TrainingArguments

import argparse
from datetime import datetime
from functools import partial

try:
    from .image_utils import ImagePreprocessingConfig, resize_image_if_needed
except ImportError:
    from image_utils import ImagePreprocessingConfig, resize_image_if_needed


def build_instruction_text() -> str:
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
    return "\n".join(question)


# Function to format the dataset for Qwen-VL (Batched version)
def format_data(examples, image_config: ImagePreprocessingConfig):
    questions = [build_instruction_text()] * len(examples['image'])

    all_messages = []
    for image, answer, question in zip(examples['image'], examples['answer_key'], questions):
        answer_text = str(answer).strip().upper()
        processed_image = resize_image_if_needed(image, image_config)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_image},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer_text}
                ]
            },
        ]
        all_messages.append(conversation)
        
    return {"messages": all_messages}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MCQ visual model.")
    parser.add_argument(
        "--model-id-or-path",
        type=str,
        default="unsloth/Qwen3-VL-8B-Instruct",
        help="HF model ID or local model path used as the training base model.",
    )
    parser.add_argument(
        "--output-model-path",
        required=True,
        help="Directory path where the trained model and tokenizer will be saved.",
    )
    parser.add_argument(
        "--no-resize-images",
        action="store_true",
        help="Disable image resizing before building training examples.",
    )
    parser.add_argument(
        "--max-image-long-side",
        type=positive_int,
        default=2048,
        help="Maximum allowed longest side for training images.",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=positive_int,
        default=2_000_000,
        help="Maximum allowed total pixels for training images.",
    )
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return parsed


def get_image_preprocessing_config(args: argparse.Namespace) -> ImagePreprocessingConfig:
    return ImagePreprocessingConfig(
        resize_images=not args.no_resize_images,
        max_image_long_side=args.max_image_long_side,
        max_image_pixels=args.max_image_pixels,
    )


def main():
    args = parse_args()


    image_config = get_image_preprocessing_config(args)

    print("Loading model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_id_or_path,
        use_gradient_checkpointing = "unsloth",
    )
    
    # Configure LoRA
    model = FastVisionModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0.05,
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

    print("Formatting dataset...")
    if image_config.resize_images:
        print(
            "Image preprocessing: resize enabled "
            f"(longest_side<={image_config.max_image_long_side}, "
            f"pixels<={image_config.max_image_pixels})"
        )
    else:
        print("Image preprocessing: resize disabled")
    formatted_dataset = train_dataset.map(
        partial(format_data, image_config=image_config),
        batched=True,
        num_proc=num_proc,
    )
    
    print("Setting up trainer...")
    training_args = TrainingArguments(
        output_dir = "./outputs",
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1, 
        warmup_ratio = 0.2, 
        num_train_epochs = 2,
        learning_rate = 2e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1, 
        optim = "adamw_torch",
        seed = 3407,
        run_name = f"Qwen3-VL-8B-Instruct-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        save_strategy = "epoch", 
        eval_strategy = "no",
        report_to = "wandb",
        remove_unused_columns = False,
        average_tokens_across_devices = False,
        dataloader_num_workers = 4,
        dataloader_pin_memory = True,
        dataloader_persistent_workers = True,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = formatted_dataset,
        max_seq_length = 512,
        dataset_num_proc = 4,
        args = training_args,
    )

    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained(args.output_model_path)
    tokenizer.save_pretrained(args.output_model_path)
    print("Done.")


if __name__ == "__main__":
    main()
