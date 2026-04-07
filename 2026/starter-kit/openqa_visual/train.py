import unsloth
from transformers import TrainingArguments
from unsloth import (
    FastVisionModel,
    UnslothVisionDataCollator,
    is_bfloat16_supported,
)
from unsloth.trainer import SFTTrainer

import argparse
from datetime import datetime
from functools import partial
from pathlib import Path

try:
    from .image_utils import ImagePreprocessingConfig, resize_image_if_needed
except ImportError:
    from image_utils import ImagePreprocessingConfig, resize_image_if_needed


DEFAULT_DATASET = "SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual"


QUESTION = "\n".join(
    (
        "The image contains an examination question.",
        "Anlayse the image and provide a concise answer to the question.",
        "Answer only with the final answer, without any explanation or reasoning steps."
    )
)


def format_data(examples, image_config: ImagePreprocessingConfig):
    all_messages = []
    for image, answer in zip(
        examples["image"],
        examples["answer"],
    ):
        processed_image = resize_image_if_needed(image, image_config)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_image},
                    {"type": "text", "text": QUESTION},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": str(answer).strip()},
                ],
            },
        ]
        all_messages.append(conversation)

    return {"messages": all_messages}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OpenQA visual model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Hugging Face dataset name or local dataset path used for training.",
    )
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


def sample_training_dataset(full_train_dataset):
    from datasets import concatenate_datasets

    from collections import defaultdict
    import random

    rng = random.Random(3407)
    lang_to_indices = defaultdict(list)
    for idx, lang in enumerate(full_train_dataset["language"]):
        lang_to_indices[lang].append(idx)

    unique_languages = list(lang_to_indices.keys())
    print(f"Sampling up to 512 records from each of the {len(unique_languages)} languages...")
    subsets = []
    for lang, indices in lang_to_indices.items():
        rng.shuffle(indices)
        selected = indices[: min(512, len(indices))]
        sampled_subset = full_train_dataset.select(selected)
        subsets.append(sampled_subset)
        print(f"  - {lang}: {len(sampled_subset)} records")

    return concatenate_datasets(subsets).shuffle(seed=3407)


def load_training_split(dataset_name_or_path: str):
    from datasets import load_dataset

    dataset_path = Path(dataset_name_or_path)
    if dataset_path.exists():
        parquet_files = sorted((dataset_path / "data").glob("train-*.parquet"))
        if parquet_files:
            return load_dataset(
                "parquet",
                data_files=[str(path) for path in parquet_files],
                split="train",
            )

    return load_dataset(dataset_name_or_path, split="train")


def main():

    args = parse_args()
    image_config = get_image_preprocessing_config(args)

    print("Loading model...")
    model, tokenizer = FastModel.from_pretrained(
        max_seq_length = 2048,
        load_in_4bit = False,     # MoE QLoRA not recommended, dense 27B is fine
        load_in_16bit = True,     # bf16/16-bit LoRA
        full_finetuning = False,
    )

    print(f"Loading dataset from {args.dataset}...")
    import multiprocessing

    num_proc = min(multiprocessing.cpu_count(), 32)
    full_train_dataset = load_training_split(args.dataset)
    train_dataset = sample_training_dataset(full_train_dataset)
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
        output_dir="./outputs",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        num_train_epochs=1,
        learning_rate=2e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_torch",
        seed=3407,
        run_name=f"Qwen3-VL-8B-Instruct-OpenQA-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        save_strategy="epoch",
        eval_strategy="no",
        report_to="wandb",
        remove_unused_columns=False,
        average_tokens_across_devices=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=formatted_dataset,
        max_seq_length=512,
        dataset_num_proc=4,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    # model.save_pretrained(args.output_model_path)
    # tokenizer.save_pretrained(args.output_model_path)
    model.save_pretrained_merged(args.output_model_path, tokenizer, save_method = "lora")

    print("Done.")


if __name__ == "__main__":
    main()
