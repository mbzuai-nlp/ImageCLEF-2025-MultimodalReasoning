import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT LoRA adapter into a Qwen3-VL base model.",
    )
    parser.add_argument(
        "--base-model-path",
        required=True,
        help="Path to the base Hugging Face model directory.",
    )
    parser.add_argument(
        "--adapter-model-path",
        required=True,
        help="Path to the adapter directory containing adapter_config.json.",
    )
    parser.add_argument(
        "--output-model-path",
        required=True,
        help="Directory where the merged model will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_model_path)
    output_path.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading base model from {args.base_model_path} ...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"Loading adapter from {args.adapter_model_path} ...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_model_path,
        is_trainable=False,
    )

    print("Merging adapter weights into the base model ...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to {output_path} ...")
    merged_model.save_pretrained(output_path)

    processor_source = (
        args.adapter_model_path
        if Path(args.adapter_model_path, "preprocessor_config.json").exists()
        else args.base_model_path
    )
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
    )
    processor.save_pretrained(output_path)
    print("Done.")


if __name__ == "__main__":
    main()
