import argparse
import json
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib import error, request

from tqdm.auto import tqdm

try:
    from .image_utils import (
        ImagePreprocessingConfig,
        encode_image_data_url,
        resize_image_if_needed,
    )
except ImportError:
    from image_utils import (
        ImagePreprocessingConfig,
        encode_image_data_url,
        resize_image_if_needed,
    )


DEFAULT_DATASET = "SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual"

QUESTION = "\n".join(
    (
        "The image contains an examination question. Your task is to retrieve and answer the question in the image.",
        "Perform the following steps:",
        "1. Provide the question found within the image.",
        "2. Provide a concise answer to the question, without any explanation or reasoning steps.",
    )
)


@dataclass(frozen=True)
class InferenceConfig:
    api_base: str
    api_key: str
    model_id_or_path: str
    max_tokens: int
    request_timeout: int
    max_retries: int
    retry_delay: float
    max_concurrent_requests: int
    resize_images: bool
    max_image_long_side: int
    max_image_pixels: int
    jpeg_quality: int


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return parsed


def sanitize_generated_answer(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "predictions"


def get_image_preprocessing_config(config: InferenceConfig) -> ImagePreprocessingConfig:
    return ImagePreprocessingConfig(
        resize_images=config.resize_images,
        max_image_long_side=config.max_image_long_side,
        max_image_pixels=config.max_image_pixels,
        jpeg_quality=config.jpeg_quality,
    )


def normalize_api_base(api_base: str) -> str:
    normalized = api_base.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def extract_message_text(payload: dict) -> str:
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = [item["text"] for item in content if item["type"] == "text"]
        return "\n".join(text_parts).strip()

    return str(content).strip()



def build_payload(
    config: InferenceConfig,
    image_data_url: str
) -> dict[str, Any]:
    return {
        "model": config.model_id_or_path,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": QUESTION},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": config.max_tokens,
    }


def post_chat_completion(
    api_base: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.lower() not in {"none", "null"}:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(
        url=f"{api_base}/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def request_prediction(item_id: Any, payload: dict[str, Any], config: InferenceConfig) -> str:
    last_error = None
    for attempt in range(1, config.max_retries + 1):
        try:
            response = post_chat_completion(
                api_base=config.api_base,
                api_key=config.api_key,
                payload=payload,
                timeout=config.request_timeout,
            )
            return extract_message_text(response)
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < config.max_retries:
                time.sleep(config.retry_delay)

    raise RuntimeError(
        f"Inference failed for item {item_id} after {config.max_retries} attempts."
    ) from last_error


def infer_example(
    index: int,
    example: dict[str, Any],
    config: InferenceConfig,
) -> dict[str, Any]:
    item_id = example["question_id"]
    image_config = get_image_preprocessing_config(config)
    processed_image = resize_image_if_needed(example["image"], image_config)
    payload = build_payload(
        config,
        encode_image_data_url(processed_image, image_config)
    )
    response_text = request_prediction(item_id=item_id, payload=payload, config=config)
    result = {
        "index": index,
        "id": item_id,
        "language": example["language"],
        "prediction": sanitize_generated_answer(response_text),
        "response_text": response_text,
    }
    return result


def get_total_examples(dataset, split: str) -> int | None:
    info = getattr(dataset, "info", None)
    splits = getattr(info, "splits", None)
    if not splits:
        return None

    split_info = splits.get(split)
    total_examples = getattr(split_info, "num_examples", None)
    if isinstance(total_examples, int) and total_examples > 0:
        return total_examples
    return None


def submit_next_example(
    executor: ThreadPoolExecutor,
    example_iter: Iterator[tuple[int, dict[str, Any]]],
    futures: dict[Future[dict[str, Any]], int],
    config: InferenceConfig,
) -> bool:
    try:
        index, example = next(example_iter)
    except StopIteration:
        return False

    future = executor.submit(infer_example, index, example, config)
    futures[future] = index
    return True


def run_parallel_inference(
    dataset,
    config: InferenceConfig,
    split: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    futures: dict[Future[dict[str, Any]], int] = {}
    example_iter = enumerate(dataset)
    total_examples = get_total_examples(dataset, split=split)

    with ThreadPoolExecutor(max_workers=config.max_concurrent_requests) as executor:
        for _ in range(config.max_concurrent_requests):
            if not submit_next_example(executor, example_iter, futures, config):
                break

        with tqdm(total=total_examples, desc="Inference", unit="example") as progress:
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    futures.pop(future, None)
                    results.append(future.result())
                    progress.update(1)
                    submit_next_example(executor, example_iter, futures, config)

    results.sort(key=lambda row: row["index"])
    return results


def print_preview(results: list[dict[str, Any]]) -> None:
    for row in results[:3]:
        print(f"\n--- Example {row['index'] + 1} ---")
        print(f"Model Output: {row['response_text']}")
        print(f"Saved Prediction: {row['prediction']}")


def build_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "prediction": row["prediction"],
        "language": row["language"]
    }


def load_inference_split(dataset_name_or_path: str, split: str):
    from datasets import load_dataset

    dataset_path = Path(dataset_name_or_path)
    if dataset_path.exists():
        parquet_files = sorted((dataset_path / "data").glob(f"{split}-*.parquet"))
        if parquet_files:
            return load_dataset(
                "parquet",
                data_files=[str(path) for path in parquet_files],
                split="train",
                streaming=True,
            )

    return load_dataset(dataset_name_or_path, split=split, streaming=True)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Run OpenQA visual inference against an OpenAI-compatible API.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Hugging Face dataset name or local dataset path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--model-id-or-path",
        required=True,
        help="Model identifier to send to the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:30010/v1"),
        help="Base URL for the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "None"),
        help="API key for the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=128,
        help="Maximum number of tokens to generate per example.",
    )
    parser.add_argument(
        "--request-timeout",
        type=positive_int,
        default=300,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=positive_int,
        default=3,
        help="Maximum number of retries per request.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Delay between failed request retries in seconds.",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=positive_int,
        default=16,
        help="Maximum number of simultaneous inference requests.",
    )
    parser.add_argument(
        "--no-resize-images",
        action="store_true",
        help="Disable image resizing before request serialization.",
    )
    parser.add_argument(
        "--max-image-long-side",
        type=positive_int,
        default=2048,
        help="Maximum allowed longest side for request images.",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=positive_int,
        default=2_000_000,
        help="Maximum allowed total pixels for request images.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=positive_int,
        default=90,
        help="JPEG quality used for opaque images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "predictions",
        help="Directory where predictions JSON will be saved.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional prediction filename. Defaults to a name derived from the model id.",
    )
    parser.add_argument(
        "--gold-output-name",
        type=str,
        default=None,
        help="Optional gold filename. When provided, writes a JSON list of OpenQA gold rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InferenceConfig(
        api_base=normalize_api_base(args.api_base),
        api_key=args.api_key,
        model_id_or_path=args.model_id_or_path,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        max_concurrent_requests=args.max_concurrent_requests,
        resize_images=not args.no_resize_images,
        max_image_long_side=args.max_image_long_side,
        max_image_pixels=args.max_image_pixels,
        jpeg_quality=args.jpeg_quality,
    )

    print(f"Loading {args.split} dataset from {args.dataset}...")
    dataset = load_inference_split(args.dataset, args.split)
    if config.resize_images:
        print(
            "Image preprocessing: resize enabled "
            f"(longest_side<={config.max_image_long_side}, "
            f"pixels<={config.max_image_pixels}, "
            f"jpeg_quality={config.jpeg_quality}, "
            "opaque=JPEG, alpha=PNG)"
        )
    else:
        print(
            "Image preprocessing: resize disabled "
            f"(jpeg_quality={config.jpeg_quality}, opaque=JPEG, alpha=PNG)"
        )

    print(
        f"Sending inference requests to {config.api_base} "
        f"with up to {config.max_concurrent_requests} concurrent workers..."
    )
    results = run_parallel_inference(
        dataset,
        config=config,
        split=args.split,
    )
    print_preview(results)

    pred_rows = [build_prediction_row(row) for row in results]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{sanitize_filename(args.model_id_or_path)}.json"
    pred_path = output_dir / output_name

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_rows, f, ensure_ascii=False, indent=2)

    print(f"\nSaved predictions to: {pred_path}")


if __name__ == "__main__":
    main()
