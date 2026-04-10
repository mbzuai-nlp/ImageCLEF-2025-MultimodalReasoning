import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib import error, request

from tqdm.auto import tqdm

try:
    from .dataset_utils import load_dataset_split, resolve_item_id
    from .image_utils import (
        ImagePreprocessingConfig,
        encode_image_data_url,
        resize_image_if_needed,
    )
    from .question_utils import QUESTION_EXTRACTION_PROMPT, sanitize_extracted_question
except ImportError:
    from dataset_utils import load_dataset_split, resolve_item_id
    from image_utils import (
        ImagePreprocessingConfig,
        encode_image_data_url,
        resize_image_if_needed,
    )
    from question_utils import QUESTION_EXTRACTION_PROMPT, sanitize_extracted_question


DEFAULT_DATASET = "SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual"


@dataclass(frozen=True)
class ExtractionConfig:
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
        return "\n".join(part for part in text_parts if part).strip()

    return str(content).strip()


def get_image_preprocessing_config(config: ExtractionConfig) -> ImagePreprocessingConfig:
    return ImagePreprocessingConfig(
        resize_images=config.resize_images,
        max_image_long_side=config.max_image_long_side,
        max_image_pixels=config.max_image_pixels,
        jpeg_quality=config.jpeg_quality,
    )


def build_payload(
    config: ExtractionConfig,
    image_data_url: str,
) -> dict[str, Any]:
    return {
        "model": config.model_id_or_path,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUESTION_EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
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


def request_question(item_id: Any, payload: dict[str, Any], config: ExtractionConfig) -> str:
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
        f"Question extraction failed for item {item_id} after {config.max_retries} attempts."
    ) from last_error


def extract_example(
    index: int,
    example: dict[str, Any],
    split: str,
    config: ExtractionConfig,
) -> dict[str, Any]:
    item_id = resolve_item_id(example)
    image_config = get_image_preprocessing_config(config)
    processed_image = resize_image_if_needed(example["image"], image_config)
    payload = build_payload(config, encode_image_data_url(processed_image, image_config))
    response_text = request_question(item_id=item_id, payload=payload, config=config)
    return {
        "index": index,
        "id": item_id,
        "split": split,
        "language": example["language"],
        "question": sanitize_extracted_question(response_text),
    }


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
    split: str,
    config: ExtractionConfig,
) -> bool:
    try:
        index, example = next(example_iter)
    except StopIteration:
        return False

    future = executor.submit(extract_example, index, example, split, config)
    futures[future] = index
    return True


def run_parallel_extraction(
    dataset,
    split: str,
    config: ExtractionConfig,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    futures: dict[Future[dict[str, Any]], int] = {}
    example_iter = enumerate(dataset)
    total_examples = get_total_examples(dataset, split=split)

    with ThreadPoolExecutor(max_workers=config.max_concurrent_requests) as executor:
        for _ in range(config.max_concurrent_requests):
            if not submit_next_example(executor, example_iter, futures, split, config):
                break

        with tqdm(total=total_examples, desc=f"Extracting {split}", unit="example") as progress:
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    futures.pop(future, None)
                    results.append(future.result())
                    progress.update(1)
                    submit_next_example(executor, example_iter, futures, split, config)

    results.sort(key=lambda row: row["index"])
    return results


def write_jsonl(output_file: Path, rows: list[dict[str, Any]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "split": row["split"],
                        "language": row["language"],
                        "question": row["question"],
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")


def print_preview(rows: list[dict[str, Any]]) -> None:
    for row in rows[:3]:
        print(f"\n--- {row['split']} example {row['index'] + 1} ---")
        print(f"Extracted Question: {row['question']}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Extract OpenQA question text from images into a JSONL file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Hugging Face dataset name or local dataset path.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev"],
        help="Dataset splits to process into one JSONL file.",
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
        default=512,
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
        help="Maximum number of simultaneous extraction requests.",
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
        "--output-file",
        type=Path,
        default=project_root / "predictions" / "openqa_visual_questions.jsonl",
        help="Destination JSONL file for extracted questions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExtractionConfig(
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

    all_rows: list[dict[str, Any]] = []
    for split in args.splits:
        print(f"Loading {split} dataset from {args.dataset}...")
        dataset = load_dataset_split(args.dataset, split=split, streaming=True)
        split_rows = run_parallel_extraction(dataset, split=split, config=config)
        all_rows.extend(split_rows)

    print_preview(all_rows)
    write_jsonl(args.output_file, all_rows)
    print(f"\nSaved extracted questions to: {args.output_file}")


if __name__ == "__main__":
    main()
