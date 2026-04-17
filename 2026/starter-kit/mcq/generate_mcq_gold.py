#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def resolve_item_id(example: dict[str, Any], index: int, id_mode: str) -> Any:
    if id_mode == "index":
        return index

    for key in ("id", "sample_id"):
        value = example.get(key)
        if value is not None:
            return value
    return index


def resolve_gold_answer(example: dict[str, Any], item_id: Any) -> str:
    for key in ("answer", "answer_key"):
        value = example.get(key)
        if value is not None:
            return str(value).strip().upper()

    raise ValueError(
        f"Item {item_id} does not contain 'answer' or 'answer_key'; cannot export gold."
    )


def iter_local_dataset_examples(dataset_path: Path, split: str) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet_files = sorted((dataset_path / "data").glob(f"{split}-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found for split '{split}' under {dataset_path / 'data'}."
        )

    preferred_columns = ("id", "sample_id", "answer", "answer_key")
    first_file_columns = set(pq.ParquetFile(parquet_files[0]).schema.names)
    columns = [name for name in preferred_columns if name in first_file_columns]
    if not columns:
        raise ValueError(
            f"Could not find any of {preferred_columns} in {parquet_files[0]}."
        )

    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=columns)
        yield from table.to_pylist()


def iter_dataset_examples(
    dataset_name_or_path: str,
    split: str,
    cache_dir: Path,
) -> Iterable[dict[str, Any]]:
    dataset_path = Path(dataset_name_or_path)
    if dataset_path.exists():
        yield from iter_local_dataset_examples(dataset_path, split)
        return

    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        dataset_name_or_path,
        split=split,
        streaming=True,
        cache_dir=str(cache_dir),
    )
    features = getattr(dataset, "features", None)
    if features:
        keep_columns = {"id", "sample_id", "answer", "answer_key"}
        drop_columns = [name for name in features if name not in keep_columns]
        if drop_columns:
            dataset = dataset.remove_columns(drop_columns)

    yield from dataset


def parse_args() -> argparse.Namespace:
    project_root = Path.cwd()

    parser = argparse.ArgumentParser(
        description="Generate evaluator-compatible MCQ gold JSON from an HF dataset split.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MBZUAI/EXAMS-V",
        help="Hugging Face dataset name or local dataset path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to export.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Destination JSON file containing a list of {id, answer}.",
    )
    parser.add_argument(
        "--id-mode",
        choices=("resolved", "index"),
        default="resolved",
        help="How to populate the id field. 'resolved' prefers id/sample_id, 'index' uses row order.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_root / ".cache" / "huggingface",
        help="Writable Hugging Face cache directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.split} split from {args.dataset}...")

    rows = []
    for index, example in enumerate(
        iter_dataset_examples(args.dataset, args.split, args.cache_dir)
    ):
        item_id = resolve_item_id(example, index, args.id_mode)
        rows.append(
            {
                "id": item_id,
                "answer": resolve_gold_answer(example, item_id),
            }
        )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(rows)} gold rows to: {args.output_file}")


if __name__ == "__main__":
    main()
