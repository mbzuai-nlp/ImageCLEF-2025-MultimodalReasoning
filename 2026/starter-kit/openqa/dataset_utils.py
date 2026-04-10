import importlib
import sys
from pathlib import Path
from typing import Any


def import_hf_datasets_module():
    existing_module = sys.modules.get("datasets")
    if existing_module is not None and hasattr(existing_module, "load_dataset"):
        return existing_module

    project_root = Path(__file__).resolve().parent.parent
    removed_entries: list[tuple[int, str]] = []
    shadowed_module = sys.modules.pop("datasets", None)

    for index in range(len(sys.path) - 1, -1, -1):
        entry = sys.path[index]
        try:
            resolved = Path(entry or ".").resolve()
        except OSError:
            continue
        if resolved == project_root:
            removed_entries.append((index, sys.path.pop(index)))

    try:
        datasets_module = importlib.import_module("datasets")
    except Exception:
        if shadowed_module is not None:
            sys.modules["datasets"] = shadowed_module
        raise
    finally:
        for index, entry in reversed(removed_entries):
            sys.path.insert(index, entry)

    if not hasattr(datasets_module, "load_dataset"):
        raise ImportError(
            "Expected the Hugging Face 'datasets' package, but imported a module "
            "without 'load_dataset'."
        )

    return datasets_module


def load_dataset_split(dataset_name_or_path: str, split: str, streaming: bool = False):
    datasets_module = import_hf_datasets_module()
    load_dataset = datasets_module.load_dataset

    dataset_path = Path(dataset_name_or_path)
    if dataset_path.exists():
        parquet_files = sorted((dataset_path / "data").glob(f"{split}-*.parquet"))
        if parquet_files:
            return load_dataset(
                "parquet",
                data_files={split: [str(path) for path in parquet_files]},
                split=split,
                streaming=streaming,
            )

    return load_dataset(dataset_name_or_path, split=split, streaming=streaming)


def resolve_item_id(example: dict[str, Any]) -> Any:
    return example["question_id"]


def get_batched_item_ids(examples: dict[str, list[Any]]) -> list[Any]:
    return list(examples["question_id"])
