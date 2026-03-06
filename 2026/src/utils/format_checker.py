#!/usr/bin/env python3

import argparse
import json
import sys
from typing import Dict, List

REQUIRED_KEYS = {"id", "prediction", "language"}


def load_submission(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    if not isinstance(data, list):
        raise ValueError("Submission must be a JSON list or {'data': [...]}.")

    if len(data) == 0:
        raise ValueError("Submission is empty.")

    return data


def validate_items(items: List[Dict]) -> None:
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx} is not a JSON object.")

        missing = REQUIRED_KEYS - set(item.keys())
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(
                f"Item at index {idx} is missing required key(s): {missing_keys}."
            )


def check_format(path: str) -> int:
    try:
        items = load_submission(path)
        validate_items(items)
    except Exception as e:
        print(f"[INVALID] Could not read submission: {e}")
        return 1

    print(
        f"[VALID] Submission format is correct. Found {len(items)} item(s) with keys: {sorted(REQUIRED_KEYS)}"
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate participant submission JSON format."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to participant submission JSON file.",
    )
    args = parser.parse_args()

    sys.exit(check_format(args.input_file))


if __name__ == "__main__":
    main()
