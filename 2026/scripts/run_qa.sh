#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/src/utils/format_checker.py" \
  --input_file "$ROOT_DIR/src/utils/example_pred_qa.json"

python "$ROOT_DIR/src/evaluation/evaluate_qa.py" \
  --pred_file "$ROOT_DIR/src/utils/example_pred_qa.json" \
  --gold_file "$ROOT_DIR/src/utils/example_gold_qa.json"
