#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/src/utils/format_checker.py" \
  --input_file "$ROOT_DIR/src/utils/example_pred_mcq.json"

python "$ROOT_DIR/src/evaluation/evaluate_mcq.py" \
  --pred_file "$ROOT_DIR/src/utils/example_pred_mcq.json" \
  --gold_file "$ROOT_DIR/src/utils/example_gold_mcq.json" \
  --print_score True
