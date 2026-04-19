import json
import argparse
import os
from inspect import signature
from typing import Dict, List

ID_FIELD_NAME = "id"
PRED_ANSWER_FIELD_NAME = "answer_key"
GOLD_ANSWER_FIELD_NAME = "answer"
ALT_PRED_ANSWER_FIELD_NAME = "prediction"

# Full language name -> ISO 639-1 two-letter code.
_LANGUAGE_ISO2: Dict[str, str] = {
    "bulgarian": "bg",
    "chinese": "zh",
    "croatian": "hr",
    "english": "en",
    "italian": "it",
    "serbian": "sr",
}


def normalise_language(lang: str) -> str:
    if not lang:
        return "unknown"
    stripped = str(lang).strip()
    if len(stripped) == 2:
        return stripped.lower()
    return _LANGUAGE_ISO2.get(stripped.lower(), stripped)


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def atomic_write_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def round_float_values(obj, ndigits: int = 4):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_float_values(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_float_values(v, ndigits) for v in obj]
    return obj


def _read_list_or_data(path: str, file_label: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        jsn_data = json.load(f)

    if isinstance(jsn_data, dict) and "data" in jsn_data:
        jsn_data = jsn_data["data"]

    if not isinstance(jsn_data, list):
        raise Exception(
            f"{file_label} file format is invalid. JSON root must be a list or {{'data': [...]}}."
        )

    if len(jsn_data) == 0:
        raise Exception(f"{file_label} file is empty.")

    return jsn_data


def call_validations(functions, *args, **kwargs):
    results = []
    for func in functions:
        sig = signature(func)
        func_args = {}

        for param in sig.parameters:
            if param in kwargs:
                func_args[param] = kwargs[param]
            elif args:
                func_args[param] = args[0]
                args = args[1:]
        results.append(func(**func_args))
    return results


################### Validation functions ###################
def are_keys_correct(pred):
    for item in pred.values():
        answer = item["answer"]
        if not (
            answer == "A"
            or answer == "B"
            or answer == "C"
            or answer == "D"
            or answer == "E"
        ):
            raise Exception(
                "Unsupported answer: {}. Cannot score predictions.".format(answer)
            )


def dataset_size_match(pred, gold):
    if not len(pred) == len(gold):
        raise Exception(
            "Invalid pred file. Prediction size does not match test or duplicate ids were found."
        )


def ids_match(pred, gold):
    for key in pred.keys():
        try:
            gold[key]
        except:
            raise Exception(
                "Invalid submission. Test data does not contain id: {}".format(key)
            )


################### Validation functions ###################
def read_data(path, answer_field_name, file_label):
    data = {}
    jsn_data = _read_list_or_data(path, file_label)

    for idx, el in enumerate(jsn_data):
        if not isinstance(el, dict):
            raise Exception(
                f"{file_label} file format is invalid. Each item must be an object."
            )

        if ID_FIELD_NAME not in el:
            raise Exception(
                f"{file_label} file format is invalid. Each object must contain id and {answer_field_name}."
            )

        resolved_answer_field = answer_field_name
        if file_label == "Prediction" and answer_field_name not in el:
            if ALT_PRED_ANSWER_FIELD_NAME in el:
                resolved_answer_field = ALT_PRED_ANSWER_FIELD_NAME

        if resolved_answer_field not in el:
            raise Exception(
                f"{file_label} file format is invalid. Each object must contain id and {answer_field_name}."
            )

        item_id = el[ID_FIELD_NAME]
        if item_id in data:
            raise Exception(
                f"{file_label} file format is invalid. Duplicate id found at index {idx}: {item_id}"
            )

        data[item_id] = {
            "answer": str(el[resolved_answer_field]).strip().upper(),
            "language": el.get("language", "unknown"),
        }
    return data


def load_pred_gold(pred_path, gold_path):
    # Load data
    pred = read_data(pred_path, PRED_ANSWER_FIELD_NAME, "Prediction")
    gold = read_data(gold_path, GOLD_ANSWER_FIELD_NAME, "Gold")

    # Validate data
    validations = [dataset_size_match, are_keys_correct, ids_match]
    val_results = call_validations(validations, pred=pred, gold=gold)

    if False in val_results:
        return None, None

    return pred, gold


def evaluate(pred_path, gold_path):
    report = evaluate_with_language(pred_path, gold_path)
    return report["accuracy"]


def evaluate_with_language(pred_path, gold_path):
    pred, gold = load_pred_gold(pred_path, gold_path)

    correct = 0.0
    total = len(gold)
    per_language_counts: Dict[str, Dict[str, float]] = {}

    for qstn_id, gold_item in gold.items():
        pred_item = pred[qstn_id]

        if pred_item["answer"] == gold_item["answer"]:
            correct += 1

        lang = gold_item.get("language") or pred_item.get("language") or "unknown"
        lang = normalise_language(lang)

        if lang not in per_language_counts:
            per_language_counts[lang] = {"correct": 0.0, "total": 0.0}

        per_language_counts[lang]["total"] += 1
        if pred_item["answer"] == gold_item["answer"]:
            per_language_counts[lang]["correct"] += 1

    overall_accuracy = (correct / total) if total > 0 else 0.0

    per_language = {}
    for lang in sorted(per_language_counts.keys()):
        lang_total = per_language_counts[lang]["total"]
        lang_correct = per_language_counts[lang]["correct"]
        per_language[lang] = {
            "accuracy": (lang_correct / lang_total) if lang_total > 0 else 0.0,
            "num_samples": int(lang_total),
        }

    report = {
        "num_samples": total,
        "accuracy": overall_accuracy,
        "per_language": per_language,
    }
    return round_float_values(report, ndigits=4)


"""
    Example script usage:
        python evaluate.py --pred_file="./pred.json" --gold_file="./gold.json" --print_score="True"
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, help="path to submission file")
    parser.add_argument("--gold_file", type=str, help="path to gold truth file")
    parser.add_argument(
        "--out_file",
        type=str,
        default="scores.json",
        help="Path to write metrics report JSON. Defaults to 'scores.json'.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print metrics report to stdout after writing file.",
    )
    parser.add_argument(
        "--print_score",
        type=bool,
        default=False,
        help="Backward-compatible flag to print overall accuracy.",
    )
    args = parser.parse_args()

    report = evaluate_with_language(args.pred_file, args.gold_file)

    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    ensure_outdir(out_dir)
    atomic_write_json(args.out_file, report)

    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Metrics written to: {os.path.abspath(args.out_file)}")

    if args.print_score:
        print(report["accuracy"])
