#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse
from typing import List, Dict

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import single_meteor_score

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


load_dotenv(find_dotenv(), override=True)

_ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
)


def ensure_outdir(p: str):
    os.makedirs(p, exist_ok=True)


def atomic_write_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def avg(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def round_float_values(obj, ndigits: int = 4):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_float_values(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_float_values(v, ndigits) for v in obj]
    return obj


def _read_list_or_data(path: str, kind: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    if not isinstance(data, list):
        raise ValueError(f"{kind} file must be a list or {{'data': [...]}}")

    if len(data) == 0:
        raise ValueError(f"{kind} file is empty")

    return data


def _build_id_map(
    items: List[Dict], required_fields: set, kind: str
) -> Dict[int, Dict]:
    id_map = {}
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{kind} item at index {idx} is not a dictionary")

        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(
                f"{kind} item at index {idx} is missing required fields: {missing}"
            )

        try:
            item_id = int(item["id"])
        except Exception:
            raise ValueError(
                f"{kind} item at index {idx} has a non-integer id: {item.get('id')}"
            )

        if item_id in id_map:
            raise ValueError(f"{kind} file contains duplicate id: {item_id}")

        id_map[item_id] = item

    return id_map


def load_input_data(gold_path: str, pred_path: str) -> List[Dict]:
    gold_items = _read_list_or_data(gold_path, "Gold")
    pred_items = _read_list_or_data(pred_path, "Prediction")

    gold_map = _build_id_map(
        gold_items,
        {"id", "answer"},
        "Gold",
    )
    pred_map = _build_id_map(
        pred_items,
        {"id", "prediction"},
        "Prediction",
    )

    gold_ids = set(gold_map.keys())
    pred_ids = set(pred_map.keys())

    missing_in_pred = gold_ids - pred_ids
    extra_in_pred = pred_ids - gold_ids

    if missing_in_pred or extra_in_pred:
        msg = []
        if missing_in_pred:
            msg.append(f"Missing prediction ids: {sorted(list(missing_in_pred))[:5]}")
        if extra_in_pred:
            msg.append(f"Unknown prediction ids: {sorted(list(extra_in_pred))[:5]}")
        raise ValueError(" | ".join(msg))

    merged = []
    for item in gold_items:
        item_id = int(item["id"])
        question = gold_map[item_id].get("question", pred_map[item_id].get("question"))
        if question is None:
            raise ValueError(
                f"Missing question for id {item_id}. Include 'question' in either gold or prediction file."
            )

        merged_item = {
            "id": item_id,
            "question": question,
            "answer": gold_map[item_id]["answer"],
            "prediction": pred_map[item_id]["prediction"],
        }

        for key in ("image_id", "language"):
            if key in gold_map[item_id]:
                merged_item[key] = gold_map[item_id][key]
            elif key in pred_map[item_id]:
                merged_item[key] = pred_map[item_id][key]

        merged.append(merged_item)

    return merged


def sentence_bleu_n(hyp: str, ref: str, n: int) -> float:
    """
    Sentence BLEU with max n-gram order = n.
    Returns a score in [0, 1].
    """
    metric = BLEU(max_ngram_order=n, effective_order=True)
    return float(metric.sentence_score(hyp, [ref]).score / 100.0)


def ensure_nltk_meteor_resources():
    """
    METEOR may require NLTK resources depending on environment.
    We'll try to download minimal resources if missing.
    """
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)


def meteor_tokenize(text: str):
    return _TOKEN_RE.findall(text)


def sentence_meteor(hyp: str, ref: str) -> float:
    ref_toks = meteor_tokenize(ref)
    hyp_toks = meteor_tokenize(hyp)
    return float(single_meteor_score(ref_toks, hyp_toks))


def sentence_rouge_scores(hyp: str, ref: str) -> Dict[str, float]:
    """
    ROUGE f-measure in [0,1].
    Returns dict for rouge-1/2/L.
    """
    scores = _ROUGE_SCORER.score(ref, hyp)  # reference first, hypothesis second
    return {
        "rouge-1": float(scores["rouge1"].fmeasure),
        "rouge-2": float(scores["rouge2"].fmeasure),
        "rouge-l": float(scores["rougeL"].fmeasure),
    }


def load_comet_model(model_name="Unbabel/wmt22-comet-da"):
    ckpt = download_model(model_name)
    return load_from_checkpoint(ckpt)


def comet_scores_batch(
    model, srcs: List[str], mts: List[str], refs: List[str], batch_size=16
):
    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, refs)]
    out = model.predict(data, batch_size=batch_size, accelerator="cpu", num_workers=1)
    return [float(x) for x in out.scores]


def step_bleu(items: List[Dict]) -> Dict[str, float]:
    bleu1, bleu2, bleu3, bleu4 = [], [], [], []
    for it in tqdm(items, desc="BLEU-1..4"):
        hyp = str(it["prediction"])
        ref = str(it["answer"])

        bleu1.append(sentence_bleu_n(hyp, ref, 1))
        bleu2.append(sentence_bleu_n(hyp, ref, 2))
        bleu3.append(sentence_bleu_n(hyp, ref, 3))
        bleu4.append(sentence_bleu_n(hyp, ref, 4))

    score_1 = avg(bleu1)
    score_2 = avg(bleu2)
    score_3 = avg(bleu3)
    score_4 = avg(bleu4)

    return {
        "bleu-1": score_1,
        "bleu-2": score_2,
        "bleu-3": score_3,
        "bleu-4": score_4,
        "bleu_avg": avg([score_1, score_2, score_3, score_4]),
    }


def step_rouge(items: List[Dict]) -> Dict[str, float]:
    rouge1, rouge2, rougel = [], [], []

    for it in tqdm(items, desc="ROUGE-1/2/L"):
        hyp = str(it["prediction"])
        ref = str(it["answer"])
        r = sentence_rouge_scores(hyp, ref)

        rouge1.append(r["rouge-1"])
        rouge2.append(r["rouge-2"])
        rougel.append(r["rouge-l"])

    return {
        "rouge-1": avg(rouge1),
        "rouge-2": avg(rouge2),
        "rouge-l": avg(rougel),
    }


def step_meteor(items: List[Dict]) -> float:
    meteor_scores = []

    for it in tqdm(items, desc="METEOR"):
        hyp = str(it["prediction"])
        ref = str(it["answer"])
        meteor_scores.append(sentence_meteor(hyp, ref))

    return avg(meteor_scores)


def step_comet(items: List[Dict], batch_size: int = 64) -> float:
    """
    Computes a single meaningful COMET score for QA/text generation:
      src = question
      mt  = prediction
      ref = answer

    Returns the task-level average COMET score.
    """
    model = load_comet_model("Unbabel/wmt22-comet-da")

    srcs = [str(it["question"]) for it in items]
    mts = [str(it["prediction"]) for it in items]
    refs = [str(it["answer"]) for it in items]

    all_scores = []
    for s in tqdm(range(0, len(items), batch_size), desc="COMET"):
        e = s + batch_size
        scores = comet_scores_batch(
            model,
            srcs[s:e],
            mts[s:e],
            refs[s:e],
            batch_size=batch_size,
        )
        all_scores.extend(scores)

    return avg(all_scores)


def evaluate_openqa(items: List[Dict], batch_size_comet: int = 64) -> Dict:
    ensure_nltk_meteor_resources()

    bleu_scores = step_bleu(items)
    rouge_scores = step_rouge(items)
    meteor = step_meteor(items)
    comet = step_comet(items, batch_size=batch_size_comet)

    report = {
        "num_samples": len(items),
        "bleu_scores": {
            "bleu-1": bleu_scores["bleu-1"],
            "bleu-2": bleu_scores["bleu-2"],
            "bleu-3": bleu_scores["bleu-3"],
            "bleu-4": bleu_scores["bleu-4"],
        },
        "bleu_avg": bleu_scores["bleu_avg"],
        "rouge_scores": rouge_scores,
        "meteor": meteor,
        "comet": comet,
    }

    return round_float_values(report, ndigits=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--gold_file", required=True)
    ap.add_argument("--batch_size_comet", type=int, default=64)
    ap.add_argument(
        "--out_file",
        type=str,
        default="2026/src/evaluation/automatic_metrics/metrics.json",
        help="Path to save the average task-level metrics report.",
    )
    args = ap.parse_args()

    items = load_input_data(args.gold_file, args.pred_file)

    report = evaluate_openqa(items, batch_size_comet=args.batch_size_comet)

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        ensure_outdir(out_dir)
    atomic_write_json(args.out_file, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
