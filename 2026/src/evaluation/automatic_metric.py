#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, time, argparse
from typing import List, Dict

from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import sacrebleu
from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import single_meteor_score

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


load_dotenv(find_dotenv(), override=True)


def ensure_outdir(p: str):
    os.makedirs(p, exist_ok=True)


def atomic_write_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_input_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support optional wrapper {"data": [...]}
    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    if not isinstance(data, list):
        raise ValueError("Input must be a list or {'data': [...]}")

    if len(data) == 0:
        raise ValueError("Input data is empty")

    required_fields = {"id", "image_id", "question", "answer", "prediction"}

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx} is not a dictionary")

        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(
                f"Item at index {idx} is missing required fields: {missing}"
            )

    return data


def load_json(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_id_map(rows: List[Dict]) -> Dict[int, Dict]:
    out = {}
    for r in rows:
        try:
            out[int(r["id"])] = r
        except:
            pass
    return out


def items_by_id(items: List[Dict]) -> Dict[int, Dict]:
    return {int(i["id"]): i for i in items}


def update_json_field(path: str, updates: Dict[int, Dict]):
    data = load_json(path)
    by_id = to_id_map(data)
    for _id, fields in updates.items():
        row = by_id.get(_id, {"id": _id})
        for k in ("answer", "prediction"):
            if k in fields:
                row[k] = fields[k]
        for k, v in fields.items():
            if k not in ("id", "answer", "prediction"):
                row[k] = v
        by_id[_id] = row
    atomic_write_json(path, list(by_id.values()))


def sentence_bleu_n(hyp: str, ref: str, n: int) -> float:
    """
    Sentence BLEU with max n-gram order = n.
    Returns a score in [0, 100] like SacreBLEU.
    """
    metric = BLEU(max_ngram_order=n, effective_order=True)
    return float(metric.sentence_score(hyp, [ref]).score)


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
    return float(single_meteor_score(ref_toks, hyp_toks) * 100.0)


def sentence_rouge_scores(hyp: str, ref: str) -> Dict[str, float]:
    """
    ROUGE f-measure in [0,1]; we scale to [0,100] to match BLEU style.
    Returns dict for rouge-1/2/L.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, hyp)  # reference first, hypothesis second
    return {
        "rouge-1": float(scores["rouge1"].fmeasure * 100.0),
        "rouge-2": float(scores["rouge2"].fmeasure * 100.0),
        "rouge-l": float(scores["rougeL"].fmeasure * 100.0),
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


def step_bleu(items: List[Dict], metrics_path: str):
    # Skip items already computed
    done = {int(d["id"]) for d in load_json(metrics_path) if "bleu_scores" in d}

    for it in tqdm(items, desc="BLEU-1..4"):
        _id = int(it["id"])
        if _id in done:
            continue

        hyp = str(it["prediction"])
        ref = str(it["answer"])

        b1 = sentence_bleu_n(hyp, ref, 1)
        b2 = sentence_bleu_n(hyp, ref, 2)
        b3 = sentence_bleu_n(hyp, ref, 3)
        b4 = sentence_bleu_n(hyp, ref, 4)
        avg = (b1 + b2 + b3 + b4) / 4.0

        update_json_field(
            metrics_path,
            {
                _id: {
                    "answer": it["answer"],
                    "prediction": it["prediction"],
                    "bleu_scores": {
                        "bleu-1": b1,
                        "bleu-2": b2,
                        "bleu-3": b3,
                        "bleu-4": b4,
                    },
                    "bleu_avg": avg,  # remove this line if you don't want the avg stored
                }
            },
        )


def step_rouge(items: List[Dict], metrics_path: str):
    done = {int(d["id"]) for d in load_json(metrics_path) if "rouge_scores" in d}

    for it in tqdm(items, desc="ROUGE-1/2/L"):
        _id = int(it["id"])
        if _id in done:
            continue

        hyp = str(it["prediction"])
        ref = str(it["answer"])
        r = sentence_rouge_scores(hyp, ref)

        update_json_field(
            metrics_path,
            {
                _id: {
                    "answer": it["answer"],
                    "prediction": it["prediction"],
                    "rouge_scores": r,
                }
            },
        )


def step_meteor(items: List[Dict], metrics_path: str):
    done = {int(d["id"]) for d in load_json(metrics_path) if "meteor" in d}

    for it in tqdm(items, desc="METEOR"):
        _id = int(it["id"])
        if _id in done:
            continue

        hyp = str(it["prediction"])
        ref = str(it["answer"])
        m = sentence_meteor(hyp, ref)

        update_json_field(
            metrics_path,
            {
                _id: {
                    "answer": it["answer"],
                    "prediction": it["prediction"],
                    "meteor": m,
                }
            },
        )


def step_comet(items: List[Dict], metrics_path: str, batch_size: int = 64):
    """
    Computes a single meaningful COMET score for QA/text generation:
      src = question
      mt  = prediction
      ref = answer

    Writes incrementally to metrics_path under key: "comet".
    """
    data = load_json(metrics_path)
    by_id = to_id_map(data)
    base = items_by_id(items)

    model = load_comet_model("Unbabel/wmt22-comet-da")

    pending_ids, srcs, mts, refs = [], [], [], []

    for it in items:
        _id = int(it["id"])
        row = by_id.get(_id, {"id": _id})
        if "comet" not in row:
            pending_ids.append(_id)
            srcs.append(str(it["question"]))
            mts.append(str(it["prediction"]))
            refs.append(str(it["answer"]))

    for s in tqdm(range(0, len(pending_ids), batch_size), desc="COMET"):
        e = s + batch_size
        scores = comet_scores_batch(
            model,
            srcs[s:e],
            mts[s:e],
            refs[s:e],
            batch_size=batch_size,
        )
        update_json_field(
            metrics_path,
            {
                pid: {
                    "answer": base[pid]["answer"],
                    "prediction": base[pid]["prediction"],
                    "comet": scores[i],
                }
                for i, pid in enumerate(pending_ids[s:e])
            },
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--batch_size_comet", type=int, default=64)
    args = ap.parse_args()

    out_dir = os.path.join("2026/src/evaluation/automatic_metrics")
    ensure_outdir(out_dir)
    metrics_path = os.path.join(out_dir, f"metrics.json")

    items = load_input_data(args.input)

    step_bleu(items, metrics_path)
    step_rouge(items, metrics_path)
    step_meteor(items, metrics_path)
    step_comet(items, metrics_path, batch_size=args.batch_size_comet)


if __name__ == "__main__":
    main()
