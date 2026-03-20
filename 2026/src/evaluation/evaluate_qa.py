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
import warnings
warnings.filterwarnings("ignore")

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Full language name → ISO 639-1 two-letter code.
# Keys are lower-cased for case-insensitive lookup.
_LANGUAGE_ISO2: Dict[str, str] = {
    "bulgarian": "bg",
    "chinese": "zh",
    "croatian": "hr",
    "english": "en",
    "italian": "it",
    "serbian": "sr",
}


def normalise_language(lang: str) -> str:
    """Return the ISO 639-1 two-letter code for *lang*.

    If *lang* is already a 2-letter code it is returned as-is (lower-cased).
    Unknown names are returned unchanged so they still surface in reports.
    """
    if not lang:
        return "unknown"
    stripped = lang.strip()
    if len(stripped) == 2:
        return stripped.lower()
    return _LANGUAGE_ISO2.get(stripped.lower(), stripped)


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

        item_id = item["question_id"]
        if item_id in id_map:
            raise ValueError(f"{kind} file contains duplicate id: {item_id}")

        id_map[item_id] = item

    return id_map


def _join_answers(answers) -> str:
    """
    answers in the JSON is a list of strings (one per sub-question).
    Join them with a newline so metric functions receive a single string.
    """
    if isinstance(answers, list):
        return "\n".join(str(a) for a in answers)
    return str(answers)


def load_input_data(gold_path: str, pred_path: str) -> List[Dict]:
    gold_items = _read_list_or_data(gold_path, "Gold")
    pred_items = _read_list_or_data(pred_path, "Prediction")

    gold_map = _build_id_map(
        gold_items,
        {"question_id", "answers"},
        "Gold",
    )
    # Prediction file only needs question_id + answers
    pred_map = _build_id_map(
        pred_items,
        {"question_id", "answers"},
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
        item_id = item["question_id"]

        merged_item = {
            "question_id": item_id,
            # Raw lists kept for potential future use
            "gold_answers": gold_map[item_id]["answers"],
            "pred_answers": pred_map[item_id]["answers"],
            # Pre-joined strings used by all metric functions
            "gold_answers_str": _join_answers(gold_map[item_id]["answers"]),
            "pred_answers_str": _join_answers(pred_map[item_id]["answers"]),
        }

        for key in ("language",):
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


def _group_by_language(items: List[Dict]) -> Dict[str, List[Dict]]:
    """Partition items by their 'language' field. Items without one go to 'unknown'."""
    groups: Dict[str, List[Dict]] = {}
    for it in items:
        lang = it.get("language", "unknown") or "unknown"
        groups.setdefault(lang, []).append(it)
    return groups


def step_bleu(items: List[Dict]):
    """Returns (overall_dict, {lang: overall_dict})."""
    groups = _group_by_language(items)
    lang_scores = {}
    all_b1, all_b2, all_b3, all_b4 = [], [], [], []

    for lang, subset in groups.items():
        b1, b2, b3, b4 = [], [], [], []
        for it in tqdm(subset, desc=f"BLEU-1..4 [{lang}]"):
            hyp, ref = it["pred_answers_str"], it["gold_answers_str"]
            b1.append(sentence_bleu_n(hyp, ref, 1))
            b2.append(sentence_bleu_n(hyp, ref, 2))
            b3.append(sentence_bleu_n(hyp, ref, 3))
            b4.append(sentence_bleu_n(hyp, ref, 4))
        all_b1.extend(b1); all_b2.extend(b2); all_b3.extend(b3); all_b4.extend(b4)
        s1, s2, s3, s4 = avg(b1), avg(b2), avg(b3), avg(b4)
        lang_scores[lang] = {"bleu-1": s1, "bleu-2": s2, "bleu-3": s3, "bleu-4": s4,
                             "bleu_avg": avg([s1, s2, s3, s4])}

    s1, s2, s3, s4 = avg(all_b1), avg(all_b2), avg(all_b3), avg(all_b4)
    overall = {"bleu-1": s1, "bleu-2": s2, "bleu-3": s3, "bleu-4": s4,
               "bleu_avg": avg([s1, s2, s3, s4])}
    return overall, lang_scores


def step_rouge(items: List[Dict]):
    """Returns (overall_dict, {lang: dict})."""
    groups = _group_by_language(items)
    lang_scores = {}
    all_r1, all_r2, all_rl = [], [], []

    for lang, subset in tqdm(groups.items(), desc="ROUGE-1/2/L (languages)"):
        r1, r2, rl = [], [], []
        for it in subset:
            hyp, ref = it["pred_answers_str"], it["gold_answers_str"]
            r = sentence_rouge_scores(hyp, ref)
            r1.append(r["rouge-1"]); r2.append(r["rouge-2"]); rl.append(r["rouge-l"])
        all_r1.extend(r1); all_r2.extend(r2); all_rl.extend(rl)
        lang_scores[lang] = {"rouge-1": avg(r1), "rouge-2": avg(r2), "rouge-l": avg(rl)}

    overall = {"rouge-1": avg(all_r1), "rouge-2": avg(all_r2), "rouge-l": avg(all_rl)}
    return overall, lang_scores


def step_meteor(items: List[Dict]):
    """Returns (overall_float, {lang: float})."""
    groups = _group_by_language(items)
    lang_scores = {}
    all_scores = []

    for lang, subset in tqdm(groups.items(), desc="METEOR (languages)"):
        scores = [sentence_meteor(it["pred_answers_str"], it["gold_answers_str"]) for it in subset]
        all_scores.extend(scores)
        lang_scores[lang] = avg(scores)

    return avg(all_scores), lang_scores


def step_comet(items: List[Dict], batch_size: int = 64):
    """
    Computes COMET score for open-ended QA.

    wmt22-comet-da requires (src, mt, ref). Since the dataset does not
    include the question text, we use the gold answer string as `src`
    (a common reference-guided proxy when no source sentence is available).
      src = gold answers (reference as source proxy)
      mt  = predicted answers
      ref = gold answers

    Returns (overall_float, {lang: float}).
    """
    model = load_comet_model("Unbabel/wmt22-comet-da")

    # Score every item in one batched pass
    srcs = [it["gold_answers_str"] for it in items]  # src = gold, no question text available
    mts  = [it["pred_answers_str"] for it in items]
    refs = [it["gold_answers_str"] for it in items]

    all_scores = []
    for s in tqdm(range(0, len(items), batch_size), desc="COMET"):
        e = s + batch_size
        all_scores.extend(
            comet_scores_batch(model, srcs[s:e], mts[s:e], refs[s:e], batch_size=batch_size)
        )

    # Per-language averages using the same ordering as `items`
    lang_buckets: Dict[str, List[float]] = {}
    for it, sc in zip(items, all_scores):
        lang = it.get("language", "unknown") or "unknown"
        lang_buckets.setdefault(lang, []).append(sc)

    lang_scores = {lang: avg(scores) for lang, scores in lang_buckets.items()}
    return avg(all_scores), lang_scores


def evaluate_openqa(items: List[Dict], batch_size_comet: int = 64) -> Dict:
    ensure_nltk_meteor_resources()

    bleu_overall, bleu_lang   = step_bleu(items)
    rouge_overall, rouge_lang = step_rouge(items)
    meteor_overall, meteor_lang = step_meteor(items)
    comet_overall, comet_lang   = step_comet(items, batch_size=batch_size_comet)

    # Flat comet keys as requested: comet_overall, comet_{lang}
    comet_section: Dict = {"comet_overall": comet_overall}
    for lang, score in sorted(comet_lang.items()):
        comet_section[f"comet_{normalise_language(lang)}"] = score

    # Nested per-language for other metrics
    langs = sorted(set(list(bleu_lang) + list(rouge_lang) + list(meteor_lang)))
    per_language = {
        lang: {
            "bleu_avg": bleu_lang.get(lang, {}).get("bleu_avg", 0.0),
            "rouge_l":  rouge_lang.get(lang, {}).get("rouge-l", 0.0),
            "meteor":   meteor_lang.get(lang, 0.0),
        }
        for lang in langs
    }

    report = {
        "num_samples": len(items),
        "bleu_avg":    bleu_overall["bleu_avg"],
        "rouge_l":     rouge_overall["rouge-l"],
        "meteor":      meteor_overall,
        **comet_section,
        "per_language": per_language,
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
        default="scores.json",
        help="Path to write the metrics report JSON. Defaults to 'metrics.json' in the current working directory.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print the metrics report to stdout after writing the file.",
    )
    args = ap.parse_args()

    items = load_input_data(args.gold_file, args.pred_file)

    report = evaluate_openqa(items, batch_size_comet=args.batch_size_comet)

    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    ensure_outdir(out_dir)
    atomic_write_json(args.out_file, report)

    if args.verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Metrics written to: {os.path.abspath(args.out_file)}")


if __name__ == "__main__":
    main()
