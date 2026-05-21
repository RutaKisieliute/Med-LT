#!/usr/bin/env python3

# SRIPT COUNTS STATISTICS FROM PREDICTIONS
# INSTALL: pip install sentence-transformers nltk rouge-score
# RUN: python general_evaluation.py

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# CONFIGURATION
INPUT_CSV = Path(r"input_csv_file")
OUTPUT_DIR = Path(r"trained_results_dir")

# BLEU, ROUGE AND METEOR SETTINGS
MULTI_CORRECT_METRIC = "rougeL_f1"
MULTI_CORRECT_THRESHOLD = 0.5

# SEMANTIC SIMILARITY SETTINGS
ONE_WORD_SEMANTIC_THRESHOLD = 0.7
SEMANTIC_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

REQUIRED_COLUMNS = ["image", "question_lt", "correct_answer_lt", "predicted_answer_lt"]

_SEMANTIC_MODEL = None

# TEXT NORMALIZATION
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    return text.split()

# GET ANSWER TYPE
def answer_type(correct_answer: str) -> str:
    norm = normalize_text(correct_answer)
    if norm in {"taip", "ne"}:
        return "yes_no"

    tokens = tokenize(correct_answer)
    if len(tokens) == 1:
        return "one_word"

    return "multi_word"

# CHECK IF CORRECT ANSWER IS INSIDE GENERATED ANSWER
def contains_correct_answer(correct_answer: str, predicted_answer: str) -> int:
    correct_norm = normalize_text(correct_answer)
    predicted_tokens = tokenize(predicted_answer)

    if not correct_norm:
        return 0

    correct_tokens = correct_norm.split()
    if len(correct_tokens) == 1:
        return int(correct_tokens[0] in predicted_tokens)

    predicted_norm = " ".join(predicted_tokens)
    return int(correct_norm in predicted_norm)

def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

# STANDARD NLTK BLEU SCORE
def standard_bleu(reference: str, candidate: str) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError as exc:
        raise ImportError("NLTK is required for BLEU. Install it with: pip install nltk") from exc

    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if not ref_tokens or not cand_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    return float(sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing))

# STANDARD ROUGE-SCORE PACKAGE ROUGE SCORE
def standard_rouge(reference: str, candidate: str) -> Dict[str, Dict[str, float]]:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError("rouge-score is required for ROUGE. Install it with: pip install rouge-score") from exc

    reference_norm = normalize_text(reference)
    candidate_norm = normalize_text(candidate)

    if not reference_norm or not candidate_norm:
        empty = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {"rouge1": empty, "rouge2": empty, "rougeL": empty}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference_norm, candidate_norm)

    return {
        "rouge1": {
            "precision": float(scores["rouge1"].precision),
            "recall": float(scores["rouge1"].recall),
            "f1": float(scores["rouge1"].fmeasure),
        },
        "rouge2": {
            "precision": float(scores["rouge2"].precision),
            "recall": float(scores["rouge2"].recall),
            "f1": float(scores["rouge2"].fmeasure),
        },
        "rougeL": {
            "precision": float(scores["rougeL"].precision),
            "recall": float(scores["rougeL"].recall),
            "f1": float(scores["rougeL"].fmeasure),
        },
    }

# STANDARD NLTK METEOR SCORE
def standard_meteor(reference: str, candidate: str) -> float:
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError as exc:
        raise ImportError("NLTK is required for METEOR. Install it with: pip install nltk") from exc

    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if not ref_tokens or not cand_tokens:
        return 0.0

    try:
        return meteor_score([ref_tokens], cand_tokens)
    except LookupError:
        import nltk
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        return meteor_score([ref_tokens], cand_tokens)


# GETTING SEMANTIC SIMILARITY MODEL
def get_semantic_model():
    global _SEMANTIC_MODEL

    if _SEMANTIC_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for one-word semantic similarity. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        _SEMANTIC_MODEL = SentenceTransformer(SEMANTIC_MODEL_NAME)

    return _SEMANTIC_MODEL

# COUNTING SEMANTIC SIMILARITY
def semantic_similarity(reference: str, candidate: str) -> float:
    reference = normalize_text(reference)
    candidate = normalize_text(candidate)

    if not reference or not candidate:
        return 0.0

    try:
        from sentence_transformers import util
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for one-word semantic similarity. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    model = get_semantic_model()
    embeddings = model.encode([reference, candidate], convert_to_tensor=True)
    return float(util.cos_sim(embeddings[0], embeddings[1]).item())

# CHECK IF YES, NO OR SOMETHING DIFERENT WAS GENERATED
def yes_no_label_from_generated(predicted_answer: str) -> str:
    tokens = tokenize(predicted_answer)
    has_yes = "taip" in tokens
    has_no = "ne" in tokens

    if has_yes and not has_no:
        return "taip"
    if has_no and not has_yes:
        return "ne"

    return "unknown"

# CSV FUNCTIONS
def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Found columns: {reader.fieldnames}")
        return list(reader)

def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# RESULTS EVALUATION FUNCTIONS
def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0

def evaluate(input_csv: Path, output_dir: Path, multi_correct_metric: str, multi_correct_threshold: float) -> None:
    if multi_correct_metric not in {"bleu", "rouge1_f1", "rouge2_f1", "rougeL_f1", "meteor"}:
        raise ValueError("MULTI_CORRECT_METRIC must be one of: bleu, rouge1_f1, rouge2_f1, rougeL_f1, meteor")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(input_csv)

    per_answer_rows = []

    yes_no_tp = yes_no_fp = yes_no_fn = yes_no_tn = 0

    for idx, row in enumerate(rows, start=1):
        correct = row["correct_answer_lt"]
        predicted = row["predicted_answer_lt"]
        a_type = answer_type(correct)

        result = {
            "row_id": idx,
            "image": row.get("image", ""),
            "question_lt": row.get("question_lt", ""),
            "correct_answer_lt": correct,
            "predicted_answer_lt": predicted,
            "answer_type": a_type,
            "binary_correct": 0,
            "one_word_exact_match": "",
            "one_word_semantic_similarity": "",
            "one_word_correct_by_semantic": "",
            "one_word_exact_plus_semantic_correct": "",
            "bleu": "",
            "rouge1_precision": "",
            "rouge1_recall": "",
            "rouge1_f1": "",
            "rouge2_precision": "",
            "rouge2_recall": "",
            "rouge2_f1": "",
            "rougeL_precision": "",
            "rougeL_recall": "",
            "rougeL_f1": "",
            "meteor": "",
        }

        # YES / NO TYPE
        if a_type == "yes_no":
            result["binary_correct"] = contains_correct_answer(correct, predicted)

        # ONE WORD TYPE
        elif a_type == "one_word":
            exact_match = contains_correct_answer(correct, predicted)

            result["one_word_exact_match"] = exact_match

            if exact_match:
                result["one_word_semantic_similarity"] = ""
                result["one_word_correct_by_semantic"] = 0
                result["one_word_exact_plus_semantic_correct"] = 1
                result["binary_correct"] = 1
            else:
                similarity = semantic_similarity(correct, predicted)
                semantic_correct = int(similarity >= ONE_WORD_SEMANTIC_THRESHOLD)

                result["one_word_semantic_similarity"] = similarity
                result["one_word_correct_by_semantic"] = semantic_correct
                result["one_word_exact_plus_semantic_correct"] = semantic_correct
                result["binary_correct"] = semantic_correct

        # MULTI WORDS TYPE
        else:
            bleu = standard_bleu(correct, predicted)
            rouge_scores = standard_rouge(correct, predicted)
            r1 = rouge_scores["rouge1"]
            r2 = rouge_scores["rouge2"]
            rl = rouge_scores["rougeL"]
            meteor = standard_meteor(correct, predicted)

            metric_values = {
                "bleu": bleu,
                "rouge1_f1": r1["f1"],
                "rouge2_f1": r2["f1"],
                "rougeL_f1": rl["f1"],
                "meteor": meteor,
            }

            result["binary_correct"] = int(metric_values[multi_correct_metric] >= multi_correct_threshold)

            result.update({
                "bleu": bleu,
                "rouge1_precision": r1["precision"],
                "rouge1_recall": r1["recall"],
                "rouge1_f1": r1["f1"],
                "rouge2_precision": r2["precision"],
                "rouge2_recall": r2["recall"],
                "rouge2_f1": r2["f1"],
                "rougeL_precision": rl["precision"],
                "rougeL_recall": rl["recall"],
                "rougeL_f1": rl["f1"],
                "meteor": meteor,
            })

        # YES / NO TYPE
        if a_type == "yes_no":
            true_label = normalize_text(correct)
            pred_label = yes_no_label_from_generated(predicted)

            if true_label == "taip" and pred_label == "taip":
                yes_no_tp += 1
            elif true_label == "ne" and pred_label == "taip":
                yes_no_fp += 1
            elif true_label == "taip" and pred_label == "ne":
                yes_no_fn += 1
            elif true_label == "ne" and pred_label == "ne":
                yes_no_tn += 1
            else:
                if true_label == "taip":
                    yes_no_fn += 1
                elif true_label == "ne":
                    yes_no_fp += 1

        per_answer_rows.append(result)

    # RESULTS GENERATION
    per_answer_fields = [
        "row_id", "image", "question_lt", "correct_answer_lt", "predicted_answer_lt",
        "answer_type", "binary_correct",
        "one_word_exact_match",
        "one_word_semantic_similarity",
        "one_word_correct_by_semantic",
        "one_word_exact_plus_semantic_correct",
        "bleu", "rouge1_precision", "rouge1_recall", "rouge1_f1",
        "rouge2_precision", "rouge2_recall", "rouge2_f1",
        "rougeL_precision", "rougeL_recall", "rougeL_f1", "meteor",
    ]

    write_csv(output_dir / "general_per_answer_results.csv", per_answer_rows, per_answer_fields)

    total = len(per_answer_rows)
    correct_total = sum(int(r["binary_correct"]) for r in per_answer_rows)

    by_type = {}
    for a_type in ["yes_no", "one_word", "multi_word"]:
        subset = [r for r in per_answer_rows if r["answer_type"] == a_type]
        by_type[a_type] = {
            "count": len(subset),
            "accuracy": safe_div(sum(int(r["binary_correct"]) for r in subset), len(subset)),
        }

    yes_no_precision = safe_div(yes_no_tp, yes_no_tp + yes_no_fp)
    yes_no_recall = safe_div(yes_no_tp, yes_no_tp + yes_no_fn)
    yes_no_f1 = safe_div(2 * yes_no_precision * yes_no_recall, yes_no_precision + yes_no_recall)

    one_word_rows = [r for r in per_answer_rows if r["answer_type"] == "one_word"]

    one_word_exact_correct_count = sum(
        int(r["one_word_exact_match"])
        for r in one_word_rows
        if r["one_word_exact_match"] != ""
    )

    one_word_semantic_correct_count = sum(
        int(r["one_word_correct_by_semantic"])
        for r in one_word_rows
        if r["one_word_correct_by_semantic"] != ""
    )

    one_word_exact_plus_semantic_correct_count = sum(
        int(r["one_word_exact_plus_semantic_correct"])
        for r in one_word_rows
        if r["one_word_exact_plus_semantic_correct"] != ""
    )

    one_word_incorrect_count = len(one_word_rows) - one_word_exact_plus_semantic_correct_count

    multi_rows = [r for r in per_answer_rows if r["answer_type"] == "multi_word"]

    summary = [{
        "input_file": str(input_csv),
        "total_answers": total,
        "overall_accuracy": safe_div(correct_total, total),
        "yes_no_count": by_type["yes_no"]["count"],
        "yes_no_accuracy": by_type["yes_no"]["accuracy"],
        "one_word_count": by_type["one_word"]["count"],
        "one_word_exact_correct_count": one_word_exact_correct_count,
        "one_word_semantic_correct_count": one_word_semantic_correct_count,
        "one_word_exact_plus_semantic_correct_count": one_word_exact_plus_semantic_correct_count,
        "one_word_incorrect_count": one_word_incorrect_count,
        "one_word_accuracy_exact_only": safe_div(
            one_word_exact_correct_count,
            by_type["one_word"]["count"]
        ),
        "one_word_accuracy_exact_plus_semantic": safe_div(
            one_word_exact_plus_semantic_correct_count,
            by_type["one_word"]["count"]
        ),
        "one_word_accuracy": by_type["one_word"]["accuracy"],
        "multi_word_count": by_type["multi_word"]["count"],
        "multi_word_binary_accuracy": by_type["multi_word"]["accuracy"],
        "multi_word_correct_metric": multi_correct_metric,
        "multi_word_correct_threshold": multi_correct_threshold,
        "one_word_semantic_threshold": ONE_WORD_SEMANTIC_THRESHOLD,
        "avg_one_word_semantic_similarity": mean([
            float(r["one_word_semantic_similarity"])
            for r in per_answer_rows
            if r["answer_type"] == "one_word" and r["one_word_semantic_similarity"] != ""
        ]),
        "avg_bleu_multi_word": mean([float(r["bleu"]) for r in multi_rows if r["bleu"] != ""]),
        "avg_rouge1_f1_multi_word": mean([float(r["rouge1_f1"]) for r in multi_rows if r["rouge1_f1"] != ""]),
        "avg_rouge2_f1_multi_word": mean([float(r["rouge2_f1"]) for r in multi_rows if r["rouge2_f1"] != ""]),
        "avg_rougeL_f1_multi_word": mean([float(r["rougeL_f1"]) for r in multi_rows if r["rougeL_f1"] != ""]),
        "avg_meteor_multi_word": mean([float(r["meteor"]) for r in multi_rows if r["meteor"] != ""]),
        "yes_no_tp": yes_no_tp,
        "yes_no_fp": yes_no_fp,
        "yes_no_fn": yes_no_fn,
        "yes_no_tn": yes_no_tn,
        "yes_no_precision_for_taip": yes_no_precision,
        "yes_no_recall_for_taip": yes_no_recall,
        "yes_no_f1_for_taip": yes_no_f1,
    }]

    write_csv(output_dir / "general_summary_results.csv", summary, list(summary[0].keys()))

    print(f"Saved: {output_dir / 'general_per_answer_results.csv'}")
    print(f"Saved: {output_dir / 'general_summary_results.csv'}")

# MAIN FUNCTION
def main() -> None:
    evaluate(
        input_csv=INPUT_CSV,
        output_dir=OUTPUT_DIR,
        multi_correct_metric=MULTI_CORRECT_METRIC,
        multi_correct_threshold=MULTI_CORRECT_THRESHOLD,
    )

if __name__ == "__main__":
    main()
