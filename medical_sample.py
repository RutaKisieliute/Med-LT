#!/usr/bin/env python3

# RUN: python medical_sample.py

import csv
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, List

# CONFIGURATIONS
INPUT_CSV = Path(r"C:\\Users\\Rutos\\Desktop\\DATA2\\evaluation_scripts\\medemma\\trained_pathvqa_test_others_predictions.csv")
OUTPUT_CSV = Path(r"100\\medgemma-trained-path-other-random_answers_100.csv")

SAMPLE_SIZE = 100
RANDOM_SEED = 42

REQUIRED_COLUMNS = ["correct_answer_lt", "predicted_answer_lt"]


# TEXT NORMALIZATION FUNCTION
def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


# CSV FUNCTIONS
def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. Found columns: {reader.fieldnames}"
            )

        return list(reader)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# MAIN FUNCTION
def select_random_answers() -> None:
    rows = read_csv(INPUT_CSV)

    sample_size = min(SAMPLE_SIZE, len(rows))

    random.seed(RANDOM_SEED)
    sampled_rows = random.sample(list(enumerate(rows, start=1)), sample_size)

    selected_predicted_answers = [
        row.get("predicted_answer_lt", "")
        for _, row in sampled_rows
    ]

    total_predicted_words = sum(
        len(tokenize(answer))
        for answer in selected_predicted_answers
    )

    output_rows = []
    for original_row_id, row in sampled_rows:
        correct_answer = row.get("correct_answer_lt", "")
        predicted_answer = row.get("predicted_answer_lt", "")

        output_rows.append({
            "original_row_id": original_row_id,
            "correct_answer_lt": correct_answer,
            "predicted_answer_lt": predicted_answer,
            "predicted_word_count": len(tokenize(predicted_answer)),
        })

    write_csv(
        OUTPUT_CSV,
        output_rows,
        [
            "original_row_id",
            "correct_answer_lt",
            "predicted_answer_lt",
            "predicted_word_count",
        ]
    )

    print(f"Selected rows: {len(sampled_rows)}")
    print(f"Total predicted words in selected array: {total_predicted_words}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    select_random_answers()