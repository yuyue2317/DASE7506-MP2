"""Analyze training composition and validation performance; no test access."""

import json
from pathlib import Path

import numpy as np

from baseline_vote import predict as vote_predict
from common import load_inputs, parse_answer
from fusion_selector import predict as fusion_predict
from linear_selector import predict as linear_predict
from train_linear import make_labels


def get_groups(bank):
    distinct_counts = []
    parseable_counts = []

    for row in bank:
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]
        valid = [answer for answer in answers if answer is not None]
        distinct_counts.append(len(set(valid)))
        parseable_counts.append(len(valid))

    distinct_counts = np.asarray(distinct_counts)
    parseable_counts = np.asarray(parseable_counts)

    return {
        "all": np.ones(len(bank), dtype=bool),
        "answer_disagreement": distinct_counts >= 2,
        "one_distinct_valid_answer": distinct_counts == 1,
        "no_valid_answer": distinct_counts == 0,
        "has_unparseable_candidate": parseable_counts < 4,
    }


def main():
    train_bank, _ = load_inputs("train")
    val_bank, val_arrays = load_inputs("validation")

    train_groups = get_groups(train_bank)
    val_groups = get_groups(val_bank)

    composition = {
        "train": {
            name: int(mask.sum())
            for name, mask in train_groups.items()
        },
        "validation": {
            name: int(mask.sum())
            for name, mask in val_groups.items()
        },
    }

    print("Data composition:")
    print(json.dumps(composition, indent=2))

    labels = make_labels(val_bank, "validation")
    rows = np.arange(len(val_bank))

    methods = {
        "majority_vote": vote_predict(val_bank, val_arrays, None),
        "linear": linear_predict(
            val_bank, val_arrays, Path("runs/linear")
        ),
        "fusion": fusion_predict(
            val_bank, val_arrays, Path("runs/fusion")
        ),
    }

    results = {}
    for method, choices in methods.items():
        choices = np.asarray(choices, dtype=np.int64)
        correct = labels[rows, choices].astype(bool)
        results[method] = {}

        for group, mask in val_groups.items():
            n = int(mask.sum())
            count = int(correct[mask].sum())
            results[method][group] = {
                "correct": count,
                "n": n,
                "accuracy": 100.0 * count / n if n else None,
            }

    report = {
        "group_definition":
            "Disagreement means at least two distinct non-None "
            "answers from the official parser.",
        "composition": composition,
        "validation_results": results,
    }

    output = Path("runs/analysis")
    output.mkdir(parents=True, exist_ok=True)
    (output / "difficulty-analysis.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nValidation results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()