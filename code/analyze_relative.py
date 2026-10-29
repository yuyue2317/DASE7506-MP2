"""Analyze the final relative model on validation only."""

import json
from pathlib import Path

import numpy as np

from common import load_inputs, parse_answer
from fusion_selector import agreement_features
from relative_selector import make_features
from train_linear import make_labels


def main():
    bank, arrays = load_inputs("validation")
    labels = make_labels(bank, "validation")
    checkpoint = Path("runs/relative")

    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )

    with np.load(checkpoint / "model.npz", allow_pickle=False) as model:
        features = make_features(arrays["embeddings"])
        scores = (
            (features - model["mean"]) / model["scale"]
        ) @ model["weight"] + float(model["bias"].item())

    agreement, valid = agreement_features(bank)
    scores = np.where(
        valid, scores + config["alpha"] * agreement, -np.inf
    )
    choices = scores.argmax(axis=1)

    records = []
    for i, row in enumerate(bank):
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]
        distinct = set(answer for answer in answers if answer is not None)
        choice = int(choices[i])

        # Compare the best score for each distinct final answer.
        # Comparing candidate scores alone can confuse duplicate answers
        # with genuine disagreement between answers.
        answer_scores = {}
        for j, answer in enumerate(answers):
            if answer is not None:
                answer_scores[answer] = max(
                    answer_scores.get(answer, -np.inf),
                    float(scores[i, j]),
                )

        ranked_scores = sorted(answer_scores.values(), reverse=True)
        margin = (
            ranked_scores[0] - ranked_scores[1]
            if len(ranked_scores) >= 2 else None
        )

        records.append({
            "id": row["id"],
            "question": row["question"],
            "candidate_answers": answers,
            "correct_candidate_indices": [
                j for j in range(4) if labels[i, j] == 1
            ],
            "selected_index": choice,
            "correct": bool(labels[i, choice]),
            "distinct_valid_answers": len(distinct),
            "answer_margin": margin,
            "candidate_scores": [
                float(value) if np.isfinite(value) else None
                for value in scores[i]
            ],
            "candidates": row["candidates"],
        })

    hard = [
        record for record in records
        if record["distinct_valid_answers"] >= 2
    ]
    ranked = sorted(hard, key=lambda record: record["answer_margin"])

    confidence_groups = []
    for name, indices in zip(
        ["lowest_margin", "middle_low", "middle_high", "highest_margin"],
        np.array_split(np.arange(len(ranked)), 4),
    ):
        group = [ranked[int(index)] for index in indices]
        confidence_groups.append({
            "group": name,
            "n": len(group),
            "correct": sum(record["correct"] for record in group),
            "accuracy": (
                100 * sum(record["correct"] for record in group) / len(group)
                if group else None
            ),
            "minimum_margin": (
                min(record["answer_margin"] for record in group)
                if group else None
            ),
            "maximum_margin": (
                max(record["answer_margin"] for record in group)
                if group else None
            ),
        })

    errors = [record for record in records if not record["correct"]]
    high_margin_errors = sorted(
        [
            record for record in errors
            if record["answer_margin"] is not None
        ],
        key=lambda record: record["answer_margin"],
        reverse=True,
    )[:5]

    summary = {
        "split": "validation",
        "correct": sum(record["correct"] for record in records),
        "n": len(records),
        "errors": len(errors),
        "disagreement_correct": sum(record["correct"] for record in hard),
        "disagreement_n": len(hard),
        "confidence_groups_on_disagreement_questions": confidence_groups,
    }

    output = Path("runs/relative_analysis")
    output.mkdir(parents=True, exist_ok=True)

    for filename, content in [
        ("summary.json", summary),
        ("validation-errors.json", errors),
        ("high-margin-errors.json", high_margin_errors),
    ]:
        (output / filename).write_text(
            json.dumps(content, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2))
    print("\nSaved analysis to runs/relative_analysis/")


if __name__ == "__main__":
    main()