"""Small predefined validation search and complementary-error analysis."""

import json
import shutil
from pathlib import Path

import numpy as np

from baseline_vote import predict as vote_predict
from common import load_inputs
from fusion_selector import agreement_features, choose, linear_scores
from train_linear import make_labels


def main():
    bank, arrays = load_inputs("validation")
    labels = make_labels(bank, "validation")
    rows = np.arange(len(bank))

    scores = linear_scores(arrays, Path("runs/linear"))
    agreement, valid = agreement_features(bank)

    vote_choices = np.asarray(vote_predict(bank, arrays, None))
    linear_choices = scores.argmax(axis=1)

    vote_ok = labels[rows, vote_choices].astype(bool)
    linear_ok = labels[rows, linear_choices].astype(bool)

    comparison = {
        "both_correct": int((vote_ok & linear_ok).sum()),
        "only_vote_correct": int((vote_ok & ~linear_ok).sum()),
        "only_linear_correct": int((~vote_ok & linear_ok).sum()),
        "both_wrong": int((~vote_ok & ~linear_ok).sum()),
        "different_candidate_choices": int(
            (vote_choices != linear_choices).sum()
        ),
    }
    print("Error comparison:")
    print(json.dumps(comparison, indent=2))

    experiments = []
    best_correct = -1

    print("Fusion experiments:")
    for alpha in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        choices = choose(scores, agreement, valid, alpha)
        correct = int(labels[rows, choices].sum())

        result = {
            "alpha": alpha,
            "correct": correct,
            "n": len(bank),
            "accuracy": 100.0 * correct / len(bank),
        }
        experiments.append(result)
        print(json.dumps(result), flush=True)

        if correct > best_correct:
            best_correct = correct
            best_alpha = alpha

    output = Path("runs/fusion")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2("runs/linear/model.npz", output / "model.npz")

    config = {
        "alpha": best_alpha,
        "mask_unparseable": True,
        "tie_break": "first_candidate",
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    record = {
        "selection_split": "validation",
        "comparison": comparison,
        "experiments": experiments,
        "selected_config": config,
    }
    (output / "validation_experiments.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    print(f"Selected alpha: {best_alpha}")


if __name__ == "__main__":
    main()