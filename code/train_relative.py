"""Train a relative-feature model using disagreement training groups."""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analyze_difficulty import get_groups
from common import load_inputs
from fusion_selector import agreement_features, choose
from relative_selector import make_features
from train_linear import make_labels


def main():
    train_bank, train_arrays = load_inputs("train")
    val_bank, val_arrays = load_inputs("validation")

    train_mask = get_groups(train_bank)["answer_disagreement"]
    val_mask = get_groups(val_bank)["answer_disagreement"]

    train_labels = make_labels(train_bank, "train")
    val_labels = make_labels(val_bank, "validation")

    train_features = make_features(
        train_arrays["embeddings"][train_mask]
    )
    val_features = make_features(val_arrays["embeddings"])
    dimension = train_features.shape[-1]

    train_x = train_features.reshape(-1, dimension)
    train_y = train_labels[train_mask].ravel()
    val_x = val_features.reshape(-1, dimension)

    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_x)
    val_z = scaler.transform(val_x)

    agreement, valid = agreement_features(val_bank)
    rows = np.arange(len(val_bank))

    output = Path("runs/relative")
    output.mkdir(parents=True, exist_ok=True)

    results = []
    best_key = None
    best_config = None

    for c in [0.001, 0.01, 0.1, 1.0]:
        model = LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=2000,
            random_state=17,
        )
        model.fit(train_z, train_y)
        scores = model.decision_function(val_z).reshape(-1, 4)

        for alpha in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
            choices = choose(scores, agreement, valid, alpha)
            correct = val_labels[rows, choices].astype(bool)

            full_correct = int(correct.sum())
            hard_correct = int(correct[val_mask].sum())

            result = {
                "C": c,
                "alpha": alpha,
                "full_correct": full_correct,
                "full_accuracy": 100.0 * full_correct / len(val_bank),
                "disagreement_correct": hard_correct,
                "disagreement_accuracy":
                    100.0 * hard_correct / int(val_mask.sum()),
                "iterations": int(model.n_iter_[0]),
            }
            results.append(result)
            print(json.dumps(result), flush=True)

            key = (hard_correct, full_correct)
            if best_key is None or key > best_key:
                best_key = key
                best_config = {
                    "C": c,
                    "alpha": alpha,
                    "feature_mode": "original_plus_centered",
                    "training_subset": "answer_disagreement",
                    "mask_unparseable": True,
                    "tie_break": "first_candidate",
                    "seed": 17,
                }

                np.savez(
                    output / "model.npz",
                    mean=scaler.mean_,
                    scale=scaler.scale_,
                    weight=model.coef_[0],
                    bias=model.intercept_,
                )

    (output / "config.json").write_text(
        json.dumps(best_config, indent=2),
        encoding="utf-8",
    )
    (output / "validation_experiments.json").write_text(
        json.dumps({
            "selection_split": "validation",
            "training_questions": int(train_mask.sum()),
            "experiments": results,
            "selected_config": best_config,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nSelected configuration:")
    print(json.dumps(best_config, indent=2))
    print(f"Best disagreement correct: {best_key[0]}")
    print(f"Best full correct: {best_key[1]}")


if __name__ == "__main__":
    main()