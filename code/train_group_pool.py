"""Choose pooling using three-fold CV on training questions only."""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from analyze_difficulty import get_groups
from common import load_inputs
from group_selector import select_groups
from relative_selector import make_features
from train_linear import make_labels


MODES = ["baseline", "sum", "mean"]


def fit_model(embeddings, labels):
    features = make_features(embeddings)
    dimension = features.shape[-1]

    scaler = StandardScaler()
    x = scaler.fit_transform(features.reshape(-1, dimension))

    model = LogisticRegression(
        C=0.001,
        solver="lbfgs",
        max_iter=2000,
        random_state=17,
    )
    model.fit(x, labels.ravel())

    if int(model.n_iter_[0]) >= 2000:
        raise RuntimeError("Model reached the iteration limit.")

    return scaler, model


def get_scores(embeddings, scaler, model):
    features = make_features(embeddings)
    dimension = features.shape[-1]
    x = scaler.transform(features.reshape(-1, dimension))
    return model.decision_function(x).reshape(-1, 4)


def main():
    start = time.perf_counter()

    bank, arrays = load_inputs("train")
    labels = make_labels(bank, "train")

    hard_indices = np.flatnonzero(
        get_groups(bank)["answer_disagreement"]
    )
    hard_bank = [bank[int(i)] for i in hard_indices]
    embeddings = arrays["embeddings"][hard_indices]
    labels = labels[hard_indices]

    output = Path("runs/group_pool")
    output.mkdir(parents=True, exist_ok=True)

    totals = {mode: 0 for mode in MODES}
    fold_records = []

    splitter = KFold(
        n_splits=3,
        shuffle=True,
        random_state=17,
    )

    for fold, (fit_indices, dev_indices) in enumerate(
        splitter.split(hard_indices), start=1
    ):
        scaler, model = fit_model(
            embeddings[fit_indices],
            labels[fit_indices],
        )
        scores = get_scores(
            embeddings[dev_indices], scaler, model
        )
        dev_bank = [hard_bank[int(i)] for i in dev_indices]

        fold_result = {
            "fold": fold,
            "fitting_questions": len(fit_indices),
            "development_questions": len(dev_indices),
            "iterations": int(model.n_iter_[0]),
            "results": {},
        }

        for mode in MODES:
            choices = np.asarray(
                select_groups(dev_bank, scores, mode),
                dtype=np.int64,
            )
            correct = int(
                labels[dev_indices, choices].sum()
            )
            totals[mode] += correct

            fold_result["results"][mode] = {
                "correct": correct,
                "accuracy": 100.0 * correct / len(dev_indices),
            }

        fold_records.append(fold_result)
        print(json.dumps(fold_result), flush=True)

    summary = {
        mode: {
            "correct": totals[mode],
            "n": len(hard_indices),
            "accuracy": 100.0 * totals[mode] / len(hard_indices),
        }
        for mode in MODES
    }

    # Ties retain baseline first, then sum, then mean.
    selected_mode = max(MODES, key=lambda mode: totals[mode])

    print("\nOut-of-fold results:")
    print(json.dumps(summary, indent=2))
    print(f"Selected pooling: {selected_mode}")

    # Refit using training data only.
    scaler, model = fit_model(embeddings, labels)

    np.savez(
        output / "model.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        weight=model.coef_[0],
        bias=model.intercept_,
    )

    config = {
        "pooling": selected_mode,
        "C": 0.001,
        "baseline_alpha": 0.25,
        "feature_mode": "original_plus_centered",
        "training_subset": "answer_disagreement",
        "seed": 17,
        "selection": "three_fold_training_question_cv",
    }

    (output / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    record = {
        "config": config,
        "folds": fold_records,
        "out_of_fold_results": summary,
        "training_question_ids": [
            row["id"] for row in hard_bank
        ],
        "total_training_search_seconds":
            time.perf_counter() - start,
    }

    (output / "training_results.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    print("Saved to runs/group_pool")


if __name__ == "__main__":
    main()