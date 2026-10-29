"""Small bounded MLP experiment with question-level three-fold CV."""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from analyze_difficulty import get_groups
from common import load_inputs
from fusion_selector import agreement_features, choose
from relative_selector import make_features
from train_linear import make_labels


WIDTHS = [32, 64]
ALPHAS = [0.0, 0.25, 0.5]


def fit_model(embeddings, labels, width):
    features = make_features(embeddings).astype(np.float64)
    dimension = features.shape[-1]

    scaler = StandardScaler()
    x = scaler.fit_transform(features.reshape(-1, dimension))

    model = MLPClassifier(
        hidden_layer_sizes=(width,),
        activation="relu",
        solver="adam",
        alpha=1.0,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=150,
        early_stopping=False,
        tol=0.0001,
        n_iter_no_change=15,
        random_state=17,
    )
    model.fit(x, labels.ravel())

    return scaler, model


def get_scores(embeddings, scaler, model):
    features = make_features(embeddings).astype(np.float64)
    normalized = (
        features - scaler.mean_
    ) / scaler.scale_

    hidden = np.maximum(
        normalized @ model.coefs_[0] + model.intercepts_[0],
        0.0,
    )
    return (
        hidden @ model.coefs_[1] + model.intercepts_[1]
    )[..., 0]


def main():
    start = time.perf_counter()

    bank, arrays = load_inputs("train")
    labels = make_labels(bank, "train")
    indices = np.flatnonzero(
        get_groups(bank)["answer_disagreement"]
    )

    embeddings = arrays["embeddings"][indices]
    labels = labels[indices]

    agreement, valid = agreement_features(bank)
    agreement = agreement[indices]
    valid = valid[indices]

    folds = list(KFold(
        n_splits=3,
        shuffle=True,
        random_state=17,
    ).split(indices))

    best_correct = -1
    best_config = None
    experiments = []

    for width in WIDTHS:
        fold_scores = []
        training_records = []

        for fold, (fit_indices, dev_indices) in enumerate(
            folds, start=1
        ):
            print(
                f"Training width={width}, fold={fold}...",
                flush=True,
            )

            scaler, model = fit_model(
                embeddings[fit_indices],
                labels[fit_indices],
                width,
            )

            scores = get_scores(
                embeddings[dev_indices], scaler, model
            )
            fold_scores.append(scores)
            training_records.append({
                "fold": fold,
                "iterations": int(model.n_iter_),
                "training_loss": float(model.loss_),
                "reached_iteration_limit":
                    bool(model.n_iter_ >= model.max_iter),
            })

        for alpha in ALPHAS:
            counts = []

            for (_, dev_indices), scores in zip(folds, fold_scores):
                choices = choose(
                    scores,
                    agreement[dev_indices],
                    valid[dev_indices],
                    alpha,
                )
                counts.append(int(
                    labels[dev_indices, choices].sum()
                ))

            correct = sum(counts)
            record = {
                "width": width,
                "alpha": alpha,
                "fold_correct": counts,
                "correct": correct,
                "n": len(indices),
                "accuracy": 100.0 * correct / len(indices),
                "training": training_records,
            }
            experiments.append(record)
            print(json.dumps(record), flush=True)

            # Ties keep the earlier configuration.
            if correct > best_correct:
                best_correct = correct
                best_config = {
                    "width": width,
                    "alpha": alpha,
                    "weight_regularization": 1.0,
                    "feature_mode": "original_plus_centered",
                    "training_subset": "answer_disagreement",
                    "seed": 17,
                    "selection": "three_fold_training_question_cv",
                    "max_iter": 150,
                    "early_stopping": False,
                }

    print("\nRefitting selected configuration...", flush=True)
    scaler, model = fit_model(
        embeddings, labels, best_config["width"]
    )

    # Check our exported forward pass against sklearn probabilities.
    check_scores = get_scores(embeddings[:8], scaler, model)
    check_features = make_features(
        embeddings[:8]
    ).astype(np.float64).reshape(-1, 2048)
    sklearn_probability = model.predict_proba(
        scaler.transform(check_features)
    )[:, 1]
    exported_probability = np.exp(
        -np.logaddexp(0.0, -check_scores.ravel())
    )
    np.testing.assert_allclose(
        exported_probability,
        sklearn_probability,
        rtol=1e-7,
        atol=1e-9,
    )

    output = Path("runs/mlp")
    output.mkdir(parents=True, exist_ok=True)

    np.savez(
        output / "model.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        weight0=model.coefs_[0],
        bias0=model.intercepts_[0],
        weight1=model.coefs_[1],
        bias1=model.intercepts_[1],
    )

    (output / "config.json").write_text(
        json.dumps(best_config, indent=2),
        encoding="utf-8",
    )

    record = {
        "config": best_config,
        "experiments": experiments,
        "best_out_of_fold_correct": best_correct,
        "baseline_out_of_fold_correct": 1699,
        "training_question_ids": [
            bank[int(i)]["id"] for i in indices
        ],
        "final_iterations": int(model.n_iter_),
        "final_reached_iteration_limit":
            bool(model.n_iter_ >= model.max_iter),
        "export_forward_check": "passed",
        "complete_training_search_seconds":
            time.perf_counter() - start,
    }

    (output / "training_results.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    print("\nSelected configuration:")
    print(json.dumps(best_config, indent=2))
    print(f"MLP CV: {best_correct}/{len(indices)}")
    print(f"Existing baseline CV: 1699/{len(indices)}")
    print("Export forward check: passed")


if __name__ == "__main__":
    main()