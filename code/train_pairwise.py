"""Pairwise linear ranking with question-level three-fold CV."""

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from analyze_difficulty import get_groups
from common import load_inputs
from fusion_selector import agreement_features, choose
from train_linear import make_labels


REGULARIZATIONS = [0.1, 0.01, 0.001]
ALPHAS = [0.0, 0.25, 0.5, 1.0]


def fit_pairwise(embeddings, labels, valid, regularization):
    dimension = embeddings.shape[-1]
    scaler = StandardScaler()

    features = scaler.fit_transform(
        embeddings.reshape(-1, dimension)
    ).reshape(embeddings.shape)

    differences = []
    pair_weights = []

    for x, y, mask in zip(features, labels, valid):
        positives = np.flatnonzero((y == 1) & mask)
        negatives = np.flatnonzero((y == 0) & mask)

        if not len(positives) or not len(negatives):
            raise ValueError("Expected both correct and wrong valid answers.")

        # Every question contributes equal total weight,
        # regardless of its number of correct/wrong pairs.
        count = len(positives) * len(negatives)

        for positive in positives:
            for negative in negatives:
                differences.append(x[positive] - x[negative])
                pair_weights.append(1.0 / count)

    differences = np.asarray(differences, dtype=np.float64)
    pair_weights = np.asarray(pair_weights, dtype=np.float64)
    pair_weights /= pair_weights.sum()

    def objective(weight):
        margins = differences @ weight

        loss = np.dot(
            pair_weights, np.logaddexp(0.0, -margins)
        )
        loss += 0.5 * regularization * np.dot(weight, weight)

        gradient = differences.T @ (
            -pair_weights * expit(-margins)
        )
        gradient += regularization * weight

        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(dimension, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not result.success:
        raise RuntimeError(str(result.message))

    return scaler, result.x, {
        "iterations": int(result.nit),
        "pairs": len(differences),
        "objective": float(result.fun),
    }


def get_scores(embeddings, scaler, weight):
    return ((embeddings - scaler.mean_) / scaler.scale_) @ weight


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

    # Same question-level folds as the pooling experiment.
    folds = list(KFold(
        n_splits=3,
        shuffle=True,
        random_state=17,
    ).split(indices))

    experiments = []
    best_correct = -1
    best_config = None

    for regularization in REGULARIZATIONS:
        fold_scores = []
        optimization_records = []

        for fit_indices, dev_indices in folds:
            scaler, weight, optimization = fit_pairwise(
                embeddings[fit_indices],
                labels[fit_indices],
                valid[fit_indices],
                regularization,
            )
            scores = get_scores(
                embeddings[dev_indices], scaler, weight
            )
            fold_scores.append(scores)
            optimization_records.append(optimization)

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
                "regularization": regularization,
                "alpha": alpha,
                "fold_correct": counts,
                "correct": correct,
                "n": len(indices),
                "accuracy": 100.0 * correct / len(indices),
                "optimization": optimization_records,
            }
            experiments.append(record)
            print(json.dumps(record), flush=True)

            if correct > best_correct:
                best_correct = correct
                best_config = {
                    "regularization": regularization,
                    "alpha": alpha,
                    "objective": "pairwise_logistic",
                    "features": "original_embeddings",
                    "question_weighting": "equal",
                    "training_subset": "answer_disagreement",
                    "seed": 17,
                    "selection": "three_fold_training_question_cv",
                }

    print("\nRefitting selected configuration...")
    scaler, weight, optimization = fit_pairwise(
        embeddings, labels, valid,
        best_config["regularization"],
    )

    output = Path("runs/pairwise")
    output.mkdir(parents=True, exist_ok=True)

    # Compatible with the existing fusion_selector.py.
    np.savez(
        output / "model.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        weight=weight,
        bias=np.asarray([0.0]),
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
        "final_optimization": optimization,
        "complete_training_search_seconds":
            time.perf_counter() - start,
    }
    (output / "training_results.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    print("\nSelected configuration:")
    print(json.dumps(best_config, indent=2))
    print(f"Pairwise CV: {best_correct}/{len(indices)}")
    print(f"Existing baseline CV: 1699/{len(indices)}")


if __name__ == "__main__":
    main()