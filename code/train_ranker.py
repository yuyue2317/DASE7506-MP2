"""Train a listwise linear ranker; tune on an internal training split."""

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from analyze_difficulty import get_groups
from common import load_inputs
from fusion_selector import agreement_features, choose
from train_linear import make_labels


def fit_ranker(embeddings, labels, valid, regularization):
    dimension = embeddings.shape[-1]

    scaler = StandardScaler()
    flat = embeddings.reshape(-1, dimension)
    features = scaler.fit_transform(flat).reshape(embeddings.shape)

    positive = labels.astype(bool) & valid
    if not np.all(positive.any(axis=1)):
        raise ValueError("Every training group needs a correct candidate.")

    def objective(weight):
        scores = features @ weight

        valid_scores = np.where(valid, scores, -np.inf)
        positive_scores = np.where(positive, scores, -np.inf)

        denominator = logsumexp(valid_scores, axis=1)
        numerator = logsumexp(positive_scores, axis=1)

        loss = np.mean(denominator - numerator)
        loss += 0.5 * regularization * np.dot(weight, weight)

        valid_prob = np.exp(valid_scores - denominator[:, None])
        positive_prob = np.exp(positive_scores - numerator[:, None])

        gradient = np.einsum(
            "ni,nid->d", valid_prob - positive_prob, features
        ) / len(features)
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
        raise RuntimeError(f"Optimization failed: {result.message}")

    return scaler, result.x, {
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "converged": bool(result.success),
    }


def score_model(embeddings, scaler, weight):
    return ((embeddings - scaler.mean_) / scaler.scale_) @ weight


def main():
    start = time.perf_counter()

    bank, arrays = load_inputs("train")
    labels = make_labels(bank, "train")
    hard_mask = get_groups(bank)["answer_disagreement"]

    hard_indices = np.flatnonzero(hard_mask)
    fit_indices, dev_indices = train_test_split(
        hard_indices,
        test_size=0.2,
        random_state=17,
        shuffle=True,
    )

    embeddings = arrays["embeddings"]
    agreement, valid = agreement_features(bank)

    print(f"Internal fitting questions: {len(fit_indices)}")
    print(f"Internal development questions: {len(dev_indices)}")

    output = Path("runs/ranker")
    output.mkdir(parents=True, exist_ok=True)

    experiments = []
    best_correct = -1
    best_config = None

    # Predefined search; no official validation/test references used.
    for regularization in [0.1, 0.01, 0.001]:
        scaler, weight, optimization = fit_ranker(
            embeddings[fit_indices],
            labels[fit_indices],
            valid[fit_indices],
            regularization,
        )

        scores = score_model(embeddings[dev_indices], scaler, weight)

        for alpha in [0.0, 0.25, 0.5, 1.0]:
            choices = choose(
                scores,
                agreement[dev_indices],
                valid[dev_indices],
                alpha,
            )
            correct = int(
                labels[dev_indices, choices].sum()
            )

            record = {
                "regularization": regularization,
                "alpha": alpha,
                "correct": correct,
                "n": len(dev_indices),
                "accuracy": 100.0 * correct / len(dev_indices),
                **optimization,
            }
            experiments.append(record)
            print(json.dumps(record), flush=True)

            # Ties keep the earlier configuration.
            if correct > best_correct:
                best_correct = correct
                best_config = {
                    "regularization": regularization,
                    "alpha": alpha,
                    "objective": "correct_set_listwise",
                    "training_subset": "answer_disagreement",
                    "mask_unparseable": True,
                    "tie_break": "first_candidate",
                    "seed": 17,
                }

    print("\nRefitting on all disagreement training questions...")
    scaler, weight, final_optimization = fit_ranker(
        embeddings[hard_indices],
        labels[hard_indices],
        valid[hard_indices],
        best_config["regularization"],
    )

    # Same asset format as fusion_selector.py.
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
        "selection_data": "internal split of training disagreement groups",
        "fit_ids": [bank[int(i)]["id"] for i in fit_indices],
        "development_ids": [bank[int(i)]["id"] for i in dev_indices],
        "experiments": experiments,
        "selected_config": best_config,
        "final_training_questions": len(hard_indices),
        "final_optimization": final_optimization,
        "total_training_search_seconds": time.perf_counter() - start,
    }
    (output / "training_results.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    print("\nSelected configuration:")
    print(json.dumps(best_config, indent=2))
    print("Saved to runs/ranker")


if __name__ == "__main__":
    main()