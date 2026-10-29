"""Select an answer group using a fixed pooling rule."""

import json

import numpy as np

from common import parse_answer
from relative_selector import make_features


def sigmoid(scores):
    # Stable sigmoid without requiring SciPy at inference.
    return np.exp(-np.logaddexp(0.0, -scores))


def select_groups(bank, scores, mode):
    probabilities = sigmoid(scores)
    choices = []

    for i, row in enumerate(bank):
        groups = {}

        for j, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            if answer is not None:
                groups.setdefault(answer, []).append(j)

        if not groups:
            choices.append(0)
            continue

        best_group_score = -np.inf
        best_choice = 0

        # Dict insertion order gives a fixed tie rule.
        for indices in groups.values():
            if mode == "baseline":
                group_score = (
                    max(scores[i, j] for j in indices)
                    + 0.25 * (len(indices) - 1)
                )
            elif mode == "sum":
                group_score = sum(
                    probabilities[i, j] for j in indices
                )
            elif mode == "mean":
                group_score = np.mean(
                    [probabilities[i, j] for j in indices]
                )
            else:
                raise ValueError(f"Unknown pooling mode: {mode}")

            # Return the highest-scoring candidate in the group.
            representative = max(
                indices, key=lambda j: scores[i, j]
            )

            if group_score > best_group_score:
                best_group_score = group_score
                best_choice = representative

        choices.append(int(best_choice))

    return choices


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("Please supply a checkpoint directory.")

    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )

    with np.load(checkpoint / "model.npz", allow_pickle=False) as model:
        features = make_features(arrays["embeddings"])
        scores = (
            (features - model["mean"]) / model["scale"]
        ) @ model["weight"] + float(model["bias"].item())

    return select_groups(bank, scores, config["pooling"])