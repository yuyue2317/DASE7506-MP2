"""Combine embedding scores with answer agreement."""

import json

import numpy as np

from common import parse_answer


def agreement_features(bank):
    agreement = np.zeros((len(bank), 4), dtype=np.float64)
    valid = np.zeros((len(bank), 4), dtype=bool)

    for i, row in enumerate(bank):
        answers = [
            parse_answer(candidate["text"])
            for candidate in row["candidates"]
        ]

        for j, answer in enumerate(answers):
            if answer is not None:
                valid[i, j] = True
                agreement[i, j] = sum(
                    other == answer
                    for k, other in enumerate(answers)
                    if k != j
                )

    return agreement, valid


def linear_scores(arrays, checkpoint):
    with np.load(checkpoint / "model.npz", allow_pickle=False) as model:
        mean = model["mean"]
        scale = model["scale"]
        weight = model["weight"]
        bias = float(model["bias"].item())

    return ((arrays["embeddings"] - mean) / scale) @ weight + bias


def choose(scores, agreement, valid, alpha):
    combined = scores + alpha * agreement

    # An unparseable answer cannot earn credit under the official scorer.
    combined = np.where(valid, combined, -np.inf)

    # If every answer is unparseable, argmax returns index 0.
    return combined.argmax(axis=1)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("Please supply --checkpoint runs/fusion")

    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )
    scores = linear_scores(arrays, checkpoint)
    agreement, valid = agreement_features(bank)
    choices = choose(scores, agreement, valid, config["alpha"])

    return [int(index) for index in choices]