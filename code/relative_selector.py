"""Candidate selector using original and within-question features."""

import json

import numpy as np

from fusion_selector import agreement_features, choose


def make_features(embeddings):
    relative = embeddings - embeddings.mean(axis=1, keepdims=True)
    return np.concatenate([embeddings, relative], axis=-1)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("Please supply a checkpoint directory.")

    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )

    with np.load(checkpoint / "model.npz", allow_pickle=False) as model:
        mean = model["mean"]
        scale = model["scale"]
        weight = model["weight"]
        bias = float(model["bias"].item())

    features = make_features(arrays["embeddings"])
    scores = ((features - mean) / scale) @ weight + bias
    agreement, valid = agreement_features(bank)
    choices = choose(scores, agreement, valid, config["alpha"])

    return [int(index) for index in choices]