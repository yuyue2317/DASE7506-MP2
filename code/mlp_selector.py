"""Small MLP selector with NumPy-only checkpoint inference."""

import json

import numpy as np

from fusion_selector import agreement_features, choose
from relative_selector import make_features


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("Please supply a checkpoint directory.")

    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )

    with np.load(checkpoint / "model.npz", allow_pickle=False) as model:
        features = make_features(arrays["embeddings"])
        normalized = (
            features - model["mean"]
        ) / model["scale"]

        hidden = np.maximum(
            normalized @ model["weight0"] + model["bias0"],
            0.0,
        )
        scores = (
            hidden @ model["weight1"] + model["bias1"]
        )[..., 0]

    agreement, valid = agreement_features(bank)
    choices = choose(scores, agreement, valid, config["alpha"])

    return [int(index) for index in choices]