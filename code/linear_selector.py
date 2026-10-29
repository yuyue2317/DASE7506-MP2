"""Linear candidate selector with NumPy-only inference."""

import numpy as np


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError("Please supply --checkpoint runs/linear")

    with np.load(checkpoint / "model.npz", allow_pickle=False) as model:
        mean = model["mean"]
        scale = model["scale"]
        weight = model["weight"]
        bias = float(model["bias"].item())

    embeddings = arrays["embeddings"]
    scores = ((embeddings - mean) / scale) @ weight + bias

    return [int(index) for index in scores.argmax(axis=1)]