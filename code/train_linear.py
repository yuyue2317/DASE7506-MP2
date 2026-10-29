"""Train on training references; select regularization on validation."""

import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import DATA, load_inputs, parse_answer, read_jsonl


def make_labels(bank, split):
    references = read_jsonl(DATA / f"gsm8k_{split}.jsonl")
    assert len(bank) == len(references)

    labels = []
    for row, reference in zip(bank, references):
        assert row["id"] == reference["id"]
        target = parse_answer(reference["answer"])
        assert target is not None

        labels.append([
            int(
                (answer := parse_answer(candidate["text"])) is not None
                and answer == target
            )
            for candidate in row["candidates"]
        ])

    return np.asarray(labels, dtype=np.int64)


def main():
    output = Path("runs/linear")
    output.mkdir(parents=True, exist_ok=True)

    train_bank, train_arrays = load_inputs("train")
    val_bank, val_arrays = load_inputs("validation")

    train_y = make_labels(train_bank, "train")
    val_y = make_labels(val_bank, "validation")

    train_x = train_arrays["embeddings"].reshape(-1, 1024)
    val_x = val_arrays["embeddings"].reshape(-1, 1024)

    # Fit preprocessing on training data only.
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train_x)
    val_z = scaler.transform(val_x)

    # A small, predefined search; no test data is used.
    results = []
    best_correct = -1

    for c in [0.001, 0.01, 0.1, 1.0]:
        model = LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=2000,
            random_state=17,
        )
        model.fit(train_z, train_y.ravel())

        scores = model.decision_function(val_z).reshape(-1, 4)
        choices = scores.argmax(axis=1)
        correct = int(
            val_y[np.arange(len(val_bank)), choices].sum()
        )
        accuracy = 100.0 * correct / len(val_bank)

        result = {
            "C": c,
            "correct": correct,
            "n": len(val_bank),
            "accuracy": accuracy,
            "iterations": int(model.n_iter_[0]),
        }
        results.append(result)
        print(json.dumps(result), flush=True)

        # Equal scores keep the earlier, more regularized model.
        if correct > best_correct:
            best_correct = correct
            best_c = c

            # Save numeric assets for lightweight NumPy inference.
            np.savez(
                output / "model.npz",
                mean=scaler.mean_,
                scale=scaler.scale_,
                weight=model.coef_[0],
                bias=model.intercept_,
            )

    record = {
        "seed": 17,
        "sklearn_version": sklearn.__version__,
        "selection_split": "validation",
        "best_C": best_c,
        "experiments": results,
    }
    (output / "training_results.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    print(f"Best C: {best_c}")
    print(f"Saved assets to: {output}")


if __name__ == "__main__":
    main()