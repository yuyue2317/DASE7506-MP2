"""Evaluate the validation-selected relative model."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    output = root / "runs"
    checkpoint = output / "relative"

    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )
    if config["C"] != 0.001 or config["alpha"] != 0.25:
        raise ValueError("Unexpected frozen configuration.")

    env = os.environ.copy()
    for name in [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    ]:
        env[name] = "4"

    commands = [
        [
            sys.executable, "predict.py",
            "--method", "relative_selector.py",
            "--checkpoint", "runs/relative",
            "--split", "test",
            "--output", "runs/relative-test-predictions.json",
        ],
        [
            sys.executable, "score_predictions.py",
            "--split", "test",
            "--predictions", "runs/relative-test-predictions.json",
            "--output", "runs/relative-test-score.json",
        ],
    ]

    start = time.perf_counter()
    for command in commands:
        subprocess.run(command, cwd=root, env=env, check=True)
    elapsed = time.perf_counter() - start

    assets = []
    total_uncompressed = 0

    for name in ["model.npz", "config.json"]:
        path = checkpoint / name
        raw = path.read_bytes()

        if path.suffix == ".npz":
            with zipfile.ZipFile(path) as archive:
                size = sum(item.file_size for item in archive.infolist())
        else:
            size = len(raw)

        total_uncompressed += size
        assets.append({
            "path": path.relative_to(root).as_posix(),
            "file_bytes": len(raw),
            "uncompressed_bytes": size,
            "sha256": hashlib.sha256(raw).hexdigest(),
        })

    source_hashes = {}
    for name in [
        "relative_selector.py",
        "fusion_selector.py",
        "train_relative.py",
        "train_linear.py",
        "analyze_difficulty.py",
        "baseline_vote.py",
        "linear_selector.py",
        "evaluate_relative.py",
        "requirements-lock.txt",
    ]:
        source_hashes[name] = hashlib.sha256(
            (root / name).read_bytes()
        ).hexdigest()

    score = json.loads(
        (output / "relative-test-score.json").read_text(encoding="utf-8")
    )

    record = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "thread_limit": 4,
        "config": config,
        "full_prediction_and_scoring_seconds": elapsed,
        "inference_assets_uncompressed_bytes": total_uncompressed,
        "inference_assets_uncompressed_MiB":
            total_uncompressed / (1024 ** 2),
        "assets": assets,
        "source_sha256": source_hashes,
        "score": score,
    }

    (output / "relative-final-evaluation-record.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    print("\nFinal evaluation record:")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()