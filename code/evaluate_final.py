"""Evaluate the frozen method and record full-process wall time."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    output = root / "runs"
    checkpoint = output / "fusion"

    config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    )
    if config["alpha"] != 0.5:
        raise ValueError("Expected the frozen alpha=0.5 configuration.")

    env = os.environ.copy()
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        env[name] = "4"

    commands = [
        [
            sys.executable, "predict.py",
            "--method", "fusion_selector.py",
            "--checkpoint", "runs/fusion",
            "--split", "test",
            "--output", "runs/fusion-test-predictions.json",
        ],
        [
            sys.executable, "score_predictions.py",
            "--split", "test",
            "--predictions", "runs/fusion-test-predictions.json",
            "--output", "runs/fusion-test-score.json",
        ],
    ]

    # Include fresh process startup, imports, loading and scoring.
    start = time.perf_counter()
    for command in commands:
        subprocess.run(command, cwd=root, env=env, check=True)
    elapsed = time.perf_counter() - start

    # All inference assets required by this method.
    asset_paths = [
        checkpoint / "model.npz",
        checkpoint / "config.json",
    ]
    assets = []
    total_uncompressed = 0

    for path in asset_paths:
        raw = path.read_bytes()

        if path.suffix == ".npz":
            import zipfile
            with zipfile.ZipFile(path) as archive:
                uncompressed = sum(
                    item.file_size for item in archive.infolist()
                )
        else:
            uncompressed = len(raw)

        total_uncompressed += uncompressed
        assets.append({
            "path": str(path.relative_to(root)),
            "file_bytes": len(raw),
            "uncompressed_bytes": uncompressed,
            "sha256": hashlib.sha256(raw).hexdigest(),
        })

    source_hashes = {}
    for name in [
        "baseline_vote.py",
        "train_linear.py",
        "linear_selector.py",
        "tune_fusion.py",
        "fusion_selector.py",
        "evaluate_final.py",
        "requirements-lock.txt",
    ]:
        source_hashes[name] = hashlib.sha256(
            (root / name).read_bytes()
        ).hexdigest()

    score = json.loads(
        (output / "fusion-test-score.json").read_text(encoding="utf-8")
    )

    record = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "thread_limit": 4,
        "full_prediction_and_scoring_seconds": elapsed,
        "inference_assets_uncompressed_bytes": total_uncompressed,
        "inference_assets_uncompressed_MiB":
            total_uncompressed / (1024 ** 2),
        "assets": assets,
        "source_sha256": source_hashes,
        "score": score,
    }

    (output / "final-evaluation-record.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

    print("\nFinal evaluation record:")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()