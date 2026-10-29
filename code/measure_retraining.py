"""Measure a later rerun, restoring the original final checkpoint afterward."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time


def hashes(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file()}


def main():
    root = Path(__file__).resolve().parent
    target = root / 'runs/relative'
    record_path = root / 'runs/supplementary-retraining-resources.json'
    if record_path.exists():
        raise FileExistsError('Supplementary record exists; archive it before another run.')
    before = hashes(target)
    env = os.environ.copy()
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        env[name] = '4'
    with tempfile.TemporaryDirectory(prefix='mp2-checkpoint-backup-') as tmp:
        backup = Path(tmp) / 'relative'
        shutil.copytree(target, backup)
        try:
            start = time.perf_counter()
            result = subprocess.run([sys.executable, 'train_relative.py'], cwd=root, env=env)
            elapsed = time.perf_counter() - start
            regenerated = hashes(target)
        finally:
            shutil.rmtree(target)
            shutil.copytree(backup, target)
        if hashes(target) != before:
            raise RuntimeError('Checkpoint restoration verification failed.')
    record = {
        'measurement_type': 'later_rerun_not_original_historical_duration',
        'command': 'python train_relative.py',
        'returncode': result.returncode,
        'complete_training_and_search_seconds': elapsed,
        'platform': platform.platform(), 'processor': platform.processor(),
        'python': sys.version, 'configured_threads': 4,
        'fits': 4, 'agreement_settings_per_fit': 6,
        'original_files_restored': True,
        'regenerated_hashes': regenerated,
        'original_hashes': before,
        'model_bytes_match': regenerated.get('model.npz') == before.get('model.npz'),
    }
    record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
