"""Verify packaged identity and records without reading reference answers."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    source = json.loads((root / 'SOURCE_SHA256.json').read_text(encoding='utf-8'))
    for name, expected in source.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Source hash mismatch: {name}')

    manifest = json.loads((root / 'ASSET_MANIFEST.json').read_text(encoding='utf-8'))
    for asset in manifest['assets']:
        actual = hashlib.sha256((root / asset['path']).read_bytes()).hexdigest()
        if actual != asset['sha256']:
            raise ValueError(f"Asset hash mismatch: {asset['path']}")

    original = json.loads((root / 'runs/relative-test-predictions.json').read_text())
    clean = json.loads((root / 'runs/clean-test-predictions.json').read_text())
    for record in (original, clean):
        if record['split'] != 'test' or len(record['ids']) != 467:
            raise ValueError('Unexpected saved test record')
        if len(set(record['ids'])) != 467 or len(record['selected']) != 467:
            raise ValueError('Incomplete test IDs or choices')
        if not all(type(i) is int and 0 <= i < 4 for i in record['selected']):
            raise ValueError('Invalid choice')
    if original['ids'] != clean['ids'] or original['selected'] != clean['selected']:
        raise ValueError('Clean environment predictions differ')

    for name in ['runs/relative-test-score.json', 'runs/clean-test-score.json']:
        score = json.loads((root / name).read_text())
        if (score['correct'], score['n']) != (377, 467):
            raise ValueError(f'Unexpected score record: {name}')
        if abs(score['accuracy'] - 100 * 377 / 467) > 1e-10:
            raise ValueError('Inconsistent saved accuracy')

    print(json.dumps({
        'status': 'passed',
        'source_files_checked': len(source),
        'checkpoint_hashes': 'matched',
        'clean_environment_saved_choices': 'all 467 identical',
        'reported_test_correct': 377,
        'reported_test_n': 467,
        'official_scoring_rerun': False,
        'note': 'Identity and saved-record consistency checks only; use the official scorer for accuracy.'
    }, indent=2))


if __name__ == '__main__':
    main()
