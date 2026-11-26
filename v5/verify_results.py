"""Verify regenerated fifth-edition CSVs against the published reference hashes."""
from hashlib import file_digest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    expected = json.loads((ROOT / 'reference/manifest.json').read_text(encoding='utf-8'))['outputs']
    failures = []
    for name, checksum in sorted(expected.items()):
        path = ROOT / name
        if not path.is_file():
            failures.append(f'Missing: {name}')
            continue
        with path.open('rb') as stream:
            if file_digest(stream, 'sha256').hexdigest() != checksum:
                failures.append(f'Checksum differs: {name}')
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        return 1
    print(f'Verified {len(expected)} fifth-edition result tables (SHA-256).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
