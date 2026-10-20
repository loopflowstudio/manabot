"""Verify an explicitly retained transfer inventory without running a model.

The control inventory can be admitted while remaining checkpoints transfer.
Full evaluation separately requires verification of the entire original inventory.
"""

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Literal

SOURCE = '3c3681d2b26a7fc8167928fa63fe24a6cd3d058a'
ROOT = Path('.runs/grid-007/learning-017/mini-transfer')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(scope: Literal['control', 'full']) -> None:
    if platform.node().split('.')[0] != 'mini-heart':
        raise RuntimeError('Transfer verification must run on the destination Mini')
    manifest = ROOT / ('control-manifest.json' if scope == 'control' else 'manifest.json')
    output = ROOT / ('control-verification.json' if scope == 'control' else 'verification.json')
    if output.exists():
        raise RuntimeError('Verification already retained; inspect before repeating')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    assert source == SOURCE
    changed = subprocess.check_output(
        ['git', 'diff', '--name-only', SOURCE], text=True
    ).splitlines()
    repair_path = ROOT / 'worker-logging-fix.json'
    repair_sha256: str | None = None
    if changed:
        repair = json.loads(repair_path.read_text())
        assert changed == ['experiments/runners/gpu_job.py']
        assert repair['path'] == changed[0] and repair['scientific_source'] == SOURCE
        base = subprocess.check_output(['git', 'show', SOURCE + ':' + changed[0]])
        assert hashlib.sha256(base).hexdigest() == repair['base_sha256']
        assert digest(Path(changed[0])) == repair['patched_sha256']
        repair_sha256 = digest(repair_path)
    inventory = json.loads(manifest.read_text())
    assert isinstance(inventory, dict) and inventory['source'] == SOURCE
    files = inventory['files']
    assert isinstance(files, list) and files
    verified: set[str] = set()
    total_bytes = 0
    for row in files:
        assert isinstance(row, dict)
        name, size, sha256 = row['path'], row['bytes'], row['sha256']
        assert isinstance(name, str) and isinstance(size, int) and isinstance(sha256, str)
        path = Path(name)
        assert not path.is_absolute() and path.resolve().is_relative_to(Path.cwd())
        assert name not in verified and path.is_file()
        assert path.stat().st_size == size and digest(path) == sha256, name
        verified.add(name)
        total_bytes += size
    proof = {
        'observed_at': datetime.now(UTC).isoformat(), 'host': platform.node(),
        'scope': scope, 'source': source, 'source_verified': True,
        'logging_repair_sha256': repair_sha256,
        'manifest_sha256': digest(manifest), 'files': len(verified),
        'bytes': total_bytes, 'all_files_verified': True,
    }
    output.write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('control', 'full'), required=True)
    args = parser.parse_args()
    verify(args.scope)


if __name__ == '__main__':
    main()
