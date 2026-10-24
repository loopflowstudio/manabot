"""Atomic file replacement for retained run and report records."""

import json
import os
from pathlib import Path


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)
