"""Atomic file replacement for retained run and report records."""

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
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


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Fence local filesystem writers; fail immediately if another owner is active."""
    with path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("evaluation already has an active attempt") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
