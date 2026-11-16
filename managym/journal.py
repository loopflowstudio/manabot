"""Durable private JSON records shared by game supervision and capture.

Callers own schemas and retention. Appends sync file bytes; creation and atomic
replacement also sync the directory entry. No open descriptor survives a call.
"""

import json
import os
from pathlib import Path
from uuid import uuid4


def json_line(row: object) -> bytes:
    return (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_file(path: Path, body: bytes, flags: int) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | flags, 0o600), "wb") as output:
        output.write(body)
        output.flush()
        os.fsync(output.fileno())


class DurableJournal:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, row: object, *, create: bool = False) -> int:
        body = json_line(row)
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _write_file(self.path, body, os.O_CREAT | os.O_EXCL if create else os.O_APPEND)
        if create:
            _sync_directory(self.path.parent)
        return len(body)

    def replace(self, row: object) -> int:
        body = json_line(row)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            # The temporary name need not survive a crash; only the final name does.
            _write_file(temporary, body, os.O_CREAT | os.O_EXCL)
            temporary.replace(self.path)
            _sync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return len(body)
