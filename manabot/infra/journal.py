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


class DurableJournal:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, row: object, *, create: bool = False) -> int:
        body = json_line(row)
        self._write(body, create=create)
        return len(body)

    def _write(self, body: bytes, *, create: bool = False) -> None:
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_WRONLY | (os.O_CREAT | os.O_EXCL if create else os.O_APPEND)
        with os.fdopen(os.open(self.path, flags, 0o600), "wb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        if create:
            _sync_directory(self.path.parent)

    def replace(self, row: object) -> int:
        body = json_line(row)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            # The temporary name need not survive a crash; only the final name does.
            with os.fdopen(
                os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
            ) as output:
                output.write(body)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(self.path)
            _sync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return len(body)
