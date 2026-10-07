"""A transactional local control store for offline subprocess lifecycle fixtures."""

from pathlib import Path
import sqlite3

from manabot.remote.job_store import StoredValue
from manabot.remote.plan import digest


class FileStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS objects (key TEXT PRIMARY KEY, data BLOB, etag TEXT)"
            )

    def read(self, key: str) -> StoredValue | None:
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT data, etag FROM objects WHERE key=?", (key,)
            ).fetchone()
        return StoredValue(row[0], row[1]) if row else None

    def create(self, key: str, data: bytes) -> bool:
        with sqlite3.connect(self.path) as db:
            result = db.execute(
                "INSERT OR IGNORE INTO objects VALUES (?,?,?)",
                (key, data, digest(data)),
            )
            return result.rowcount == 1

    def replace(self, key: str, data: bytes, etag: str) -> bool:
        with sqlite3.connect(self.path) as db:
            result = db.execute(
                "UPDATE objects SET data=?, etag=? WHERE key=? AND etag=?",
                (data, digest(data), key, etag),
            )
            return result.rowcount == 1
