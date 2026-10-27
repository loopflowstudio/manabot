"""MTG-183's evaluation as one combined DuckDB database: matches, tables and card evidence.

Jack Heart's direction (2026-10-27): the evaluation's records belong in one
combined database, not a storage object per game, and the arrangement should
be cost-efficient over the long term. Until Etude keeps eval runs that way
itself (Etude Task eb70f8), this module folds every MTG-183 eval run into one
file and publishes it as a single R2 object.

Tables, one row per stored fact, each with the eval run (`source`) it came from:

- `matches`: one row per stored match (`key`, the Etude record as JSON);
- `games`, `card_plays`, `card_counts`, `cards`: the eval tables' rows, as
  Etude delivered them (its `run` and `match` columns included);
- `evidence`: each game's replay-verified card evidence, by the relative
  `path` card tables reference, with its `sha256`, size and gzipped bytes;
- `parts`: every source object or file already folded in, so ingestion is
  idempotent and resumable.

Sources are an eval run's folder in the evaluations bucket (the Mac mini's
`mtg183-eval-cards`, and earlier rented calibrations) or a rented batch's
returned run folder (`experiments.elemental_long_rent.collect`). A table part
its writer merged away (listed in a `tables/<id>.removed` note) is skipped.
Nothing is deleted from any source.

    uv run --extra artifacts --extra evaluation python -m experiments.elemental_long_db \\
        ingest-bucket mtg183-eval-cards
    ... publish
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Protocol

import duckdb
import pandas as pd

BUCKET = "manabot"
PREFIX = "evaluations"
DATABASE = Path("~/.manabot/mtg183-eval-db/mtg183-eval.duckdb").expanduser()
# The published database: one object, replaced on each publish.
PUBLISHED = f"{PREFIX}/mtg183/mtg183-eval.duckdb"
TABLES = ("games", "card_plays", "card_counts", "cards")
_PART = re.compile(
    r"^tables/(?P<table>[a-z_]+)\.(?P<ordinal>\d+)\.(?P<id>[0-9a-f]+)\.parquet$"
)


class Folder(Protocol):
    """An eval run's folder: relative keys and their bytes."""

    name: str

    def keys(self) -> Iterator[str]: ...

    def read(self, key: str) -> bytes: ...


class BucketFolder:
    """An eval run's folder in the evaluations bucket."""

    def __init__(self, run: str) -> None:
        import boto3

        self.name = run
        self.prefix = f"{PREFIX}/{run}/"
        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )

    def keys(self) -> Iterator[str]:
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=self.prefix
        )
        for page in pages:
            for entry in page.get("Contents", []):
                yield entry["Key"].removeprefix(self.prefix)

    def read(self, key: str) -> bytes:
        found = self.client.get_object(Bucket=BUCKET, Key=self.prefix + key)
        return found["Body"].read()


class LocalFolder:
    """An eval run's folder on this machine, such as a rented batch's returned one."""

    def __init__(self, name: str, root: Path) -> None:
        self.name = name
        self.root = root

    def keys(self) -> Iterator[str]:
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                yield str(path.relative_to(self.root))

    def read(self, key: str) -> bytes:
        return (self.root / key).read_bytes()


def connect(path: Path = DATABASE) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(str(path))
    db.execute(
        "CREATE TABLE IF NOT EXISTS parts (source TEXT, key TEXT, PRIMARY KEY (source, key))"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS matches (source TEXT, key TEXT, record JSON, PRIMARY KEY (source, key))"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS evidence (source TEXT, path TEXT, sha256 TEXT, bytes BIGINT,"
        " data BLOB, PRIMARY KEY (source, path))"
    )
    return db


def _known(db: duckdb.DuckDBPyConnection, source: str) -> set[str]:
    rows = db.execute("SELECT key FROM parts WHERE source = ?", [source]).fetchall()
    return {key for (key,) in rows}


def _append(db: duckdb.DuckDBPyConnection, table: str, frame: pd.DataFrame) -> None:
    """Append rows by column name, adding any column the table lacks."""
    exists = db.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()
    db.register("incoming", frame)
    if not exists or not exists[0]:
        db.execute(f"CREATE TABLE {table} AS SELECT * FROM incoming")
    else:
        have = {row[0] for row in db.execute(f"DESCRIBE {table}").fetchall()}
        for name, kind, *_ in db.execute("DESCRIBE incoming").fetchall():
            if name not in have:
                db.execute(f'ALTER TABLE {table} ADD COLUMN "{name}" {kind}')
        db.execute(f"INSERT INTO {table} BY NAME SELECT * FROM incoming")
    db.unregister("incoming")


def ingest(
    db: duckdb.DuckDBPyConnection, folder: Folder, workers: int = 32
) -> dict[str, int]:
    """Fold one eval run's folder into the database; what is already there is skipped."""
    source = folder.name
    keys = list(folder.keys())
    known = _known(db, source)
    # A writer's `tables/<id>.removed` note lists the parts it merged into a newer one.
    retired = {
        line
        for key in keys
        if key.endswith(".removed")
        for line in (_read(folder, key) or b"").decode().splitlines()
        if line
    }
    counts = {"matches": 0, "parts": 0, "evidence": 0, "skipped": 0}

    def fetch(key: str) -> tuple[str, bytes | None]:
        return key, _read(folder, key)

    pending = [k for k in keys if k not in known and not k.endswith(".removed")]
    matches = [
        k
        for k in pending
        if k.startswith("matches/") and not k.endswith(".failed.json")
    ]
    parts = [k for k in pending if _PART.match(k)]
    evidence = [k for k in pending if k.startswith("artifacts/evidence/")]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, data in pool.map(fetch, matches):
            if data is None:
                continue
            record = json.loads(data)
            db.execute(
                "INSERT OR IGNORE INTO matches VALUES (?, ?, ?)",
                [source, Path(key).name.removesuffix(".json"), json.dumps(record)],
            )
            db.execute("INSERT OR IGNORE INTO parts VALUES (?, ?)", [source, key])
            counts["matches"] += 1
        for key, data in pool.map(fetch, parts):
            found = _PART.match(key)
            assert found is not None
            if data is None or key in retired:
                # Merged away by its writer, possibly while this listing ran.
                counts["skipped"] += 1
            else:
                frame = _parquet(data)
                frame.insert(0, "source", source)
                _append(db, found["table"], frame)
                counts["parts"] += 1
            db.execute("INSERT OR IGNORE INTO parts VALUES (?, ?)", [source, key])
        batch: list[list[Any]] = []
        for key, data in pool.map(fetch, evidence):
            if data is None:
                continue
            batch.append(
                [source, key, hashlib.sha256(data).hexdigest(), len(data), data]
            )
            if len(batch) >= 500:
                _evidence(db, batch)
                counts["evidence"] += len(batch)
                batch = []
        if batch:
            _evidence(db, batch)
            counts["evidence"] += len(batch)
    return counts


def _read(folder: Folder, key: str) -> bytes | None:
    """A key's bytes, or None when it vanished after listing (a live writer merged it)."""
    try:
        return folder.read(key)
    except FileNotFoundError:
        return None
    except Exception as error:
        if type(error).__name__ == "NoSuchKey":
            return None
        raise


def _parquet(data: bytes) -> pd.DataFrame:
    """A parquet part's rows, read by DuckDB itself (no pyarrow needed)."""
    with tempfile.NamedTemporaryFile(suffix=".parquet") as part:
        part.write(data)
        part.flush()
        return duckdb.sql(f"SELECT * FROM read_parquet('{part.name}')").df()


def _evidence(db: duckdb.DuckDBPyConnection, rows: list[list[Any]]) -> None:
    db.begin()
    db.executemany("INSERT OR IGNORE INTO evidence VALUES (?, ?, ?, ?, ?)", rows)
    db.executemany(
        "INSERT OR IGNORE INTO parts VALUES (?, ?)", [[row[0], row[1]] for row in rows]
    )
    db.commit()


def publish(path: Path = DATABASE) -> str:
    """Upload the database as one object (multipart), replacing the published one."""
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    db = duckdb.connect(str(path))
    db.execute("CHECKPOINT")
    db.close()
    client.upload_file(str(path), BUCKET, PUBLISHED)
    return f"r2://{BUCKET}/{PUBLISHED}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bucket = commands.add_parser("ingest-bucket")
    bucket.add_argument("run")
    local = commands.add_parser("ingest-folder")
    local.add_argument("run")
    local.add_argument("root", type=Path)
    commands.add_parser("publish")
    commands.add_parser("summary")
    args = parser.parse_args()
    if args.command == "publish":
        print(publish())
        return
    db = connect()
    if args.command == "summary":
        for (source,) in db.execute(
            "SELECT DISTINCT source FROM parts ORDER BY 1"
        ).fetchall():
            row = db.execute(
                "SELECT (SELECT count(*) FROM matches WHERE source = $s),"
                " (SELECT count(*) FROM games WHERE source = $s AND player = 0),"
                " (SELECT count(*) FROM evidence WHERE source = $s)",
                {"s": source},
            ).fetchone()
            print(source, "matches", row[0], "games", row[1], "evidence", row[2])
        return
    folder: Folder = (
        BucketFolder(args.run)
        if args.command == "ingest-bucket"
        else LocalFolder(args.run, args.root)
    )
    print(ingest(db, folder))


if __name__ == "__main__":
    main()
