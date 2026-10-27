"""MTG-183's evaluation as one combined DuckDB database: Etude's statsdb plus card evidence.

Jack Heart's direction (2026-10-27): the evaluation's records belong in one
combined database, not a storage object per game, and the arrangement should
be cost-efficient over the long term. Until Etude keeps eval runs that way
itself (Etude Task eb70f8), this module builds one file and publishes it as a
single R2 object.

The database is Etude's own experiment database (`etude.eval.query.statsdb`:
matches, agents, `games`, `card_plays`, `card_counts`, `cards`), so the
17Lands-style card page (`manabot.eval.card_page`) and every Etude query read
it unchanged, plus one table:

- `evidence`: each game's replay-verified card evidence, by the eval run
  (`source`) and the relative `path` that `card_plays.evidence` names, with its
  `sha256`, size and gzipped bytes.

`ingest` folds one eval run's folder into a local mirror of everything but
evidence (`MIRROR`, which statsdb reads) and evidence into `EVIDENCE`; it skips
what is already there and keys a live writer removed after listing. Sources are
an eval run's folder in the evaluations bucket (the Mac mini's
`mtg183-eval-cards`, earlier rented calibrations) or a rented batch's returned
run folder (`experiments.elemental_long_rent.collect`). `build` assembles the
database; `publish` uploads it as one object.

    uv run --extra artifacts --extra evaluation python -m experiments.elemental_long_db \\
        ingest-bucket mtg183-eval-cards
    ... build
    ... publish
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any, Protocol

import duckdb

EXPERIMENT = "mtg183-elemental-eval"
BUCKET = "manabot"
PREFIX = "evaluations"
HOME = Path("~/.manabot/mtg183-eval-db").expanduser()
# Every eval run's folder except its evidence, as statsdb reads a store:
# `MIRROR/<tag>/<run>/…`, one tag per source folder.
MIRROR = HOME / "runs"
EVIDENCE = HOME / "evidence.duckdb"
DATABASE = HOME / "mtg183-eval.duckdb"
# The published database: one object, replaced on each publish.
PUBLISHED = f"{PREFIX}/mtg183/mtg183-eval.duckdb"
_EVIDENCE = "artifacts/evidence/"


class Folder(Protocol):
    """An eval run's folder: relative keys and their bytes."""

    run: str
    tag: str

    def keys(self) -> Iterator[str]: ...

    def read(self, key: str) -> bytes: ...


def _client() -> Any:
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


class BucketFolder:
    """An eval run's folder in the evaluations bucket."""

    def __init__(self, run: str) -> None:
        self.run = run
        self.tag = f"bucket-{run}"
        self.prefix = f"{PREFIX}/{run}/"
        self.client = _client()

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

    def __init__(self, run: str, root: Path, tag: str) -> None:
        self.run = run
        self.tag = tag
        self.root = root

    def keys(self) -> Iterator[str]:
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                yield str(path.relative_to(self.root))

    def read(self, key: str) -> bytes:
        return (self.root / key).read_bytes()


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


def _evidence_db() -> duckdb.DuckDBPyConnection:
    HOME.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(str(EVIDENCE))
    db.execute(
        "CREATE TABLE IF NOT EXISTS evidence (source TEXT, path TEXT, sha256 TEXT,"
        " bytes BIGINT, data BLOB, PRIMARY KEY (source, path))"
    )
    return db


def ingest(folder: Folder, workers: int = 32) -> dict[str, int]:
    """Mirror one eval run's folder and fold its evidence in; what is there is skipped.

    A run's files that change in place (`run.json`, `state.json`, Etude's
    `.removed` notes) are always taken again; parts and matches are immutable.
    """
    target = MIRROR / folder.tag / folder.run
    keys = list(folder.keys())
    with _evidence_db() as db:
        have = {
            path
            for (path,) in db.execute(
                "SELECT path FROM evidence WHERE source = ?", [folder.run]
            ).fetchall()
        }
        evidence = [k for k in keys if k.startswith(_EVIDENCE) and k not in have]
        files = [
            k
            for k in keys
            if not k.startswith(_EVIDENCE)
            and (
                not (target / k).exists()
                or k.endswith((".removed", "run.json", "state.json"))
            )
        ]
        counts = {"files": 0, "evidence": 0, "vanished": 0}

        def fetch(key: str) -> tuple[str, bytes | None]:
            return key, _read(folder, key)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, data in pool.map(fetch, files):
                if data is None:
                    counts["vanished"] += 1
                    continue
                path = target / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.with_name(path.name + ".partial").write_bytes(data)
                path.with_name(path.name + ".partial").replace(path)
                counts["files"] += 1
            rows: list[list[Any]] = []
            for key, data in pool.map(fetch, evidence):
                if data is None:
                    counts["vanished"] += 1
                    continue
                digest = hashlib.sha256(data).hexdigest()
                rows.append([folder.run, key, digest, len(data), data])
                if len(rows) >= 500:
                    _insert(db, rows)
                    counts["evidence"] += len(rows)
                    rows = []
            if rows:
                _insert(db, rows)
                counts["evidence"] += len(rows)
    return counts


def _insert(db: duckdb.DuckDBPyConnection, rows: list[list[Any]]) -> None:
    db.begin()
    db.executemany("INSERT OR IGNORE INTO evidence VALUES (?, ?, ?, ?, ?)", rows)
    db.commit()


def build() -> Path:
    """Etude's statsdb over the mirror, with the evidence table, as `DATABASE`."""
    from etude.eval.query import statsdb

    os.environ["ETUDE_HOME"] = str(HOME / "etude")
    db = statsdb(EXPERIMENT, MIRROR, rebuild=True)
    try:
        db.execute(f"ATTACH '{EVIDENCE}' AS kept (READ_ONLY)")
        db.execute("CREATE OR REPLACE TABLE evidence AS SELECT * FROM kept.evidence")
        db.execute("DETACH kept")
        db.execute("CHECKPOINT")
        kept = Path(db.execute("PRAGMA database_list").fetchall()[0][2])
    finally:
        db.close()
    shutil.copyfile(kept, DATABASE.with_suffix(".partial"))
    DATABASE.with_suffix(".partial").replace(DATABASE)
    return DATABASE


def publish() -> str:
    """Upload `DATABASE` as one object (multipart), replacing the published one."""
    _client().upload_file(str(DATABASE), BUCKET, PUBLISHED)
    return f"r2://{BUCKET}/{PUBLISHED}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bucket = commands.add_parser("ingest-bucket")
    bucket.add_argument("run")
    commands.add_parser("build")
    commands.add_parser("publish")
    args = parser.parse_args()
    if args.command == "ingest-bucket":
        print(ingest(BucketFolder(args.run)))
    elif args.command == "build":
        print(build())
    else:
        print(publish())


if __name__ == "__main__":
    main()
