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
evidence (`MIRROR`) and evidence into `EVIDENCE`; it skips
what is already there and keys a live writer removed after listing. Sources are
an eval run's folder in the evaluations bucket (the Mac mini's
`mtg183-eval-cards`, earlier rented calibrations) or a rented batch's returned
run folder (`experiments.elemental_long_rent.collect`). `build` normalizes a
copy of the mirror (`normalize`), assembles the database from it, and
`publish` uploads it as one object.

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
import json
import os
from pathlib import Path
import re
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
# Etude's home for this database's builds and reports (`ETUDE_HOME`).
ETUDE = HOME / "etude"
# The published database: one object, replaced on each publish.
PUBLISHED = f"{PREFIX}/mtg183/mtg183-eval.duckdb"
_EVIDENCE = "artifacts/evidence/"
# A rated checkpoint's player name, `uNNNNNN` (its update).
_UPDATES = re.compile(r"u\d{6}")
# Environment source digests verified to play the same game (`_same_game`).
_SAME_GAME = {
    "525fed818d16c1aebaa759dd04240c89a9593af0ee3c1575ab67caf8e86a9ad6": (
        "a3d0309ec8d75da3bdada81da9fc3635c015eb45f69a4cb9723dc0281c70ea6b"
    ),
}
# The address the Mac mini's run keyed a checkpoint by.
_ADDRESS = re.compile(r"mtg183:(?P<run>[\w.-]+)/updates-(?P<updates>\d+)")


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


def normalize(record: bytes) -> tuple[str, str, bytes]:
    """A match record as this database keeps it: its old key, new key and normalized record.

    Two fixes to what MTG-183's earlier eval runs recorded, both about the
    rated checkpoints (`uNNNNNN`), so Etude's report draws one curve per line:

    - Seats declare where each checkpoint stood (`at`) and its learner
      (`LINE`). Matches played before the seats did (etude-sdk 0.2.6) give the
      report no position for a checkpoint no training run of this experiment
      recorded. Neither is part of a key.
    - The Mac mini's run (`fb71050cf`) put each checkpoint's update and size in
      its inference config. A config is how a model plays, part of its key, so
      every checkpoint was its own config and the rented runs' `{}` seated
      different players. Position now lives in `at`, so the config is dropped;
      the match's key changes, and `_stage` re-keys its table rows to match.
    - The same run's checkpoints were keyed by an address
      (`mtg183:RUN/updates-N`); the current key is the run and player name
      (`RUN/uNNNNNN`), as `checkpoint_model` seats it. Under two keys one
      checkpoint was two players at one position.
    """
    from etude.eval.records import CheckpointKey, MatchRecord, Model

    from experiments.elemental_long_eval import LINE

    found = MatchRecord.model_validate_json(record)
    players = []
    for seated in found.match.players:
        if isinstance(seated, Model) and _UPDATES.fullmatch(seated.name):
            updates = int(seated.name[1:])
            config = seated.inference.config
            if config and set(config) - {"updates", "bytes"}:
                raise ValueError(f"{seated.name} plays under config {config}")
            if config.get("updates", updates) != updates:
                raise ValueError(f"{seated.name} says it stood at {config}")
            key = seated.checkpoint.key
            if key.address is not None:
                found_key = _ADDRESS.fullmatch(key.address)
                if found_key is None or int(found_key["updates"]) != updates:
                    raise ValueError(f"{seated.name} is keyed {key.address}")
                key = CheckpointKey(run=found_key["run"], name=seated.name)
            seated = seated.model_copy(
                update={
                    "checkpoint": seated.checkpoint.model_copy(
                        update={"key": key, "at": {"updates": updates}, "line": LINE}
                    ),
                    "inference": seated.inference.model_copy(update={"config": {}}),
                }
            )
        players.append(seated)
    match = found.match.model_copy(update={"players": tuple(players)})
    # Re-validate so the key is computed from the normalized players.
    normalized = MatchRecord.model_validate(
        found.model_copy(update={"match": match}).model_dump(mode="json")
    )
    return found.match.key, normalized.match.key, normalized.model_dump_json().encode()


def _same_game(record: bytes) -> bytes:
    """An eval run's environment, with source digests known to play the same game merged.

    The environment digests `elemental_long.py` and `elemental_grid.py`, which
    also hold orchestration. Between the Mac mini's revision (`fb71050cf`) and
    the rented batches' (`e71539d1f` on), they changed only how groups reach
    inference (CUDA endpoint, compute contract) and how training continues; the
    world, pack, limits and the pinned pair schedule are unchanged, and CPU and
    GPU agreed on the pair both played (u5950 v u5600: 0.504, 0.494). Any other
    digest stays its own environment until it is checked the same way.
    """
    found = json.loads(record)
    if found.get("files") in _SAME_GAME:
        found["files"] = _SAME_GAME[found["files"]]
    return json.dumps(found, separators=(",", ":")).encode()


def _stage() -> Path:
    """A fresh copy of the mirror with every match record normalized; returns it.

    The mirror stays as its sources hold it. Match records move to their new
    key's file, and each table part's `match` column is re-keyed alongside.
    """
    store = HOME / "store"
    # Etude updates a store's cached statsdb incrementally, which would keep
    # the previous staging's match keys; every staging starts it afresh.
    for found in (ETUDE / "statsdb").glob(f"{EXPERIMENT}.*"):
        found.unlink()
    if store.exists():
        shutil.rmtree(store)
    keys: dict[str, str] = {}
    for run in MIRROR.glob("*/*"):
        target = store / run.relative_to(MIRROR)
        for source in run.rglob("*"):
            if source.is_dir():
                continue
            relative = source.relative_to(run)
            if relative.parts[0] == "matches" and source.suffix == ".json":
                old, key, normalized = normalize(source.read_bytes())
                keys[old] = key
                suffix = source.name.removeprefix(old)
                destination = target / "matches" / f"{key}{suffix}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(normalized)
            elif relative == Path("environment.json"):
                (target / relative).parent.mkdir(parents=True, exist_ok=True)
                (target / relative).write_bytes(_same_game(source.read_bytes()))
            elif relative.parts[0] != "tables" or source.suffix != ".parquet":
                (target / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target / relative)
    mapping = duckdb.connect()
    mapping.execute("CREATE TABLE rekey (old VARCHAR PRIMARY KEY, new VARCHAR)")
    mapping.executemany("INSERT INTO rekey VALUES (?, ?)", list(keys.items()))
    for part in MIRROR.glob("*/*/tables/*.parquet"):
        destination = store / part.relative_to(MIRROR)
        destination.parent.mkdir(parents=True, exist_ok=True)
        mapping.execute(
            f"COPY (SELECT t.* REPLACE (coalesce(r.new, t.match) AS match)"
            f" FROM read_parquet('{part}') t LEFT JOIN rekey r ON r.old = t.match)"
            f" TO '{destination}' (FORMAT parquet)"
        )
    mapping.close()
    return store


def build() -> Path:
    """Etude's statsdb over the normalized mirror, with the evidence table, as `DATABASE`."""
    from etude.eval.query import statsdb

    store = _stage()
    os.environ["ETUDE_HOME"] = str(ETUDE)
    db = statsdb(EXPERIMENT, store, rebuild=True)
    try:
        db.execute(f"ATTACH '{EVIDENCE}' AS evidence_store (READ_ONLY)")
        db.execute(
            "CREATE OR REPLACE TABLE evidence AS SELECT * FROM evidence_store.main.evidence"
        )
        db.execute("DETACH evidence_store")
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
