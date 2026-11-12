"""MTG-281: project retained collection metrics without loading run manifests.

Only one raw block header and its selected arrays are held at a time. Etude's
block selection and CRC checks preserve sparse rows; this diagnostic deliberately
refuses multiple writers instead of inventing recovery semantics. JSON snapshots
retain source versions, header digests and per-update values for offline plots.
Permanent header bytes are a lower bound on index size, reconstructed at each
block's final logged update; they are not measured historical manifest sizes or
fold times. No games, training, provider allocation or remote writes occur here.

uv run --no-sync python scripts/measure_collection_growth.py --help
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import zlib

from etude.cache import Cache
from etude.run.block_file import BlockFile, parse, select
from etude.run.records import Attempt, RunFile
from etude.run.storage import S3Storage, Storage, storage_from_uri
from etude.train.block import ALIGN, MAGIC, VERSION, Header, Span
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Update(Record):
    update: int
    wall_ms: int
    metrics: dict[str, float]


class Block(Record):
    key: str
    version: str
    level: int
    first: int
    last: int
    header_bytes: int
    header_sha256: str
    first_update: int | None
    last_update: int | None
    wall_last: int


class Capture(Record):
    source: str
    run: str
    observed_at: str
    declaration: RunFile
    attempts: list[Attempt]
    versions: dict[str, str]
    blocks: list[Block]
    updates: list[Update]


@dataclass(frozen=True)
class ProjectedBlock:
    block: Block
    updates: list[Update]


@dataclass(frozen=True)
class _ByteRange:
    start: int
    body: bytes


@dataclass(frozen=True)
class _Arrays:
    ranges: list[_ByteRange]

    def read(self, span: Span, dtype: str) -> npt.NDArray[Any]:
        # The caller supplies the SDK column's dtype at this binary boundary.
        for part in self.ranges:
            offset = span.offset - part.start
            if offset >= 0 and offset + span.length <= len(part.body):
                body = part.body[offset : offset + span.length]
                if zlib.crc32(body) != span.crc32:
                    raise ValueError("metric array CRC mismatch")
                return np.frombuffer(body, dtype=dtype)
        if span.length == 0 and span.crc32 == 0:
            return np.array([], dtype=dtype)
        raise ValueError("metric array range missing")


def _project_header(body: bytes) -> Header:
    # JSON is narrowed to the SDK's typed Header after dropping the unrequested
    # matchup namespace. No metric values or byte coordinates are rewritten.
    raw = json.loads(body)
    raw["series"] = [
        column
        for column in raw["series"]
        if column["kind"] == "metric"
        and (
            column["name"] in {"rows", "retained"}
            or column["name"].startswith(("coordinates/", "time/", "memory/"))
        )
    ]
    raw["rollups"] = []
    return Header.model_validate(raw)


def _arrays(storage: Storage, key: str, header: Header, start: int) -> _Arrays:
    # Coalesce nearby arrays to avoid a remote request for every metric. Keep
    # gaps bounded so omitted matchup arrays are not fetched as one large span.
    spans = [header.wall_ms, *(c.data for c in header.clocks)]
    for column in header.series:
        spans.append(column.data)
        if column.rows is not None:
            spans.append(column.rows)
    batches: list[tuple[int, int]] = []
    for span in sorted(spans, key=lambda value: value.offset):
        if not span.length:
            continue
        end = span.offset + span.length
        if batches and span.offset <= batches[-1][1] + 4096:
            batches[-1] = (batches[-1][0], max(end, batches[-1][1]))
        else:
            batches.append((span.offset, end))
    return _Arrays(
        [
            _ByteRange(offset, storage.get_range(key, start + offset, end - offset))
            for offset, end in batches
        ]
    )


def read_block(
    storage: Storage, run: str, file: BlockFile, version: str
) -> ProjectedBlock:
    key = f"{run}/{file.key}"
    front = storage.get_range(key, 0, 16)
    if (
        len(front) != 16
        or front[:8] != MAGIC
        or int.from_bytes(front[8:12], "little") != VERSION
    ):
        raise ValueError(f"unrecognized metric block: {key}")
    length = int.from_bytes(front[12:16], "little")
    body = storage.get_range(key, 16, length)
    if len(body) != length:
        raise ValueError("truncated metric header")
    digest = hashlib.sha256(body).hexdigest()
    header = _project_header(body)
    del body
    if (header.writer, header.first, header.last) != (
        file.writer,
        file.first,
        file.last,
    ):
        raise ValueError("block name/header disagreement")
    start = 16 + length + (-(16 + length) % ALIGN)
    data = _arrays(storage, key, header, start)
    clocks = {column.name: data.read(column.data, "<f8") for column in header.clocks}
    clock = clocks["updates"]
    walls = data.read(header.wall_ms, "<i8")
    if len(clock) != header.rows or len(walls) != header.rows:
        raise ValueError("clock/wall row count disagreement")
    rows: list[Update | None] = []
    for value, wall in zip(clock, walls, strict=True):
        if np.isnan(value):
            rows.append(None)  # Rows without this clock have no update coordinate.
        elif not np.isfinite(value) or value != int(value):
            raise ValueError("invalid update coordinate")
        else:
            rows.append(Update(update=int(value), wall_ms=int(wall), metrics={}))
    for column in header.series:
        indices = (
            data.read(column.rows, "<u4")
            if column.rows is not None
            else np.arange(header.rows)
        )
        values = data.read(column.data, "<f8")
        if len(indices) != len(values):
            raise ValueError("metric row/value count disagreement")
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite retained metric: {column.name}")
        for index, value in zip(indices, values, strict=True):
            row = rows[int(index)]
            if row is not None:
                row.metrics[column.name] = float(value)
    kept = [row for row in rows if row is not None]
    block = Block(
        key=key,
        version=version,
        level=file.level,
        first=file.first,
        last=file.last,
        header_bytes=length,
        header_sha256=digest,
        first_update=min((row.update for row in kept), default=None),
        last_update=max((row.update for row in kept), default=None),
        wall_last=header.wall_last,
    )
    return ProjectedBlock(block, kept)


def capture(storage: Storage, source: str, run: str) -> Capture:
    prefix = run + "/"
    versions = storage.versions(prefix)
    declaration = RunFile.model_validate_json(storage.get(prefix + "run.json"))
    attempts = [
        Attempt.model_validate_json(storage.get(key))
        for key in versions
        if key.startswith(prefix + "metrics/")
        and key.endswith(".json")
        and not key.endswith((".index.json", ".permanent.json"))
    ]
    files = select(
        file for key in versions if (file := parse(key[len(prefix) :])) is not None
    )
    writers = {file.writer for file in files} | {attempt.writer for attempt in attempts}
    if len(writers) != 1 or len(attempts) != 1:
        raise ValueError("diagnostic requires exactly one retained writer/attempt")
    expected = 0
    blocks: list[Block] = []
    updates: dict[int, Update] = {}
    for i, file in enumerate(files):
        if file.first != expected:
            raise ValueError(f"block sequence gap before {file.key}")
        expected = file.last + 1
        projection = read_block(storage, run, file, versions[prefix + file.key])
        blocks.append(projection.block)
        # Later writes replace only the metrics they carry, matching sparse
        # metric-by-metric last-write selection on this single writer.
        for row in projection.updates:
            if row.update in updates:
                updates[row.update].metrics.update(row.metrics)
                updates[row.update].wall_ms = row.wall_ms
            else:
                updates[row.update] = row
        if i % 25 == 0:
            print(f"{run}: {i + 1}/{len(files)} blocks", flush=True)
    return Capture(
        source=source,
        run=run,
        observed_at=datetime.now(timezone.utc).isoformat(),
        declaration=declaration,
        attempts=attempts,
        versions=versions,
        blocks=blocks,
        updates=sorted(updates.values(), key=lambda row: row.update),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"snapshot already exists: {args.out}")
    storage = storage_from_uri(args.source)
    if isinstance(storage, S3Storage):
        storage.cache = Cache(args.cache)
    result = capture(storage, args.source, args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(result.model_dump_json(indent=2) + "\n")
    measured = [row.update for row in result.updates if row.metrics]
    span = (
        f"updates {measured[0]}–{measured[-1]}" if measured else "no measured updates"
    )
    print(f"Captured {span} ({len(measured)} with metrics) to {args.out}")


if __name__ == "__main__":
    main()
