"""Immutable raw opponent-memory transport, shared by collection and diagnostics.

MemoryStore interns exact record/detail bytes, never neural state. A Prefix owns
its reachable chunks so returned batches survive stream reset and later updates.
CompactObservation declares configured bounds separately from selected-batch
padding; policy admission still checks every raw value before vocabulary lookup.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import sys
from weakref import WeakValueDictionary

import numpy as np

from manabot.policy.opponent_memory import admit_memory

MEMORY_FIELDS = frozenset({"opponent_records", "opponent_details"})


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class Chunk:
    key: str
    rows: np.ndarray


@dataclass(frozen=True, slots=True)
class PrefixIdentity:
    game: str
    viewer: int
    branch: str

    def __post_init__(self) -> None:
        if not self.game or not self.branch or self.viewer not in (0, 1):
            raise ValueError("invalid opponent-memory prefix identity")


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class Tail:
    key: str
    previous: "Tail | None"
    record: Chunk
    detail: Chunk
    record_count: int
    detail_count: int


@dataclass(frozen=True, slots=True)
class Prefix:
    identity: PrefixIdentity
    tail: Tail | None

    def chunks(self) -> Iterator[tuple[Chunk, Chunk]]:
        nodes: list[Tail] = []
        tail = self.tail
        while tail is not None:
            nodes.append(tail)
            tail = tail.previous
        for node in reversed(nodes):
            yield node.record, node.detail

    @property
    def records(self) -> tuple[Chunk, ...]:
        return tuple(record for record, _ in self.chunks())

    @property
    def details(self) -> tuple[Chunk, ...]:
        return tuple(detail for _, detail in self.chunks())

    @property
    def record_count(self) -> int:
        return self.tail.record_count if self.tail is not None else 0

    @property
    def detail_count(self) -> int:
        return self.tail.detail_count if self.tail is not None else 0


class PrefixObservation(dict[str, np.ndarray]):
    """Retained ordinary rows and immutable prefixes, with no padded memory owner.

    Materialize selected rows only for admission/preparation. The returned raw
    arrays must not be retained by the rollout or its admission owner.
    """

    def __init__(
        self,
        rows: Mapping[str, np.ndarray],
        prefixes: Sequence[Prefix],
        capacities: tuple[int, int],
    ) -> None:
        if min(capacities) < 1 or any(
            p.record_count > capacities[0] or p.detail_count > capacities[1]
            for p in prefixes
        ):
            raise ValueError("opponent-memory prefix exceeds configured capacity")
        super().__init__((k, v) for k, v in rows.items() if k not in MEMORY_FIELDS)
        self.prefixes = tuple(prefixes)
        self.memory_capacities = capacities

    def materialize(self) -> "CompactObservation":
        return CompactObservation(self, self.prefixes, self.memory_capacities)


class CompactObservation(PrefixObservation):
    """Temporary selected raw rows padded to their maxima under declared bounds."""

    def __init__(
        self,
        rows: Mapping[str, np.ndarray],
        prefixes: Sequence[Prefix],
        capacities: tuple[int, int],
    ) -> None:
        super().__init__(rows, prefixes, capacities)
        self.update(gather_memory(self.prefixes))


def gather_bytes(prefixes: Sequence[Prefix]) -> int:
    """Preflight selected padding without allocating it (including duplicates)."""
    return (
        len(prefixes)
        * 4
        * (
            max((p.record_count for p in prefixes), default=0) * 18
            + max((p.detail_count for p in prefixes), default=0) * 8
        )
    )


def gather_memory(prefixes: Sequence[Prefix]) -> dict[str, np.ndarray]:
    headers = np.zeros(
        (len(prefixes), max((p.record_count for p in prefixes), default=0), 18),
        np.float32,
    )
    details = np.zeros(
        (len(prefixes), max((p.detail_count for p in prefixes), default=0), 8),
        np.float32,
    )
    for row, prefix in enumerate(prefixes):
        cursor = 0
        for ordinal, (header, chunk) in enumerate(prefix.chunks()):
            headers[row, ordinal] = header.rows[0]
            details[row, cursor : cursor + len(chunk.rows)] = chunk.rows
            cursor += len(chunk.rows)
    return {"opponent_records": headers, "opponent_details": details}


class MemoryStore:
    """One transport owner. Weak interning releases unreachable game chunks."""

    def __init__(self, capacities: tuple[int, int]) -> None:
        if min(capacities) < 1:
            raise ValueError("opponent-memory capacities must be positive")
        self.capacities = capacities
        self._tails: WeakValueDictionary[str, Tail] = WeakValueDictionary()
        self._chunks: WeakValueDictionary[str, Chunk] = WeakValueDictionary()

    def _intern(self, rows: np.ndarray) -> Chunk:
        data = rows.astype(np.float32, copy=False).tobytes()
        key = sha256(str(rows.shape).encode() + data).hexdigest()
        previous = self._chunks.get(key)
        if previous is not None:
            return previous
        # A bytes-backed array cannot be made writable by a later reader.
        chunk = Chunk(key, np.frombuffer(data, np.float32).reshape(rows.shape))
        self._chunks[key] = chunk
        return chunk

    def capture(
        self, buffers: Mapping[str, np.ndarray], row: int, identity: PrefixIdentity
    ) -> Prefix:
        headers = buffers["opponent_records"][row]
        details = buffers["opponent_details"][row]
        if headers.dtype != np.float32 or details.dtype != np.float32:
            raise ValueError("native opponent-memory transport requires float32")
        if headers.shape != (self.capacities[0], 18) or details.shape != (
            self.capacities[1],
            8,
        ):
            raise ValueError("opponent-memory capacity mismatch")
        # Dense native input is the reference ABI: admit before cropping it.
        admit_memory({key: buffers[key][row : row + 1] for key in MEMORY_FIELDS})
        count = int(headers[:, 17].sum())
        tail: Tail | None = None
        for header in headers[:count]:
            start, size = map(int, header[14:16])
            tail = self._append(
                tail,
                self._intern(header[None]),
                self._intern(details[start : start + size]),
            )
        return Prefix(identity, tail)

    def _append(self, previous: Tail | None, record: Chunk, detail: Chunk) -> Tail:
        key = sha256(
            ((previous.key if previous else "") + record.key + detail.key).encode()
        ).hexdigest()
        tail = self._tails.get(key)
        if tail is None:
            tail = Tail(
                key,
                previous,
                record,
                detail,
                1 + (previous.record_count if previous else 0),
                len(detail.rows) + (previous.detail_count if previous else 0),
            )
            self._tails[key] = tail
        return tail

    def measure(self, prefixes: Sequence[Prefix]) -> dict[str, int]:
        chunks: dict[str, Chunk] = {}
        tails: dict[str, Tail] = {}
        identities = {id(p.identity): p.identity for p in prefixes}
        for prefix in prefixes:
            tail = prefix.tail
            while tail is not None and tail.key not in tails:
                tails[tail.key] = tail
                chunks[tail.record.key] = tail.record
                chunks[tail.detail.key] = tail.detail
                tail = tail.previous
        # CPython ownership sizes. Native arrays, transient lookup dictionaries
        # and gather/preparation are separate from retained prefix ownership.
        metadata = sys.getsizeof(prefixes) + sum(
            sys.getsizeof(p) for p in {id(p): p for p in prefixes}.values()
        )
        metadata += sum(
            sys.getsizeof(i) + sys.getsizeof(i.game) + sys.getsizeof(i.branch)
            for i in identities.values()
        )
        metadata += sum(sys.getsizeof(t) + sys.getsizeof(t.key) for t in tails.values())
        metadata += sum(
            sys.getsizeof(c)
            + sys.getsizeof(c.key)
            + sys.getsizeof(c.rows)
            + sys.getsizeof(c.rows.base)
            + 33
            for c in chunks.values()
        )
        return {
            "unique_chunk_bytes": sum(c.rows.nbytes for c in chunks.values()),
            "chunk_count": len(chunks),
            "tail_count": len(tails),
            "python_metadata_bytes": metadata,
            "intern_index_bytes": sum(
                sys.getsizeof(index.data)
                + sum(sys.getsizeof(ref) for ref in index.data.values())
                for index in (self._tails, self._chunks)
            ),
            "selected_padding_bytes": gather_bytes(prefixes),
        }


@dataclass(frozen=True)
class MemoryArchive:
    """Diagnostic codec: reachable chunks once, exact per-row references."""

    capacities: tuple[int, int]
    identities: tuple[PrefixIdentity, ...]
    keys: tuple[str, ...]
    chunks: tuple[np.ndarray, ...]
    offsets: np.ndarray
    records: np.ndarray
    details: np.ndarray

    @property
    def nbytes(self) -> int:
        return (
            sum(c.nbytes for c in self.chunks)
            + self.offsets.nbytes
            + self.records.nbytes
            + self.details.nbytes
            + sum(len(k.encode()) for k in self.keys)
            + sum(
                len(p.game.encode()) + len(p.branch.encode()) + 8
                for p in self.identities
            )
            + 16
        )


def archive_memory(
    prefixes: Sequence[Prefix], capacities: tuple[int, int]
) -> MemoryArchive:
    """Build reference tables; chunk arrays are immutable views, never expanded."""
    chunks = {c.key: c for p in prefixes for c in (*p.records, *p.details)}
    keys = tuple(chunks)
    indexes = {key: index for index, key in enumerate(keys)}
    offsets = np.array([0, *np.cumsum([p.record_count for p in prefixes])], np.int64)
    return MemoryArchive(
        capacities,
        tuple(p.identity for p in prefixes),
        keys,
        tuple(chunks[key].rows for key in keys),
        offsets,
        np.array([indexes[c.key] for p in prefixes for c in p.records], np.int64),
        np.array([indexes[c.key] for p in prefixes for c in p.details], np.int64),
    )


def restore_memory(archive: MemoryArchive) -> tuple[Prefix, ...]:
    """Reject missing/corrupt chunks; reconstruct exactly without padded rollout."""
    store = MemoryStore(archive.capacities)
    if len(archive.keys) != len(archive.chunks):
        raise ValueError("missing opponent-memory chunks")
    chunks: list[Chunk] = []
    for key, rows in zip(archive.keys, archive.chunks, strict=True):
        if rows.ndim != 2 or rows.dtype != np.float32 or rows.shape[1] not in (8, 18):
            raise ValueError("invalid opponent-memory chunk shape")
        chunk = store._intern(rows)
        if key != chunk.key:
            raise ValueError("corrupt opponent-memory chunk")
        chunks.append(chunk)
    offsets = archive.offsets
    if (
        offsets.dtype != np.int64
        or offsets.shape != (len(archive.identities) + 1,)
        or offsets[0] != 0
        or (np.diff(offsets) < 0).any()
        or offsets[-1] != len(archive.records)
        or len(archive.records) != len(archive.details)
    ):
        raise ValueError("invalid opponent-memory prefix offsets")
    for refs in (archive.records, archive.details):
        if (
            refs.dtype != np.int64
            or refs.ndim != 1
            or ((refs < 0) | (refs >= len(chunks))).any()
        ):
            raise ValueError("missing opponent-memory chunk reference")
    prefixes: list[Prefix] = []
    for row, identity in enumerate(archive.identities):
        start, stop = offsets[row : row + 2]
        tail: Tail | None = None
        for record_index, detail_index in zip(
            archive.records[start:stop], archive.details[start:stop], strict=True
        ):
            record, detail = chunks[int(record_index)], chunks[int(detail_index)]
            tail = store._append(tail, record, detail)
        prefix = Prefix(identity, tail)
        if any(c.rows.shape != (1, 18) for c in prefix.records) or any(
            c.rows.shape[1] != 8 for c in prefix.details
        ):
            raise ValueError("invalid opponent-memory chunk role")
        if (
            prefix.record_count > archive.capacities[0]
            or prefix.detail_count > archive.capacities[1]
        ):
            raise ValueError("opponent-memory capacity mismatch")
        admit_memory(gather_memory([prefix]))
        prefixes.append(prefix)
    return tuple(prefixes)
