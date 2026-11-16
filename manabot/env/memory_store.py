"""Immutable raw prefix transport for records, snapshots, programs and members.

Fixed-size table chunks share unchanged prefixes, including coalesced versions.
Only selected execution rows materialize; archives own reachable chunks once.
All arbitrary dense arrays are admitted before interning, including padding.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import sys
from weakref import WeakValueDictionary

import numpy as np

from manabot.env.decision_memory import MEMORY_FIELDS, MEMORY_WIDTHS
from manabot.policy.decision_memory import admit_memory, memory_lengths

MemoryCapacities = tuple[int, int, int, int]
_CHUNK_ROWS = 32


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
            raise ValueError("invalid decision-memory prefix identity")


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class TableTail:
    key: str
    previous: "TableTail | None"
    chunk: Chunk
    count: int

    def chunks(self) -> Iterator[Chunk]:
        nodes: list[TableTail] = []
        node: TableTail | None = self
        while node is not None:
            nodes.append(node)
            node = node.previous
        for node in reversed(nodes):
            yield node.chunk


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class Tail:
    key: str
    tables: tuple[TableTail | None, ...]


@dataclass(frozen=True, slots=True)
class Prefix:
    identity: PrefixIdentity
    tail: Tail | None

    @property
    def counts(self) -> tuple[int, ...]:
        return (
            tuple(node.count if node else 0 for node in self.tail.tables)
            if self.tail
            else (0,) * len(MEMORY_FIELDS)
        )

    @property
    def record_count(self) -> int:
        return self.counts[0]

    @property
    def detail_count(self) -> int:
        return self.counts[1]


class PrefixObservation(dict[str, np.ndarray]):
    """Ordinary rows and immutable prefixes, without a dense memory owner."""

    def __init__(
        self,
        rows: Mapping[str, np.ndarray],
        prefixes: Sequence[Prefix],
        capacities: MemoryCapacities,
    ) -> None:
        if (
            len(capacities) != len(MEMORY_FIELDS)
            or min(capacities) < 1
            or any(
                any(
                    count > cap for count, cap in zip(p.counts, capacities, strict=True)
                )
                for p in prefixes
            )
        ):
            raise ValueError("decision-memory prefix exceeds configured capacity")
        super().__init__((k, v) for k, v in rows.items() if k not in MEMORY_FIELDS)
        self.prefixes = tuple(prefixes)
        self.memory_capacities = capacities

    def materialize(self) -> "CompactObservation":
        return CompactObservation(self, self.prefixes, self.memory_capacities)


class CompactObservation(PrefixObservation):
    """Selected rows padded to their maxima under explicitly declared bounds."""

    def __init__(
        self,
        rows: Mapping[str, np.ndarray],
        prefixes: Sequence[Prefix],
        capacities: MemoryCapacities,
    ) -> None:
        super().__init__(rows, prefixes, capacities)
        self.update(gather_memory(self.prefixes))


def select_rows(
    rows: Mapping[str, np.ndarray], indexes: np.ndarray
) -> dict[str, np.ndarray]:
    if isinstance(rows, PrefixObservation):
        return PrefixObservation(
            {k: v[indexes] for k, v in rows.items() if k not in MEMORY_FIELDS},
            [rows.prefixes[int(i)] for i in indexes],
            rows.memory_capacities,
        )
    return {k: v[indexes] for k, v in rows.items()}


def materialize(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return rows.materialize() if isinstance(rows, PrefixObservation) else rows


def gather_bytes(prefixes: Sequence[Prefix]) -> int:
    return (
        len(prefixes)
        * 4
        * sum(
            max((p.counts[i] for p in prefixes), default=0) * MEMORY_WIDTHS[key]
            for i, key in enumerate(MEMORY_FIELDS)
        )
    )


def gather_memory(prefixes: Sequence[Prefix]) -> dict[str, np.ndarray]:
    result = {
        key: np.zeros(
            (
                len(prefixes),
                max((p.counts[i] for p in prefixes), default=0),
                MEMORY_WIDTHS[key],
            ),
            np.float32,
        )
        for i, key in enumerate(MEMORY_FIELDS)
    }
    for row, prefix in enumerate(prefixes):
        for key, table in zip(
            MEMORY_FIELDS,
            prefix.tail.tables if prefix.tail else (None,) * len(MEMORY_FIELDS),
            strict=True,
        ):
            cursor = 0
            for chunk in table.chunks() if table else ():
                result[key][row, cursor : cursor + len(chunk.rows)] = chunk.rows
                cursor += len(chunk.rows)
    return result


class MemoryStore:
    """Weak exact-byte interning releases all unreachable game ownership."""

    def __init__(self, capacities: MemoryCapacities) -> None:
        if len(capacities) != len(MEMORY_FIELDS) or min(capacities) < 1:
            raise ValueError("decision-memory capacities must be positive")
        self.capacities = capacities
        self._tails: WeakValueDictionary[str, Tail] = WeakValueDictionary()
        self._tables: WeakValueDictionary[str, TableTail] = WeakValueDictionary()
        self._chunks: WeakValueDictionary[str, Chunk] = WeakValueDictionary()

    def _intern(self, rows: np.ndarray) -> Chunk:
        data = rows.astype(np.float32, copy=False).tobytes()
        key = sha256(str(rows.shape).encode() + data).hexdigest()
        previous = self._chunks.get(key)
        if previous is not None:
            if previous.rows.tobytes() != data:
                raise ValueError("decision-memory chunk hash collision")
            return previous
        chunk = Chunk(key, np.frombuffer(data, np.float32).reshape(rows.shape))
        self._chunks[key] = chunk
        return chunk

    def _append(self, previous: TableTail | None, chunk: Chunk) -> TableTail:
        key = sha256(
            ((previous.key if previous else "") + chunk.key).encode()
        ).hexdigest()
        tail = self._tables.get(key)
        if tail is not None and (
            tail.previous is not previous or tail.chunk is not chunk
        ):
            raise ValueError("decision-memory table hash collision")
        if tail is None:
            tail = TableTail(
                key,
                previous,
                chunk,
                len(chunk.rows) + (previous.count if previous else 0),
            )
            self._tables[key] = tail
        return tail

    def _prefix(
        self, identity: PrefixIdentity, tables: tuple[TableTail | None, ...]
    ) -> Prefix:
        if not any(tables):
            return Prefix(identity, None)
        key = sha256(
            "/".join(node.key if node else "" for node in tables).encode()
        ).hexdigest()
        tail = self._tails.get(key)
        if tail is not None and tail.tables != tables:
            raise ValueError("decision-memory prefix hash collision")
        if tail is None:
            tail = Tail(key, tables)
            self._tails[key] = tail
        return Prefix(identity, tail)

    def capture(
        self, buffers: Mapping[str, np.ndarray], row: int, identity: PrefixIdentity
    ) -> Prefix:
        for key, capacity in zip(MEMORY_FIELDS, self.capacities, strict=True):
            if buffers[key].dtype != np.float32 or buffers[key][row].shape != (
                capacity,
                MEMORY_WIDTHS[key],
            ):
                raise ValueError(
                    "decision-memory capacity mismatch or non-float32 transport"
                )
        selected = {key: buffers[key][row : row + 1] for key in MEMORY_FIELDS}
        admit_memory(selected)
        counts = memory_lengths(selected, 0)
        tables: list[TableTail | None] = []
        for key, count in zip(MEMORY_FIELDS, counts, strict=True):
            tail: TableTail | None = None
            for start in range(0, count, _CHUNK_ROWS):
                tail = self._append(
                    tail,
                    self._intern(
                        buffers[key][row, start : min(start + _CHUNK_ROWS, count)]
                    ),
                )
            tables.append(tail)
        return self._prefix(identity, tuple(tables))

    def measure(self, prefixes: Sequence[Prefix]) -> dict[str, int]:
        chunks: dict[str, Chunk] = {}
        tables: dict[str, TableTail] = {}
        tails = {p.tail.key: p.tail for p in prefixes if p.tail is not None}
        for prefix in prefixes:
            for table in (
                prefix.tail.tables if prefix.tail else (None,) * len(MEMORY_FIELDS)
            ):
                while table is not None and table.key not in tables:
                    tables[table.key] = table
                    chunks[table.chunk.key] = table.chunk
                    table = table.previous
        metadata = sys.getsizeof(prefixes) + sum(
            sys.getsizeof(p)
            + sys.getsizeof(p.identity)
            + len(p.identity.game)
            + len(p.identity.branch)
            for p in prefixes
        )
        metadata += sum(
            sys.getsizeof(node) + sys.getsizeof(node.key)
            for node in (*tables.values(), *tails.values())
        )
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
            "tail_count": len(tails) + len(tables),
            "python_metadata_bytes": metadata,
            "intern_index_bytes": sum(
                sys.getsizeof(index.data)
                + sum(sys.getsizeof(ref) for ref in index.data.values())
                for index in (self._tails, self._tables, self._chunks)
            ),
            "selected_padding_bytes": gather_bytes(prefixes),
        }


@dataclass(frozen=True)
class MemoryArchive:
    """Each prefix/table lists its exact ordered chunks in one shared dictionary."""

    capacities: MemoryCapacities
    identities: tuple[PrefixIdentity, ...]
    keys: tuple[str, ...]
    chunks: tuple[np.ndarray, ...]
    offsets: np.ndarray
    indexes: np.ndarray

    @property
    def nbytes(self) -> int:
        return (
            sum(c.nbytes for c in self.chunks)
            + self.offsets.nbytes
            + self.indexes.nbytes
            + sum(len(k) for k in self.keys)
            + sum(
                len(p.game.encode()) + len(p.branch.encode()) + 8
                for p in self.identities
            )
            + 32
        )


def archive_memory(
    prefixes: Sequence[Prefix], capacities: MemoryCapacities
) -> MemoryArchive:
    chunks: dict[str, Chunk] = {}
    indexes: dict[str, int] = {}
    offsets = [0]
    references: list[int] = []
    for prefix in prefixes:
        for table in (
            prefix.tail.tables if prefix.tail else (None,) * len(MEMORY_FIELDS)
        ):
            for chunk in table.chunks() if table else ():
                if chunk.key not in indexes:
                    indexes[chunk.key] = len(chunks)
                    chunks[chunk.key] = chunk
                references.append(indexes[chunk.key])
            offsets.append(len(references))
    return MemoryArchive(
        capacities,
        tuple(p.identity for p in prefixes),
        tuple(chunks),
        tuple(c.rows for c in chunks.values()),
        np.array(offsets, np.int64),
        np.array(references, np.int64),
    )


def restore_memory(archive: MemoryArchive) -> tuple[Prefix, ...]:
    store = MemoryStore(archive.capacities)
    if len(archive.keys) != len(archive.chunks):
        raise ValueError("missing decision-memory chunks")
    chunks: list[Chunk] = []
    for key, rows in zip(archive.keys, archive.chunks, strict=True):
        if (
            rows.ndim != 2
            or rows.dtype != np.float32
            or rows.shape[1] not in MEMORY_WIDTHS.values()
            or not 0 < len(rows) <= _CHUNK_ROWS
        ):
            raise ValueError("invalid decision-memory chunk shape")
        chunk = store._intern(rows)
        if key != chunk.key:
            raise ValueError("corrupt decision-memory chunk")
        chunks.append(chunk)
    offsets, refs = archive.offsets, archive.indexes
    if (
        offsets.dtype != np.int64
        or offsets.shape != (len(archive.identities) * len(MEMORY_FIELDS) + 1,)
        or offsets[0] != 0
        or (np.diff(offsets) < 0).any()
        or offsets[-1] != len(refs)
    ):
        raise ValueError("invalid decision-memory prefix offsets")
    if (
        refs.dtype != np.int64
        or refs.ndim != 1
        or ((refs < 0) | (refs >= len(chunks))).any()
    ):
        raise ValueError("missing decision-memory chunk reference")
    prefixes: list[Prefix] = []
    for row, identity in enumerate(archive.identities):
        tables: list[TableTail | None] = []
        for field, key in enumerate(MEMORY_FIELDS):
            position = row * len(MEMORY_FIELDS) + field
            start, stop = offsets[position : position + 2]
            tail: TableTail | None = None
            for index in refs[start:stop]:
                chunk = chunks[int(index)]
                if chunk.rows.shape[1] != MEMORY_WIDTHS[key]:
                    raise ValueError("invalid decision-memory chunk role")
                tail = store._append(tail, chunk)
            if tail and tail.count > archive.capacities[field]:
                raise ValueError("decision-memory capacity mismatch")
            tables.append(tail)
        prefix = store._prefix(identity, tuple(tables))
        admit_memory(gather_memory([prefix]))
        prefixes.append(prefix)
    return tuple(prefixes)
