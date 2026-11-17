"""Immutable raw prefix transport for records, snapshots, programs and members.

Record-aligned chunks share unchanged prefixes, including coalesced versions.
Selected execution rows assemble cached immutable features without dense history;
explicit exports and archives retain exact bytes. Dense imports check padding.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from hashlib import sha256
import sys
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import numpy as np

from manabot.env.decision_memory import MEMORY_FIELDS, MEMORY_WIDTHS
from manabot.policy.decision_memory import (
    MemoryCursor,
    admit_chunk,
    live_memory,
    record_chunks,
)

if TYPE_CHECKING:
    from manabot.policy.decision_memory import MemoryFeatures

MemoryCapacities = tuple[int, int, int, int]
_CHUNK_ROWS = 32


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class Chunk:
    """One exact-byte record run and all companion rows it introduces."""

    key: str
    rows: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if len(self.rows) != 4 or not 0 < len(self.rows[0]) <= _CHUNK_ROWS:
            raise ValueError("invalid decision-memory chunk shape")
        digest = sha256()
        immutable = []
        for key, rows in zip(MEMORY_FIELDS, self.rows, strict=True):
            if (
                rows.ndim != 2
                or rows.shape[1] != MEMORY_WIDTHS[key]
                or rows.dtype != np.float32
            ):
                raise ValueError("invalid decision-memory chunk shape")
            data = rows.tobytes()
            digest.update(str(rows.shape).encode())
            digest.update(data)
            immutable.append(np.frombuffer(data, np.float32).reshape(rows.shape))
        if digest.hexdigest() != self.key:
            raise ValueError("corrupt decision-memory chunk")
        object.__setattr__(self, "rows", tuple(immutable))


@dataclass
class ChunkFeatures:
    """One vocabulary's admitted immutable features, owned by a reachable node."""

    admission: tuple[str, int, bool] | None = None
    prepared: "tuple[str, MemoryFeatures[np.ndarray]] | None" = None

    @property
    def nbytes(self) -> int:
        cached = self.prepared
        if cached is None:
            return 0
        fragment = cached[1]
        return sum(
            value.nbytes
            for f in fields(fragment)
            if isinstance(value := getattr(fragment, f.name), np.ndarray)
        )


@dataclass(frozen=True, slots=True)
class PrefixIdentity:
    game: str
    viewer: int
    branch: str

    def __post_init__(self) -> None:
        if not self.game or not self.branch or self.viewer not in (0, 1):
            raise ValueError("invalid decision-memory prefix identity")


@dataclass(frozen=True, eq=False, slots=True, weakref_slot=True)
class Tail:
    """An admitted chunk whose key binds its exact predecessor and companions."""

    key: str
    previous: "Tail | None"
    chunk: Chunk
    cursor: MemoryCursor = field(init=False)
    # Cache only the current vocabulary on this node; no cross-game owner.
    features: ChunkFeatures

    def __post_init__(self) -> None:
        key = sha256(
            ((self.previous.key if self.previous else "") + self.chunk.key).encode()
        ).hexdigest()
        cursor = admit_chunk(
            self.chunk.rows, self.previous.cursor if self.previous else MemoryCursor()
        )
        if self.key != key:
            raise ValueError("invalid decision-memory chunk certificate")
        object.__setattr__(self, "cursor", cursor)

    def chunks(self) -> Iterator["Tail"]:
        nodes: list[Tail] = []
        node: Tail | None = self
        while node is not None:
            nodes.append(node)
            node = node.previous
        yield from reversed(nodes)


@dataclass(frozen=True, slots=True)
class Prefix:
    identity: PrefixIdentity
    tail: Tail | None

    @property
    def counts(self) -> tuple[int, ...]:
        c = self.tail.cursor if self.tail else MemoryCursor()
        return c.records, c.details, c.references, c.members

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
        if any(
            len(v) != len(prefixes) for k, v in rows.items() if k not in MEMORY_FIELDS
        ):
            raise ValueError("decision-memory prefix batch mismatch")
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
    if isinstance(rows, CompactObservation):
        selected = CompactObservation(
            {key: value[indexes] for key, value in rows.items()},
            [rows.prefixes[int(i)] for i in indexes],
            rows.memory_capacities,
        )
        # An explicit export is arbitrary dense input, including edits made
        # after export. Never replace those bytes with its original prefix.
        selected.update({key: rows[key][indexes] for key in MEMORY_FIELDS})
        return selected
    if isinstance(rows, PrefixObservation) and not any(
        key in rows for key in MEMORY_FIELDS
    ):
        return PrefixObservation(
            {k: v[indexes] for k, v in rows.items() if k not in MEMORY_FIELDS},
            [rows.prefixes[int(i)] for i in indexes],
            rows.memory_capacities,
        )
    return {k: v[indexes] for k, v in rows.items()}


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
        cursors = [0] * 4
        for node in prefix.tail.chunks() if prefix.tail else ():
            for axis, (key, rows) in enumerate(
                zip(MEMORY_FIELDS, node.chunk.rows, strict=True)
            ):
                start = cursors[axis]
                result[key][row, start : start + len(rows)] = rows
                cursors[axis] += len(rows)
    return result


@dataclass(frozen=True)
class _StreamCapture:
    epoch: int
    prefix: Prefix


class MemoryStore:
    """Weak interning plus at most two current native prefixes per stream."""

    def __init__(self, capacities: MemoryCapacities) -> None:
        if len(capacities) != len(MEMORY_FIELDS) or min(capacities) < 1:
            raise ValueError("decision-memory capacities must be positive")
        self.capacities = capacities
        self._tails: WeakValueDictionary[str, Tail] = WeakValueDictionary()
        self._chunks: WeakValueDictionary[str, Chunk] = WeakValueDictionary()
        # At most two current prefixes per vector slot, never a game-wide index.
        self._streams: dict[tuple[int, int], _StreamCapture] = {}

    def _intern(self, rows: tuple[np.ndarray, ...]) -> Chunk:
        digest = sha256()
        blocks = []
        for block in rows:
            data = block.astype(np.float32, copy=False).tobytes()
            digest.update(str(block.shape).encode())
            digest.update(data)
            blocks.append(data)
        key = digest.hexdigest()
        previous = self._chunks.get(key)
        if previous is not None:
            if any(
                a.tobytes() != b for a, b in zip(previous.rows, blocks, strict=True)
            ):
                raise ValueError("decision-memory chunk hash collision")
            return previous
        chunk = Chunk(
            key,
            tuple(
                np.frombuffer(data, np.float32).reshape(block.shape)
                for data, block in zip(blocks, rows, strict=True)
            ),
        )
        self._chunks[key] = chunk
        return chunk

    def _append(self, previous: Tail | None, chunk: Chunk) -> Tail:
        key = sha256(
            ((previous.key if previous else "") + chunk.key).encode()
        ).hexdigest()
        tail = self._tails.get(key)
        if tail is not None:
            if tail.previous is not previous or tail.chunk is not chunk:
                raise ValueError("decision-memory prefix hash collision")
            return tail
        tail = Tail(key, previous, chunk, ChunkFeatures())
        self._tails[key] = tail
        return tail

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
        rows = live_memory(selected)[0]
        tail = None
        for blocks in record_chunks(rows, _CHUNK_ROWS):
            tail = self._append(tail, self._intern(blocks))
        return Prefix(identity, tail)

    def capture_live(
        self,
        buffers: Mapping[str, np.ndarray],
        row: int,
        identity: PrefixIdentity,
        *,
        epoch: int,
        counts: Sequence[int],
    ) -> Prefix:
        """Extend a synchronous native stream read using its generation and counts.

        Only native caller-owned buffers qualify: arbitrary imports use capture.
        A generation changes on reset/fork/replacement, including repeated seeds.
        Coalescing can rewrite only the last record, so always re-read its chunk.
        The native read and buffer write must describe the same unstepped env.
        """
        if len(counts) != 4 or any(
            n < 0 or n > cap for n, cap in zip(counts, self.capacities, strict=True)
        ):
            raise ValueError("decision-memory native counts exceed capacity")
        for key, capacity in zip(MEMORY_FIELDS, self.capacities, strict=True):
            if buffers[key].dtype != np.float32 or buffers[key][row].shape != (
                capacity,
                MEMORY_WIDTHS[key],
            ):
                raise ValueError(
                    "decision-memory capacity mismatch or non-float32 transport"
                )
        previous = self._streams.get((row, identity.viewer))
        tail = None
        if previous is not None and previous.epoch == epoch:
            if previous.prefix.identity != identity or any(
                n < old for n, old in zip(counts, previous.prefix.counts, strict=True)
            ):
                raise ValueError("decision-memory native stream regressed within epoch")
            if previous.prefix.tail is not None:
                tail = previous.prefix.tail.previous
        cursor = tail.cursor if tail else MemoryCursor()
        rows = tuple(
            buffers[key][row, :count]
            for key, count in zip(MEMORY_FIELDS, counts, strict=True)
        )
        for blocks in record_chunks(rows, _CHUNK_ROWS, cursor):
            tail = self._append(tail, self._intern(blocks))
        prefix = Prefix(identity, tail)
        self._streams[row, identity.viewer] = _StreamCapture(epoch, prefix)
        return prefix

    def measure(self, prefixes: Sequence[Prefix]) -> dict[str, int]:
        chunks: dict[str, Chunk] = {}
        tails: dict[str, Tail] = {}
        owned = {id(p): p for p in prefixes}
        owned.update((id(c.prefix), c.prefix) for c in self._streams.values())
        for prefix in owned.values():
            tail = prefix.tail
            while tail is not None and tail.key not in tails:
                tails[tail.key] = tail
                chunks[tail.chunk.key] = tail.chunk
                tail = tail.previous
        metadata = sys.getsizeof(prefixes) + sum(
            sys.getsizeof(p)
            + sys.getsizeof(p.identity)
            + len(p.identity.game)
            + len(p.identity.branch)
            for p in owned.values()
        )
        metadata += sum(
            sys.getsizeof(node)
            + sys.getsizeof(node.key)
            + sys.getsizeof(node.cursor)
            + sys.getsizeof(node.features)
            for node in tails.values()
        )
        metadata += sum(
            sys.getsizeof(c)
            + sys.getsizeof(c.key)
            + sys.getsizeof(c.rows)
            + sum(sys.getsizeof(a) + sys.getsizeof(a.base) + 33 for a in c.rows)
            for c in chunks.values()
        )
        return {
            "unique_chunk_bytes": sum(
                a.nbytes for c in chunks.values() for a in c.rows
            ),
            "chunk_count": len(chunks),
            "tail_count": len(tails),
            "python_metadata_bytes": metadata,
            "feature_cache_bytes": sum(node.features.nbytes for node in tails.values()),
            "intern_index_bytes": sum(
                sys.getsizeof(index.data)
                + sum(sys.getsizeof(ref) for ref in index.data.values())
                for index in (self._tails, self._chunks)
            ),
            "stream_capture_bytes": sys.getsizeof(self._streams)
            + sum(
                sys.getsizeof(k) + sys.getsizeof(c) for k, c in self._streams.items()
            ),
            "selected_padding_bytes": gather_bytes(prefixes),
        }


@dataclass(frozen=True)
class MemoryArchive:
    """Each prefix/table lists its exact ordered chunks in one shared dictionary."""

    capacities: MemoryCapacities
    identities: tuple[PrefixIdentity, ...]
    keys: tuple[str, ...]
    chunks: tuple[tuple[np.ndarray, ...], ...]
    offsets: np.ndarray
    indexes: np.ndarray

    @property
    def nbytes(self) -> int:
        return (
            sum(a.nbytes for c in self.chunks for a in c)
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
        for node in prefix.tail.chunks() if prefix.tail else ():
            chunk = node.chunk
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
            len(rows) != 4
            or any(
                a.ndim != 2
                or a.dtype != np.float32
                or a.shape[1] != MEMORY_WIDTHS[field]
                for field, a in zip(MEMORY_FIELDS, rows, strict=True)
            )
            or not 0 < len(rows[0]) <= _CHUNK_ROWS
        ):
            raise ValueError("invalid decision-memory chunk shape")
        # Archive blocks are arbitrary input: validate integers and validity.
        dense = {field: a[None] for field, a in zip(MEMORY_FIELDS, rows, strict=True)}
        live = live_memory(dense)[0]
        if any(len(a) != len(b) for a, b in zip(live, rows, strict=True)):
            raise ValueError("invalid decision-memory chunk padding")
        chunk = store._intern(rows)
        if key != chunk.key:
            raise ValueError("corrupt decision-memory chunk")
        chunks.append(chunk)
    offsets, refs = archive.offsets, archive.indexes
    if (
        offsets.dtype != np.int64
        or offsets.shape != (len(archive.identities) + 1,)
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
        tail = None
        start, stop = offsets[row : row + 2]
        for index in refs[start:stop]:
            tail = store._append(tail, chunks[int(index)])
        prefix = Prefix(identity, tail)
        if any(
            count > cap
            for count, cap in zip(prefix.counts, archive.capacities, strict=True)
        ):
            raise ValueError("decision-memory capacity mismatch")
        prefixes.append(prefix)
    return tuple(prefixes)
