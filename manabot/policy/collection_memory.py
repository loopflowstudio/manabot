"""Lifetime-bound, weak cache of stable decision records for a frozen GRU policy."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, Self
from weakref import WeakKeyDictionary, ref

from manabot.env.memory_store import PrefixIdentity, RecordNode, record_tail
from manabot.policy.decision_memory import retained_memory

if TYPE_CHECKING:
    from manabot.policy.prepare import Prepared
    from manabot.policy.spans import PolicyObservation


class CacheArray(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: slice, /) -> Self: ...


@dataclass(frozen=True)
class CachedNode[A]:
    lifetime: tuple[PrefixIdentity, int | None]
    references: A
    state: A


class CollectionMemory[A: CacheArray]:
    """Weak record results for one inference scope; current last records never enter.

    Scope guards bind parameter versions, device and module modes. Values own only
    detached states and cumulative reference joins, never their weak keys or a
    whole execution tree. Stable roots replace all cached ancestry in preparation.
    """

    def __init__(
        self,
        concatenate: Callable[[list[A]], A],
        stack: Callable[[list[A]], A],
        state_at: Callable[[A, int], A],
        materialize: Callable[[list[A]], None] | None = None,
    ) -> None:
        self.concatenate = concatenate
        self.stack = stack
        self.state_at = state_at
        self.materialize = materialize
        self.nodes: WeakKeyDictionary[RecordNode, CachedNode[A]] = WeakKeyDictionary()

    def clear(self) -> None:
        self.nodes.clear()

    def roots(self, obs: "PolicyObservation") -> dict[str, int]:
        retained = retained_memory(obs)
        if retained is None:
            return {}
        tails = [
            (tail, (prefix.identity, prefix.native_epoch))
            for prefix in retained.prefixes
            if (tail := record_tail(prefix)) is not None
        ]
        rejected = {tail.key for tail, _ in tails}
        # A root is shared by every prefix that reaches it. If a later prefix
        # rejects an earlier selection, look past that root on the next pass.
        while True:
            roots: dict[str, int] = {}
            for tail, lifetime in tails:
                node: RecordNode | None = tail
                while node is not None:
                    cached = self.nodes.get(node)
                    if (
                        node.key not in rejected
                        and cached is not None
                        and cached.lifetime == lifetime
                    ):
                        roots[node.key] = len(cached.references)
                        break
                    rejected.add(node.key)
                    node = node.previous
            if roots.keys().isdisjoint(rejected):
                return roots

    def bind(
        self, batch: "Prepared[A]", obs: "PolicyObservation", roots: Mapping[str, int]
    ) -> "Prepared[A]":
        """Attach the frozen states selected when this batch was prepared."""
        retained = retained_memory(obs)
        if retained is None or batch.memory is None or batch.memory.fold is None:
            return batch
        fold = batch.memory.fold
        nodes: dict[str, RecordNode] = {}
        lifetimes: dict[str, tuple[PrefixIdentity, int | None]] = {}
        open_keys: set[str] = set()
        for prefix in retained.prefixes:
            node = record_tail(prefix)
            if node is not None:
                open_keys.add(node.key)
            while node is not None and node.key not in nodes:
                nodes[node.key] = node
                lifetimes[node.key] = (prefix.identity, prefix.native_epoch)
                if node.key in roots:
                    break
                node = node.previous
        records = fold.serving_records
        cached = [self.nodes[nodes[r.key]] for r in records if r.key in roots]
        pending = [
            (index, ref(nodes[r.key]), lifetimes[r.key])
            for index, r in enumerate(records, start=1)
            if r.key not in open_keys and r.key not in roots
        ]

        def retain(states: A, references: A) -> None:
            # Adjacent records usually introduce no snapshots. Share their
            # cumulative join instead of copying the entire reference prefix.
            joined = [references[:0]]  # State zero has no predecessor references.
            for record in records:
                previous = joined[record.parent]
                if record.reference_count:
                    start = record.reference_start
                    current = references[start : start + record.reference_count]
                    previous = self.concatenate([previous, current])
                joined.append(previous)
            retained_arrays: list[A] = []
            for index, weak, lifetime in pending:
                node = weak()
                if node is not None:
                    state = self.state_at(states, index)
                    self.nodes[node] = CachedNode(lifetime, joined[index], state)
                    if self.materialize is not None:
                        retained_arrays.extend((joined[index], state))
            if retained_arrays and self.materialize is not None:
                # Lazy backends release the fold graph once these independent roots exist.
                self.materialize(retained_arrays)

        fold = replace(
            fold,
            root_references=self.concatenate([node.references for node in cached])
            if cached
            else None,
            root_states=self.stack([node.state for node in cached]) if cached else None,
            retain=retain,
        )
        return replace(batch, memory=replace(batch.memory, fold=fold))
