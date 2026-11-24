"""Lifetime-bound, weak cache of stable decision records for a frozen GRU policy."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, Self
from weakref import WeakKeyDictionary, ref

from manabot.env.memory_store import PrefixIdentity, RecordNode, record_tail
from manabot.policy.decision_memory import retained_memory

if TYPE_CHECKING:
    from manabot.env.observation import ObservationBuffers
    from manabot.policy.prepare import Prepared


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

    def roots(self, obs: "ObservationBuffers") -> dict[str, int]:
        retained = retained_memory(obs)
        if retained is None:
            return {}
        roots: dict[str, int] = {}
        rejected = {
            tail.key for prefix in retained.prefixes if (tail := record_tail(prefix))
        }
        for prefix in retained.prefixes:
            node = record_tail(prefix)
            while node is not None:
                cached = self.nodes.get(node)
                if (
                    node.key not in rejected
                    and cached is not None
                    and cached.lifetime == (prefix.identity, prefix.native_epoch)
                ):
                    roots[node.key] = len(cached.references)
                    break
                rejected.add(node.key)
                roots.pop(node.key, None)
                node = node.previous
        # Rejecting one root can expose another prefix's deeper candidate.
        # Recheck only the selected suffixes until every reachable root has the
        # serving lifetime of every prefix that uses it. Normal batches stop at
        # the same boundary; mixed fork/reset lifetimes may remove more roots.
        changed = True
        while changed and roots:
            changed = False
            for prefix in retained.prefixes:
                node = record_tail(prefix)
                while node is not None:
                    if node.key in roots:
                        if self.nodes[node].lifetime != (
                            prefix.identity,
                            prefix.native_epoch,
                        ):
                            del roots[node.key]
                            changed = True
                        else:
                            break
                    node = node.previous
        return roots

    def bind(self, batch: "Prepared[A]", obs: "ObservationBuffers") -> "Prepared[A]":
        retained = retained_memory(obs)
        if retained is None or batch.memory is None or batch.memory.fold is None:
            return batch
        fold = batch.memory.fold
        root_keys = {
            key
            for index, key in enumerate(fold.keys)
            if fold.record_slices[index][0] < 0
        }
        nodes: dict[str, RecordNode] = {}
        lifetimes: dict[str, tuple[PrefixIdentity, int | None]] = {}
        for prefix in retained.prefixes:
            node = record_tail(prefix)
            while node is not None and node.key not in nodes:
                nodes[node.key] = node
                lifetimes[node.key] = (prefix.identity, prefix.native_epoch)
                if node.key in root_keys:
                    break
                node = node.previous
        cached = [self.nodes[nodes[key]] for key in fold.keys if key in root_keys]
        open_keys = {
            tail.key for prefix in retained.prefixes if (tail := record_tail(prefix))
        }
        stable = {
            node.previous.key for node in nodes.values() if node.previous is not None
        } - open_keys
        pending = [
            (i, ref(nodes[key]), lifetimes[key])
            for i, key in enumerate(fold.keys)
            if key in stable and key not in root_keys
        ]

        reference_slices = fold.reference_slices
        indexes = {key: i for i, key in enumerate(fold.keys)}
        parents = [
            indexes[node.previous.key] if node.previous and key not in root_keys else -1
            for key in fold.keys
            for node in (nodes[key],)
        ]

        def retain(states: A, references: A) -> None:
            # Adjacent records usually introduce no snapshots. Share their
            # cumulative join instead of copying the entire reference prefix.
            joined: list[A] = []
            for index, (start, count) in enumerate(reference_slices):
                parent = parents[index]
                previous = joined[parent] if parent >= 0 else references[:0]
                current = references[start : start + count]
                joined.append(
                    self.concatenate([previous, current]) if count else previous
                )
            retained_arrays: list[A] = []
            for index, weak, lifetime in pending:
                node = weak()
                if node is not None:
                    state = self.state_at(states, index + 1)
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
