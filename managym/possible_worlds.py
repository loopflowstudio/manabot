"""Read-only Python adapter for managym's canonical possible-world space.

Rust owns enumeration, ordering, exact compatible-deal weights, identity, and
materialization.  This module only parses that projection and routes canonical
world indexes back to the source engine; it never constructs a hand ontology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import sys
from typing import Any, Mapping, Sequence

POSSIBLE_WORLD_SPACE_VERSION: int = 1


class PossibleWorldError(ValueError):
    """The canonical world-space contract rejected a consumer request."""


@dataclass(frozen=True, slots=True)
class WorldQuery:
    """Typed input to managym's authoritative world-query evaluator."""

    kind: str
    card: str | None = None
    count: int | None = None

    @classmethod
    def true(cls) -> "WorldQuery":
        return cls("true")

    @classmethod
    def has(cls, card: str, at_least: int = 1) -> "WorldQuery":
        return cls("has", card, at_least)

    @classmethod
    def lacks(cls, card: str, fewer_than: int = 1) -> "WorldQuery":
        return cls("lacks", card, fewer_than)

    @classmethod
    def exactly(cls, card: str, count: int) -> "WorldQuery":
        return cls("exactly", card, count)

    @classmethod
    def not_exactly(cls, card: str, count: int) -> "WorldQuery":
        return cls("not_exactly", card, count)

    def to_dict(self) -> dict[str, str | int]:
        if self.kind == "true":
            return {"kind": "true"}
        if not self.card or self.count is None or self.count < 0:
            raise PossibleWorldError(f"invalid {self.kind!r} query")
        count_field = {
            "has": "at_least",
            "lacks": "fewer_than",
            "exactly": "count",
            "not_exactly": "count",
        }.get(self.kind)
        if count_field is None:
            raise PossibleWorldError(f"unknown query kind {self.kind!r}")
        return {"kind": self.kind, "card": self.card, count_field: self.count}


@dataclass(frozen=True, slots=True)
class SupportReceipt:
    space_identity: str
    query_digest: str
    canonical_digest: str
    canonical_query: Mapping[str, Any] | str
    support_size: int
    total_weight: int


@dataclass(frozen=True, slots=True)
class PossibleWorld:
    """One ordered canonical hand-count hypothesis and its exact weight."""

    index: int
    hand: tuple[tuple[str, int], ...]
    weight: int

    def count(self, card: str) -> int:
        return dict(self.hand).get(card, 0)


@dataclass(frozen=True, slots=True)
class PossibleWorldSpace:
    """Identity-bound view of one managym-owned hypothesis domain."""

    identity: str
    viewer: int
    opponent: int
    source_revision: int
    source_viewer_state_hash: str
    hand_size: int
    pool: tuple[tuple[str, int], ...]
    total_weight: int
    worlds: tuple[PossibleWorld, ...]
    _engine: Any = field(repr=False, compare=False, hash=False)
    _allocated_bytes: int = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_allocated_bytes", self._measure_allocated_bytes())

    @classmethod
    def from_engine(cls, engine: Any, viewer: int) -> "PossibleWorldSpace":
        try:
            payload = json.loads(engine.possible_world_space_json(viewer))
        except Exception as error:
            raise PossibleWorldError(str(error)) from error
        version = int(payload["schema_version"])
        if version != POSSIBLE_WORLD_SPACE_VERSION:
            raise PossibleWorldError(f"unsupported PossibleWorldSpace schema {version}")
        source = payload["source_observation"]
        if int(source["viewer"]) != int(payload["viewer"]):
            raise PossibleWorldError("source observation changed the space viewer")
        rows = tuple(
            PossibleWorld(
                index=int(row["index"]),
                hand=tuple(
                    (str(name), int(count))
                    for name, count in sorted(row["hand"].items())
                ),
                weight=int(row["weight"]),
            )
            for row in payload["worlds"]
        )
        if not rows or tuple(row.index for row in rows) != tuple(range(len(rows))):
            raise PossibleWorldError(
                "world rows must be non-empty and canonically indexed"
            )
        if any(
            sum(count for _, count in row.hand) != int(payload["hand_size"])
            for row in rows
        ):
            raise PossibleWorldError("a canonical world has the wrong hand size")
        total_weight = int(payload["total_weight"])
        if total_weight <= 0:
            raise PossibleWorldError("possible-world space has no compatible deals")
        if sum(row.weight for row in rows) != total_weight:
            raise PossibleWorldError("world weights do not sum to total_weight")
        return cls(
            identity=str(payload["identity"]),
            viewer=int(payload["viewer"]),
            opponent=int(payload["opponent"]),
            source_revision=int(source["revision"]),
            source_viewer_state_hash=str(source["viewer_state_hash"]),
            hand_size=int(payload["hand_size"]),
            pool=tuple(
                (str(name), int(count))
                for name, count in sorted(payload["pool"].items())
            ),
            total_weight=total_weight,
            worlds=rows,
            _engine=engine,
        )

    @property
    def support_size(self) -> int:
        return len(self.worlds)

    @property
    def allocated_bytes(self) -> int:
        """Owned Python projection memory, excluding the source engine."""

        return self._allocated_bytes

    def _measure_allocated_bytes(self) -> int:
        total = sys.getsizeof(self)
        total += sum(
            sys.getsizeof(value)
            for value in (
                self.identity,
                self.viewer,
                self.opponent,
                self.source_revision,
                self.source_viewer_state_hash,
                self.hand_size,
                self.total_weight,
            )
        )
        total += sys.getsizeof(self.pool)
        for name, count in self.pool:
            total += sys.getsizeof((name, count))
            total += sys.getsizeof(name) + sys.getsizeof(count)
        total += sys.getsizeof(self.worlds)
        for world in self.worlds:
            total += sys.getsizeof(world)
            total += sys.getsizeof(world.index) + sys.getsizeof(world.weight)
            total += sys.getsizeof(world.hand)
            for name, count in world.hand:
                total += sys.getsizeof((name, count))
                total += sys.getsizeof(name) + sys.getsizeof(count)
        return total

    @property
    def library_size(self) -> int:
        return sum(count for _, count in self.pool) - self.hand_size

    def world(self, index: int) -> PossibleWorld:
        if index < 0 or index >= len(self.worlds):
            raise PossibleWorldError(f"world index {index} is outside the space")
        return self.worlds[index]

    def materialize(
        self,
        index: int,
        *,
        seed: int,
        refresh_opponent_commitment: bool = False,
    ) -> Any:
        self.world(index)
        try:
            return self._engine.materialize_possible_world(
                self.viewer,
                self.identity,
                index,
                seed,
                refresh_opponent_commitment,
            )
        except Exception as error:
            raise PossibleWorldError(str(error)) from error

    def support(self, query: WorldQuery) -> SupportReceipt:
        try:
            payload = json.loads(
                self._engine.possible_world_support_json(
                    self.viewer,
                    self.identity,
                    json.dumps(query.to_dict(), sort_keys=True, separators=(",", ":")),
                )
            )
        except Exception as error:
            raise PossibleWorldError(str(error)) from error
        if payload["space_identity"] != self.identity:
            raise PossibleWorldError("query receipt changed space identity")
        return SupportReceipt(
            space_identity=str(payload["space_identity"]),
            query_digest=str(payload["query_digest"]),
            canonical_digest=str(payload["canonical_digest"]),
            canonical_query=payload["canonical_query"],
            support_size=int(payload["support_size"]),
            total_weight=int(payload["total_weight"]),
        )

    def flat_mc_scores(
        self,
        indexes: Sequence[int],
        seeds: Sequence[int],
        *,
        rollouts: int,
        max_steps: int,
    ) -> tuple[list[float], int, int]:
        canonical_indexes = [int(index) for index in indexes]
        for index in canonical_indexes:
            self.world(index)
        if len(canonical_indexes) != len(seeds):
            raise PossibleWorldError("world indexes and seeds must have equal length")
        try:
            scores, simulations, cap_hits = self._engine.flat_mc_scores_for_worlds(
                self.viewer,
                self.identity,
                canonical_indexes,
                [int(seed) for seed in seeds],
                int(rollouts),
                int(max_steps),
            )
        except Exception as error:
            raise PossibleWorldError(str(error)) from error
        return list(scores), int(simulations), int(cap_hits)

    def source_identity(self) -> Mapping[str, int | str]:
        return {
            "revision": self.source_revision,
            "viewer": self.viewer,
            "viewer_state_hash": self.source_viewer_state_hash,
        }
