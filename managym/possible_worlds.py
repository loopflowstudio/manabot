"""Read-only Python adapter for managym's canonical possible-world space.

Rust remains the production authority for enumeration, ordering, exact
compatible-deal weights, and query evaluation. The fixture constructor exists
only for retained decisions and focused contract tests; it applies the same
typed count-query grammar without consulting an actual hidden world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping

POSSIBLE_WORLD_SPACE_VERSION: int = 1
WORLD_SCHEMA_IDENTITY: str = "managym.possible-world-space/v1"


class PossibleWorldError(ValueError):
    """The canonical world-space contract rejected a consumer request."""


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


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

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


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
    source_history_identity: str
    hand_size: int
    pool: tuple[tuple[str, int], ...]
    total_weight: int
    worlds: tuple[PossibleWorld, ...]
    world_schema_identity: str = WORLD_SCHEMA_IDENTITY
    content_manifest_identity: str = ""
    _engine: Any | None = field(default=None, repr=False, compare=False, hash=False)
    _authority_identity: str | None = field(
        default=None, repr=False, compare=False, hash=False
    )

    @classmethod
    def from_engine(
        cls, engine: Any, viewer: int, *, viewer_history_identity: str
    ) -> "PossibleWorldSpace":
        """Load the production projection emitted by managym's Rust authority."""

        try:
            payload = json.loads(engine.possible_world_space_json(viewer))
        except Exception as error:
            raise PossibleWorldError(str(error)) from error
        version = int(payload["schema_version"])
        if version != POSSIBLE_WORLD_SPACE_VERSION:
            raise PossibleWorldError(f"unsupported PossibleWorldSpace schema {version}")
        source = payload["source_observation"]
        authority_identity = str(payload["identity"])
        canonical_pool = tuple(
            (str(name), int(count)) for name, count in sorted(payload["pool"].items())
        )
        content_manifest_identity = str(
            payload.get("content_manifest_identity", _digest({"pool": canonical_pool}))
        )
        bound_identity = _digest(
            {
                "authority_identity": authority_identity,
                "viewer_history_identity": viewer_history_identity,
                "world_schema_identity": WORLD_SCHEMA_IDENTITY,
                "content_manifest_identity": content_manifest_identity,
            }
        )
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
        space = cls(
            identity=bound_identity,
            viewer=int(payload["viewer"]),
            opponent=int(payload["opponent"]),
            source_revision=int(source["revision"]),
            source_viewer_state_hash=str(source["viewer_state_hash"]),
            source_history_identity=viewer_history_identity,
            hand_size=int(payload["hand_size"]),
            pool=canonical_pool,
            total_weight=int(payload["total_weight"]),
            worlds=rows,
            content_manifest_identity=content_manifest_identity,
            _engine=engine,
            _authority_identity=authority_identity,
        )
        space._validate()
        return space

    @classmethod
    def from_fixture(
        cls,
        *,
        viewer: int,
        source_revision: int,
        source_viewer_state_hash: str,
        source_history_identity: str,
        pool: Mapping[str, int],
        hands: Iterable[tuple[Mapping[str, int], int]],
        content_manifest_identity: str | None = None,
    ) -> "PossibleWorldSpace":
        """Build a retained, viewer-safe contract fixture.

        This constructor accepts the complete canonical support, never an
        actual hidden hand. Production callers use :meth:`from_engine`.
        """

        canonical_pool = tuple(
            (str(name), int(count)) for name, count in sorted(pool.items())
        )
        manifest_identity = content_manifest_identity or _digest(
            {"pool": canonical_pool}
        )
        raw_rows = []
        for hand, weight in hands:
            if any(int(count) < 0 for count in hand.values()):
                raise PossibleWorldError("fixture hand counts must be non-negative")
            raw_rows.append(
                (
                    tuple(
                        (str(name), int(count))
                        for name, count in sorted(hand.items())
                        if int(count) > 0
                    ),
                    int(weight),
                )
            )
        raw_rows.sort()
        worlds = tuple(
            PossibleWorld(index=index, hand=hand, weight=weight)
            for index, (hand, weight) in enumerate(raw_rows)
        )
        if not worlds:
            raise PossibleWorldError("fixture worlds must be non-empty")
        hand_sizes = {sum(count for _, count in world.hand) for world in worlds}
        if len(hand_sizes) != 1:
            raise PossibleWorldError("fixture worlds must have one public hand size")
        hand_size = next(iter(hand_sizes))
        identity_payload = {
            "schema_version": POSSIBLE_WORLD_SPACE_VERSION,
            "world_schema_identity": WORLD_SCHEMA_IDENTITY,
            "viewer": int(viewer),
            "opponent": 1 - int(viewer),
            "source_revision": int(source_revision),
            "source_viewer_state_hash": source_viewer_state_hash,
            "source_history_identity": source_history_identity,
            "content_manifest_identity": manifest_identity,
            "hand_size": hand_size,
            "pool": canonical_pool,
            "worlds": [
                {"hand": world.hand, "weight": world.weight} for world in worlds
            ],
        }
        space = cls(
            identity=_digest(identity_payload),
            viewer=int(viewer),
            opponent=1 - int(viewer),
            source_revision=int(source_revision),
            source_viewer_state_hash=source_viewer_state_hash,
            source_history_identity=source_history_identity,
            hand_size=hand_size,
            pool=canonical_pool,
            total_weight=sum(world.weight for world in worlds),
            worlds=worlds,
            content_manifest_identity=manifest_identity,
        )
        space._validate()
        return space

    def _validate(self) -> None:
        if self.viewer == self.opponent:
            raise PossibleWorldError("viewer and opponent must differ")
        if not self.worlds:
            raise PossibleWorldError("world rows must be non-empty")
        if not self.content_manifest_identity:
            raise PossibleWorldError("world space needs a content manifest identity")
        if any(count <= 0 for _, count in self.pool):
            raise PossibleWorldError("unseen-pool counts must be positive")
        if tuple(world.index for world in self.worlds) != tuple(
            range(len(self.worlds))
        ):
            raise PossibleWorldError("world rows must be canonically indexed")
        if self.total_weight <= 0 or any(world.weight <= 0 for world in self.worlds):
            raise PossibleWorldError("compatible-deal weights must be positive")
        if sum(world.weight for world in self.worlds) != self.total_weight:
            raise PossibleWorldError("world weights do not sum to total_weight")
        if len({world.hand for world in self.worlds}) != len(self.worlds):
            raise PossibleWorldError("canonical world rows must be unique")
        pool = dict(self.pool)
        for world in self.worlds:
            if sum(count for _, count in world.hand) != self.hand_size:
                raise PossibleWorldError("a canonical world has the wrong hand size")
            if any(
                count < 0 or count > pool.get(name, 0) for name, count in world.hand
            ):
                raise PossibleWorldError("a canonical world is outside the unseen pool")

    @property
    def support_size(self) -> int:
        return len(self.worlds)

    def world(self, index: int) -> PossibleWorld:
        if index < 0 or index >= len(self.worlds):
            raise PossibleWorldError(f"world index {index} is outside the space")
        return self.worlds[index]

    def support(self, query: WorldQuery) -> SupportReceipt:
        return self.condition_indexes(query, allow_empty=True)[1]

    def condition_indexes(
        self, query: WorldQuery, *, allow_empty: bool = False
    ) -> tuple[tuple[int, ...], SupportReceipt]:
        """Return Rules-selected row indexes and their identity-bound receipt."""

        if self._engine is not None:
            indexes, receipt = self._condition_through_authority(query)
        else:
            indexes, receipt = self._condition_fixture(query)
        if not indexes and not allow_empty:
            raise PossibleWorldError("query has empty support")
        return indexes, receipt

    def _condition_through_authority(
        self, query: WorldQuery
    ) -> tuple[tuple[int, ...], SupportReceipt]:
        assert self._engine is not None and self._authority_identity is not None
        try:
            encoded_query = json.dumps(
                query.to_dict(), sort_keys=True, separators=(",", ":")
            )
            support_payload = json.loads(
                self._engine.possible_world_support_json(
                    self.viewer,
                    self._authority_identity,
                    encoded_query,
                )
            )
            if int(support_payload["support_size"]) == 0:
                return (), self._support_receipt(support_payload)
            payload = json.loads(
                self._engine.possible_world_condition_json(
                    self.viewer,
                    self._authority_identity,
                    encoded_query,
                )
            )
        except Exception as error:
            raise PossibleWorldError(str(error)) from error
        if payload["space_identity"] != self._authority_identity:
            raise PossibleWorldError("query receipt changed space identity")
        indexes = tuple(int(index) for index in payload["world_indexes"])
        receipt = self._support_receipt(payload)
        self._validate_indexes(indexes, receipt)
        return indexes, receipt

    def _support_receipt(self, payload: Mapping[str, Any]) -> SupportReceipt:
        return SupportReceipt(
            space_identity=self.identity,
            query_digest=str(payload["query_digest"]),
            canonical_digest=str(payload["canonical_digest"]),
            canonical_query=payload["canonical_query"],
            support_size=int(payload["support_size"]),
            total_weight=int(payload["total_weight"]),
        )

    def _condition_fixture(
        self, query: WorldQuery
    ) -> tuple[tuple[int, ...], SupportReceipt]:
        canonical = self._canonical_query(query)

        def selected(world: PossibleWorld) -> bool:
            kind = canonical["kind"]
            if kind == "true":
                return True
            if kind == "empty":
                return False
            count = world.count(str(canonical["card"]))
            threshold = int(canonical["count"])
            return {
                "has": count >= threshold,
                "lacks": count < threshold,
                "exactly": count == threshold,
                "not_exactly": count != threshold,
            }[kind]

        indexes = tuple(world.index for world in self.worlds if selected(world))
        receipt = SupportReceipt(
            space_identity=self.identity,
            query_digest=query.digest,
            canonical_digest=_digest(canonical),
            canonical_query=canonical,
            support_size=len(indexes),
            total_weight=sum(self.world(index).weight for index in indexes),
        )
        self._validate_indexes(indexes, receipt)
        return indexes, receipt

    def _canonical_query(self, query: WorldQuery) -> dict[str, str | int]:
        payload = query.to_dict()
        kind = str(payload["kind"])
        if kind == "true":
            return {"kind": "true"}
        card = str(payload["card"])
        count = int(
            next(value for key, value in payload.items() if key not in {"kind", "card"})
        )
        maximum = min(dict(self.pool).get(card, 0), self.hand_size)
        if kind == "has":
            if count == 0:
                return {"kind": "true"}
            if count > maximum:
                return {"kind": "empty"}
        elif kind == "lacks":
            if count == 0:
                return {"kind": "empty"}
            if count > maximum:
                return {"kind": "true"}
        elif kind == "exactly":
            if count > maximum:
                return {"kind": "empty"}
            if count == 0:
                return {"kind": "lacks", "card": card, "count": 1}
        elif kind == "not_exactly":
            if count > maximum:
                return {"kind": "true"}
            if count == 0:
                return {"kind": "has", "card": card, "count": 1}
        return {"kind": kind, "card": card, "count": count}

    def _validate_indexes(
        self, indexes: tuple[int, ...], receipt: SupportReceipt
    ) -> None:
        if len(indexes) != receipt.support_size:
            raise PossibleWorldError("conditioned indexes differ from support receipt")
        if indexes != tuple(sorted(set(indexes))):
            raise PossibleWorldError("conditioned indexes are not canonical")
        for index in indexes:
            self.world(index)

    def source_identity(self) -> Mapping[str, int | str]:
        return {
            "revision": self.source_revision,
            "viewer": self.viewer,
            "viewer_state_hash": self.source_viewer_state_hash,
            "viewer_history_identity": self.source_history_identity,
            "content_manifest_identity": self.content_manifest_identity,
        }


__all__ = [
    "POSSIBLE_WORLD_SPACE_VERSION",
    "PossibleWorld",
    "PossibleWorldError",
    "PossibleWorldSpace",
    "SupportReceipt",
    "WorldQuery",
]
