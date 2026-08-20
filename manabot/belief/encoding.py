"""Deterministic policy projection of canonical manabot beliefs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
import torch

from manabot.belief.range import BeliefError, BeliefState
from managym.possible_worlds import PossibleWorldSpace

OPPONENT_OWNER_ROLE_ID = 1
LIBRARY_ZONE_ID = 0
HAND_ZONE_ID = 1


def _readonly(values: NDArray) -> NDArray:
    values.setflags(write=False)
    return values


@dataclass(frozen=True, order=True, slots=True)
class BeliefRow:
    """One canonical viewer-relative hidden count row."""

    owner_role_id: int
    hidden_zone_id: int
    card_def_id: int
    card_name: str


@dataclass(frozen=True, slots=True)
class BeliefEncodingSchema:
    """Schema-bound row order and count vocabulary for model inputs."""

    schema_identity: str
    world_schema_identity: str
    content_manifest_identity: str
    rows: tuple[BeliefRow, ...]
    count_buckets: int

    def __post_init__(self) -> None:
        if self.count_buckets < 2:
            raise BeliefError("belief encoding needs at least two count buckets")
        if not self.rows:
            raise BeliefError("belief encoding rows must be non-empty")
        if tuple(sorted(self.rows)) != self.rows:
            raise BeliefError("belief encoding rows must be canonically ordered")
        if any(
            row.card_def_id < 0 or row.owner_role_id < 0 or row.hidden_zone_id < 0
            for row in self.rows
        ):
            raise BeliefError("belief semantic ids must be non-negative")
        keys = tuple(
            (row.owner_role_id, row.hidden_zone_id, row.card_def_id)
            for row in self.rows
        )
        if len(set(keys)) != len(keys):
            raise BeliefError(
                "belief rows must be unique by owner, hidden zone, and card definition"
            )
        definitions = {(row.card_def_id, row.card_name) for row in self.rows}
        if len({card_id for card_id, _ in definitions}) != len(definitions):
            raise BeliefError(
                "one card definition id cannot name multiple card definitions"
            )
        if len({name for _, name in definitions}) != len(definitions):
            raise BeliefError(
                "one card definition name cannot map to multiple definition ids"
            )

    @property
    def identity(self) -> str:
        payload = {
            "schema_identity": self.schema_identity,
            "world_schema_identity": self.world_schema_identity,
            "content_manifest_identity": self.content_manifest_identity,
            "rows": [
                {
                    "owner_role_id": row.owner_role_id,
                    "hidden_zone_id": row.hidden_zone_id,
                    "card_def_id": row.card_def_id,
                    "card_name": row.card_name,
                }
                for row in self.rows
            ],
            "count_buckets": self.count_buckets,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class BeliefCheckpointBinding:
    """Artifact identities that bind belief weights to their runtime schema."""

    schema_identity: str
    content_manifest_identity: str

    def __post_init__(self) -> None:
        if not self.schema_identity:
            raise BeliefError("checkpoint belief schema identity must be non-empty")
        if not self.content_manifest_identity:
            raise BeliefError(
                "checkpoint belief content manifest identity must be non-empty"
            )

    @classmethod
    def from_schema(cls, schema: BeliefEncodingSchema) -> BeliefCheckpointBinding:
        return cls(
            schema_identity=schema.identity,
            content_manifest_identity=schema.content_manifest_identity,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> BeliefCheckpointBinding:
        try:
            schema_identity = checkpoint["belief_schema_identity"]
            content_manifest_identity = checkpoint["belief_content_manifest_identity"]
        except KeyError as error:
            raise BeliefError(
                "belief-enabled checkpoint is missing its schema binding"
            ) from error
        if not isinstance(schema_identity, str) or not isinstance(
            content_manifest_identity, str
        ):
            raise BeliefError("checkpoint belief identities must be strings")
        return cls(schema_identity, content_manifest_identity)

    def checkpoint_fields(self) -> dict[str, str]:
        return {
            "belief_schema_identity": self.schema_identity,
            "belief_content_manifest_identity": self.content_manifest_identity,
        }

    def validate_schema(self, schema: BeliefEncodingSchema) -> None:
        if schema.content_manifest_identity != self.content_manifest_identity:
            raise BeliefError(
                "checkpoint belief content manifest does not match the runtime"
            )
        if schema.identity != self.schema_identity:
            raise BeliefError("checkpoint belief schema does not match the runtime")


def belief_checkpoint_fields(
    schema: BeliefEncodingSchema,
    *,
    count_buckets: int,
    card_vocab_size: int,
) -> dict[str, str]:
    """Validate a model/schema pair and return its serialized artifact binding."""

    if count_buckets != schema.count_buckets:
        raise BeliefError(
            "belief checkpoint count buckets do not match the encoding schema"
        )
    if (
        card_vocab_size < 1
        or max(row.card_def_id for row in schema.rows) >= card_vocab_size
    ):
        raise BeliefError(
            "belief checkpoint card vocabulary does not cover the encoding schema"
        )
    return BeliefCheckpointBinding.from_schema(schema).checkpoint_fields()


@dataclass(frozen=True, slots=True)
class BeliefTensorView:
    """Receipt-bound marginal view consumed by policy and value inference."""

    schema_identity: str
    card_def_ids: NDArray[np.int64]
    owner_role_ids: NDArray[np.int64]
    hidden_zone_ids: NDArray[np.int64]
    count_probabilities: NDArray[np.float32]
    validity: NDArray[np.float32]
    entropy: float
    effective_support: float
    encoding_receipt: str

    def to_torch_observation(
        self, *, batch_size: int, device: torch.device | str
    ) -> dict[str, torch.Tensor]:
        """Broadcast one immutable view across a model inference batch."""

        if batch_size < 1:
            raise BeliefError("belief batch size must be positive")

        def repeated(values: NDArray, dtype: torch.dtype) -> torch.Tensor:
            tensor = torch.as_tensor(
                np.asarray(values).copy(), dtype=dtype, device=device
            )
            return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)

        return {
            "belief_card_def_ids": repeated(self.card_def_ids, torch.long),
            "belief_owner_role_ids": repeated(self.owner_role_ids, torch.long),
            "belief_hidden_zone_ids": repeated(self.hidden_zone_ids, torch.long),
            "belief_count_probabilities": repeated(
                self.count_probabilities, torch.float32
            ),
            "belief_validity": repeated(self.validity, torch.float32),
            "belief_globals": repeated(
                np.asarray([self.entropy, self.effective_support]), torch.float32
            ),
        }


def belief_schema_from_engine(
    engine: Any,
    space: PossibleWorldSpace,
    *,
    count_buckets: int,
) -> BeliefEncodingSchema:
    """Build fixed opponent hand/library rows from the native content manifest."""

    try:
        manifest = engine.content_pack_manifest()
        content_identity = str(manifest["content_digest"])
        definitions = tuple(
            (int(row["card_def_id"]), str(row["registry_name"]))
            for row in manifest["definitions"]
        )
    except Exception as error:
        raise BeliefError(
            "native content manifest is unavailable or malformed"
        ) from error
    if content_identity != space.content_manifest_identity:
        raise BeliefError("native content manifest changed from the world space")
    if not definitions:
        raise BeliefError("native content definitions must be non-empty")
    definition_names = {name for _, name in definitions}
    if not {name for name, _ in space.pool}.issubset(definition_names):
        raise BeliefError("world-space pool is outside the native content manifest")

    rows = tuple(
        sorted(
            BeliefRow(
                owner_role_id=OPPONENT_OWNER_ROLE_ID,
                hidden_zone_id=zone_id,
                card_def_id=card_id,
                card_name=name,
            )
            for card_id, name in definitions
            for zone_id in (LIBRARY_ZONE_ID, HAND_ZONE_ID)
        )
    )
    return BeliefEncodingSchema(
        schema_identity="manabot.belief-tensor/opponent-hidden-counts-v1",
        world_schema_identity=space.world_schema_identity,
        content_manifest_identity=content_identity,
        rows=rows,
        count_buckets=count_buckets,
    )


def encode_belief(
    belief: BeliefState, schema: BeliefEncodingSchema
) -> BeliefTensorView:
    """Create the deterministic, receipt-bound policy projection."""

    if belief.space.world_schema_identity != schema.world_schema_identity:
        raise BeliefError("belief world schema does not match the encoding schema")
    if belief.space.content_manifest_identity != schema.content_manifest_identity:
        raise BeliefError("belief content manifest does not match the encoding schema")
    pool = dict(belief.space.pool)
    pool_names = set(pool)
    schema_names = {row.card_name for row in schema.rows}
    if not pool_names.issubset(schema_names):
        raise BeliefError(
            "belief content vocabulary does not match the encoding schema"
        )

    row_count = len(schema.rows)
    # Accumulate in float64. Real engine spaces can contain thousands of
    # worlds, and adding their weights directly into float32 buckets can move
    # an otherwise normalized marginal outside the fail-closed tolerance.
    probabilities64 = np.zeros((row_count, schema.count_buckets), dtype=np.float64)
    belief_probabilities = belief.probabilities
    for row_index, row in enumerate(schema.rows):
        if row.owner_role_id != OPPONENT_OWNER_ROLE_ID:
            raise BeliefError(
                "the canonical world space only projects opponent-owned hidden rows"
            )
        if row.hidden_zone_id not in {HAND_ZONE_ID, LIBRARY_ZONE_ID}:
            raise BeliefError(
                "the canonical world space only projects hidden hand and library rows"
            )
        for world in belief.space.worlds:
            hand_count = world.count(row.card_name)
            count = (
                hand_count
                if row.hidden_zone_id == HAND_ZONE_ID
                else pool.get(row.card_name, 0) - hand_count
            )
            if count < 0:
                raise BeliefError(
                    f"world hand count exceeds the unseen pool for {row.card_name!r}"
                )
            if count >= schema.count_buckets:
                raise BeliefError(
                    f"count {count} for {row.card_name!r} exceeds the encoding schema"
                )
            probabilities64[row_index, count] += belief_probabilities[world.index]
    if not np.allclose(
        probabilities64.sum(axis=1),
        1.0,
        atol=1e-10,
        rtol=0.0,
    ):
        raise BeliefError("belief marginal projection is not normalized")
    probabilities = probabilities64.astype(np.float32)

    card_def_ids = np.asarray([row.card_def_id for row in schema.rows], dtype=np.int64)
    owner_role_ids = np.asarray(
        [row.owner_role_id for row in schema.rows], dtype=np.int64
    )
    hidden_zone_ids = np.asarray(
        [row.hidden_zone_id for row in schema.rows], dtype=np.int64
    )
    validity = np.ones(row_count, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(schema.identity.encode("ascii"))
    for values in (
        card_def_ids.astype("<i8", copy=False),
        owner_role_ids.astype("<i8", copy=False),
        hidden_zone_ids.astype("<i8", copy=False),
        probabilities.astype("<f4", copy=False),
        validity.astype("<f4", copy=False),
    ):
        digest.update(values.tobytes())
    digest.update(
        np.asarray([belief.entropy, belief.effective_support], dtype="<f8").tobytes()
    )
    return BeliefTensorView(
        schema_identity=schema.identity,
        card_def_ids=_readonly(card_def_ids),
        owner_role_ids=_readonly(owner_role_ids),
        hidden_zone_ids=_readonly(hidden_zone_ids),
        count_probabilities=_readonly(probabilities),
        validity=_readonly(validity),
        entropy=belief.entropy,
        effective_support=belief.effective_support,
        encoding_receipt=digest.hexdigest(),
    )


__all__ = [
    "BeliefEncodingSchema",
    "BeliefRow",
    "BeliefTensorView",
    "HAND_ZONE_ID",
    "LIBRARY_ZONE_ID",
    "OPPONENT_OWNER_ROLE_ID",
    "belief_schema_from_engine",
    "encode_belief",
]
