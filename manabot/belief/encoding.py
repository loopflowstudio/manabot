"""Deterministic policy projection of canonical manabot beliefs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from numpy.typing import NDArray
import torch

from manabot.belief.state import BeliefError, BeliefState


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
        if not self.rows or len(set(self.rows)) != len(self.rows):
            raise BeliefError("belief encoding rows must be non-empty and unique")
        if tuple(sorted(self.rows)) != self.rows:
            raise BeliefError("belief encoding rows must be canonically ordered")
        if any(
            row.card_def_id < 0
            or row.owner_role_id < 0
            or row.hidden_zone_id < 0
            for row in self.rows
        ):
            raise BeliefError("belief semantic ids must be non-negative")
        if len({row.card_def_id for row in self.rows}) != len(self.rows):
            raise BeliefError("card definition ids must be unique")
        if len({row.card_name for row in self.rows}) != len(self.rows):
            raise BeliefError("card names must be unique")

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
            tensor = torch.as_tensor(np.asarray(values).copy(), dtype=dtype, device=device)
            return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)

        return {
            "belief_card_def_ids": repeated(self.card_def_ids, torch.long),
            "belief_owner_role_ids": repeated(self.owner_role_ids, torch.long),
            "belief_hidden_zone_ids": repeated(self.hidden_zone_ids, torch.long),
            "belief_count_probabilities": repeated(
                self.count_probabilities, torch.float32
            ),
            "belief_validity": repeated(self.validity, torch.float32),
            "belief_globals": torch.tensor(
                [self.entropy, self.effective_support],
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
            .expand(batch_size, 2),
        }


def encode_belief(
    belief: BeliefState, schema: BeliefEncodingSchema
) -> BeliefTensorView:
    """Create the deterministic, receipt-bound policy projection."""

    if belief.space.world_schema_identity != schema.world_schema_identity:
        raise BeliefError("belief world schema does not match the encoding schema")
    if belief.space.content_manifest_identity != schema.content_manifest_identity:
        raise BeliefError("belief content manifest does not match the encoding schema")
    pool_names = tuple(name for name, _ in belief.space.pool)
    schema_names = tuple(row.card_name for row in schema.rows)
    if len(set(schema_names)) != len(schema_names) or set(schema_names) != set(
        pool_names
    ):
        raise BeliefError("belief content vocabulary does not match the encoding schema")

    row_count = len(schema.rows)
    probabilities = np.zeros(
        (row_count, schema.count_buckets), dtype=np.float32
    )
    for row_index, row in enumerate(schema.rows):
        for world in belief.space.worlds:
            count = world.count(row.card_name)
            if count >= schema.count_buckets:
                raise BeliefError(
                    f"count {count} for {row.card_name!r} exceeds the encoding schema"
                )
            probabilities[row_index, count] += np.float32(
                belief.normalized_distribution[world.index]
            )
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise BeliefError("belief marginal projection is not normalized")

    card_def_ids = np.asarray(
        [row.card_def_id for row in schema.rows], dtype=np.int64
    )
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
    digest.update(np.asarray([belief.entropy, belief.effective_support], dtype="<f8").tobytes())
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
    "encode_belief",
]
