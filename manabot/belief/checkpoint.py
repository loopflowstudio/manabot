"""Admit frozen belief samplers by exact bytes, schema, behavior and world."""

import hashlib
from pathlib import Path
from typing import Any, Protocol

from manabot.belief.sampling_types import SamplerSchema
from manabot.sim.search.sampling import LearnedDistribution


class SamplerCodec(Protocol):
    def decode_sampler(self, data: bytes) -> Any: ...
    def restore_sampler(
        self,
        schema: SamplerSchema,
        hidden_size: int,
        history_dropout: float,
        state: Any,
    ) -> LearnedDistribution: ...


def load_belief_sampler(
    path: Path,
    *,
    codec: SamplerCodec,
    expected_dataset_identity: str,
    expected_policy_identity: str,
    expected_schema_identity: str,
    expected_world_identity: str,
    expected_checkpoint_identity: str,
) -> LearnedDistribution:
    """Reject mismatched bytes or semantic bindings before admitting parameters."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_checkpoint_identity:
        raise ValueError("sampler checkpoint byte identity mismatch")
    # Torch's weights-only decoder is an untyped serialization boundary.
    payload: Any = codec.decode_sampler(data)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "manabot.autoregressive-belief-sampler/v1"
    ):
        raise ValueError("unknown sampler checkpoint format")
    expected = {
        "dataset_identity": expected_dataset_identity,
        "policy_identity": expected_policy_identity,
        "schema_identity": expected_schema_identity,
        "world_identity": expected_world_identity,
    }
    for key, value in expected.items():
        if not value or payload.get(key) != value:
            raise ValueError(f"sampler checkpoint {key} mismatch")
    schema_payload = payload["schema"]
    if not isinstance(schema_payload, dict):
        raise ValueError("invalid sampler checkpoint schema")
    schema = SamplerSchema(**schema_payload)
    if schema.identity != expected_schema_identity:
        raise ValueError("sampler checkpoint schema contents mismatch")
    return codec.restore_sampler(
        schema,
        payload["hidden_size"],
        payload["history_dropout"],
        payload["state_dict"],
    )
