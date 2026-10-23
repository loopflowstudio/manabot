"""Admit frozen belief samplers by exact bytes, schema, behavior and world."""

import hashlib
import io
from pathlib import Path
from typing import Any

import torch

from manabot.belief.sampling import AutoregressiveBeliefSampler, SamplerSchema


def load_belief_sampler(
    path: Path,
    *,
    expected_dataset_identity: str,
    expected_policy_identity: str,
    expected_schema_identity: str,
    expected_world_identity: str,
    expected_checkpoint_identity: str,
) -> AutoregressiveBeliefSampler:
    """Reject mismatched bytes or semantic bindings before admitting parameters."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_checkpoint_identity:
        raise ValueError("sampler checkpoint byte identity mismatch")
    # Torch's weights-only decoder is an untyped serialization boundary.
    payload: Any = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
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
    model = AutoregressiveBeliefSampler(
        schema, payload["hidden_size"], payload["history_dropout"]
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model
