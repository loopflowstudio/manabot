"""Admit frozen belief samplers by exact bytes, schema, behavior and world."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, TypedDict, Unpack

import numpy as np

from manabot.belief.protocol import SamplerCodec
from manabot.belief.sampling_types import SamplerSchema
from manabot.policy.checkpoint import decode_safetensors, encode_safetensors
from manabot.sim.search.sampling import LearnedDistribution

if TYPE_CHECKING:
    from manabot.belief.sampling_data import SamplerDataset
    from manabot.belief.sampling_fit import SamplerFitResult

FORMAT = "manabot.autoregressive-belief-sampler/v1"


def validate_parameters(parameters: Mapping[str, np.ndarray]) -> None:
    names = {
        "context.weight",
        "context.bias",
        "card.weight",
        "recurrent.input_weight",
        "recurrent.hidden_weight",
        "recurrent.input_bias",
        "recurrent.hidden_bias",
        "correction.weight",
        "correction.bias",
    }
    if difference := names.symmetric_difference(parameters):
        raise ValueError(f"sampler checkpoint parameters differ: {sorted(difference)}")
    if any(value.dtype != np.float32 for value in parameters.values()):
        raise ValueError("sampler parameters must be float32")


class SamplerIdentities(TypedDict):
    expected_dataset_identity: str
    expected_policy_identity: str
    expected_schema_identity: str
    expected_world_identity: str
    expected_checkpoint_identity: str


@dataclass(frozen=True)
class AdmittedSampler:
    model: LearnedDistribution
    training_seed: int | None


def read_belief_sampler(
    path: Path,
    *,
    codec: SamplerCodec,
    expected_dataset_identity: str,
    expected_policy_identity: str,
    expected_schema_identity: str,
    expected_world_identity: str,
    expected_checkpoint_identity: str,
) -> AdmittedSampler:
    """Reject mismatched bytes or semantic bindings before admitting parameters."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_checkpoint_identity:
        raise ValueError("sampler checkpoint byte identity mismatch")
    try:
        parameters, metadata = decode_safetensors(data)
        payload = {key: json.loads(value) for key, value in metadata.items()}
    except ValueError as error:
        raise ValueError("unknown sampler checkpoint format") from error
    if payload.get("format") != FORMAT:
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
    schema = SamplerSchema(
        **(schema_payload | {"card_names": tuple(schema_payload["card_names"])})
    )
    if schema.identity != expected_schema_identity:
        raise ValueError("sampler checkpoint schema contents mismatch")
    validate_parameters(parameters)
    metrics = payload.get("metrics", {})
    seed = metrics.get("seed") if isinstance(metrics, dict) else None
    if isinstance(metrics, dict) and "seed" in metrics and type(seed) is not int:
        raise ValueError("invalid checkpoint training seed")
    model = codec.restore_sampler(
        schema,
        payload["hidden_size"],
        payload["history_dropout"],
        parameters,
    )
    return AdmittedSampler(model, seed)


def load_belief_sampler(
    path: Path, *, codec: SamplerCodec, **identities: Unpack[SamplerIdentities]
) -> LearnedDistribution:
    return read_belief_sampler(path, codec=codec, **identities).model


def save_belief_sampler(
    path: Path,
    result: SamplerFitResult,
    dataset: SamplerDataset,
    *,
    codec: SamplerCodec,
) -> str:
    """Create one immutable checkpoint; return SHA-256 of its exact bytes."""
    if (result.dataset_identity, result.policy_identity, result.world_identity) != (
        dataset.identity,
        dataset.policy_identity,
        dataset.world_identity,
    ):
        raise ValueError(
            "sampler fit is bound to a different dataset or policy/world identity"
        )
    if result.model.schema != dataset.schema:
        raise ValueError("sampler and dataset schemas differ")
    metadata = {
        "format": FORMAT,
        "schema": asdict(dataset.schema),
        "schema_identity": dataset.schema.identity,
        "dataset_identity": dataset.identity,
        "policy_identity": dataset.policy_identity,
        "world_identity": dataset.world_identity,
        "hidden_size": result.model.hidden_size,
        "history_dropout": result.model.history_dropout,
        "metrics": asdict(result.metrics),
    }
    parameters = codec.sampler_parameters(result.model)
    validate_parameters(parameters)
    data = encode_safetensors(
        parameters,
        {key: json.dumps(value) for key, value in metadata.items()},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".sampler-", delete=False
    ) as output:
        temporary = Path(output.name)
        try:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
            os.link(temporary, path)  # Atomic publication, refusing an existing target.
        finally:
            temporary.unlink()
    return hashlib.sha256(data).hexdigest()
