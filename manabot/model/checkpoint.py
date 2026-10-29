"""Portable ActorCritic checkpoints: exact weights, world and input admission.

save_checkpoint records the model architecture, observation configuration and
explicit native matchup. load_checkpoint verifies bindings before loading weights
and returns an eval-mode model. Optimizer/collector recovery belongs to training.
Historical serialized keys remain unchanged; these are not training snapshots.
"""

import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Literal, overload
from urllib.parse import unquote, urlsplit

from etude.run.records import ArtifactRef
from etude.run.storage import storage_from_uri
from etude.train.records import CheckpointRecord
import torch

from manabot.belief.encoding import (
    BeliefCheckpointBinding,
    BeliefEncodingSchema,
    belief_checkpoint_fields,
)
from manabot.env.observation import ObservationSpace
from manabot.infra.artifacts import StoredArtifact, materialize_artifact, verify_file
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.actor_critic import ActorCritic
from manabot.model.architecture import (
    architecture_receipt,
    validate_architecture_receipt,
)
from manabot.model.world import (
    checkpoint_world,
    pool_world,
    validate_checkpoint_world,
    validate_policy_input,
)
import managym

if TYPE_CHECKING:
    # Local policy loading must not import managed evaluation machinery.
    from etude.eval.records import Player


def load_etude_checkpoint(record_path: Path) -> tuple[ActorCritic, ObservationSpace]:
    """Admit the exact policy named by an Etude checkpoint record.

    Require digest and size before loading weights. Local SDK snapshots and S3/R2
    destinations share the same admission; neither names nor tags select a model.
    """
    record = CheckpointRecord.model_validate_json(record_path.read_bytes())
    with tempfile.TemporaryDirectory(prefix="manabot-policy-") as directory:
        return load_checkpoint(
            str(fetch_etude_checkpoint(record.artifact, Path(directory) / "policy.pt"))
        )


def local_checkpoint_file(seated: "Player") -> Path:
    """The verified file a model plays; raises for code or a remote checkpoint."""
    if seated.kind != "model":
        raise ValueError("expected a model with a digest and byte count")
    artifact = seated.checkpoint.artifact
    if artifact.sha256 is None or artifact.size_bytes is None:
        raise ValueError("expected a model with a digest and byte count")
    uri = urlsplit(artifact.uri)
    if uri.scheme != "file" or uri.netloc:
        raise ValueError("checkpoint must be a verified local file")
    path = Path(unquote(uri.path))
    verify_file(path, artifact.sha256, artifact.size_bytes)
    return path


def fetch_etude_checkpoint(reference: ArtifactRef, target: Path) -> Path:
    """The verified local bytes of an Etude checkpoint artifact; loads no weights.

    A local file is verified in place and returned. An S3 or R2 object is
    downloaded to `target` through the SDK's storage adapter and verified before
    it is renamed there, so a partial download never becomes a cached policy.
    Each call writes its own partial file, so concurrent fetches of one target
    are safe.
    """
    if reference.sha256 is None or reference.size_bytes is None:
        raise ValueError("Etude checkpoint requires a digest and byte count")
    uri = urlsplit(reference.uri)
    if uri.scheme == "file" and uri.netloc in {"", "localhost"}:
        path = Path(unquote(uri.path))
        verify_file(path, reference.sha256, reference.size_bytes)
        return path
    if uri.scheme not in {"s3", "r2"} or uri.query or uri.fragment or not uri.netloc:
        raise ValueError("Etude checkpoint must name a local file or S3/R2 object")
    storage = storage_from_uri(f"{uri.scheme}://{uri.netloc}")
    key = uri.path.lstrip("/")
    descriptor, name = tempfile.mkstemp(
        prefix=f"{target.name}.", suffix=".partial", dir=target.parent
    )
    partial = Path(name)
    with os.fdopen(descriptor, "wb") as file:
        offset = 0
        while offset < reference.size_bytes:
            length = min(8 * 1024 * 1024, reference.size_bytes - offset)
            block = storage.get_range(key, offset, length)
            if len(block) != length:
                raise ValueError("incomplete Etude checkpoint download")
            file.write(block)
            offset += length
    verify_file(partial, reference.sha256, reference.size_bytes)
    partial.replace(target)
    return target


@overload
def load_checkpoint(
    path: str | StoredArtifact,
) -> tuple[ActorCritic, ObservationSpace]: ...


@overload
def load_checkpoint(
    path: str | StoredArtifact, *, include_belief_binding: Literal[True]
) -> tuple[ActorCritic, ObservationSpace, BeliefCheckpointBinding | None]: ...


def load_checkpoint(
    path: str | StoredArtifact, *, include_belief_binding: bool = False
) -> (
    tuple[ActorCritic, ObservationSpace]
    | tuple[ActorCritic, ObservationSpace, BeliefCheckpointBinding | None]
):
    """Load saved configuration and validate world/schema; verify S3 bytes before loading."""

    if isinstance(path, StoredArtifact):
        path = str(materialize_artifact(path))

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["hypers"]
    obs_space = ObservationSpace(
        ObservationConfig.from_checkpoint(config["observation_hypers"])
    )

    world_binding = validate_checkpoint_world(checkpoint, obs_space)
    agent = ActorCritic(obs_space, ModelConfig(**config["agent_hypers"]))
    value_semantic = checkpoint.get("bc", {}).get("value_semantic")
    if value_semantic is not None:
        if value_semantic not in ("signed_outcome", "win_logit"):
            raise ValueError("unsupported checkpoint value semantic")
        if agent.value_semantic is not None and value_semantic != agent.value_semantic:
            raise ValueError("checkpoint value semantic conflicts with its critic")
        agent.value_semantic = value_semantic
    validate_policy_input(agent, world_binding)
    agent.world_binding = world_binding
    binding = None
    if agent.belief_count_buckets > 0:
        binding = BeliefCheckpointBinding.from_checkpoint(checkpoint)
    if "architecture" in checkpoint:
        validate_architecture_receipt(agent, checkpoint["architecture"])
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    if include_belief_binding:
        return agent, obs_space, binding
    return agent, obs_space


def save_checkpoint(
    agent: ActorCritic,
    obs_space: ObservationSpace,
    path: str | Path,
    *,
    player_configs: list[managym.PlayerConfig] | None = None,
    content_pool: str | None = None,
    extra: dict[str, Any] | None = None,
    belief_schema: BeliefEncodingSchema | None = None,
) -> None:
    """Save a portable policy with its explicit world, input and value semantics.

    Bind either exact setups or a whole content pool, never both.
    """

    if (player_configs is None) == (content_pool is None):
        raise ValueError("bind exactly one of player_configs or content_pool")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "world_binding": (
            pool_world(content_pool, obs_space)
            if content_pool is not None
            else checkpoint_world(player_configs, obs_space)
        ),
        "architecture": architecture_receipt(agent).model_dump(mode="json"),
        "model_state_dict": agent.state_dict(),
        "global_step": 0,
        "hypers": {
            "agent_hypers": agent.config.model_dump(),
            "observation_hypers": obs_space.encoder.config.model_dump(),
            "train_hypers": {},
        },
        "bc": extra or {},
    }
    validate_policy_input(agent, checkpoint["world_binding"])
    checkpoint.update(
        belief_checkpoint_fields(
            belief_schema,
            count_buckets=agent.belief_count_buckets,
            card_vocab_size=agent.config.belief_card_vocab_size,
        )
    )
    torch.save(checkpoint, path)
