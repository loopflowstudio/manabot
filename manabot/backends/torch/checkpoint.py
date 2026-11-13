"""Portable ActorCritic checkpoints: exact weights, world and input admission.

save_checkpoint records the model architecture, observation configuration and
explicit native matchup in the shared safetensors container. load_checkpoint
reads that container or a legacy PyTorch `.pt` file, choosing by
the file's bytes, verifies bindings before loading weights and returns an
eval-mode model. Optimizer/collector recovery belongs to training. Historical
serialized keys remain unchanged; these are not training snapshots.

`BACKEND` is the torch `manabot.policy.protocol.Backend`, NumPy in and out
beside the native torch API (see the protocol section below).
"""

from collections.abc import Mapping
import json
from pathlib import Path
import tempfile
from typing import Any, Literal, overload

from etude.train.records import CheckpointRecord
import numpy as np
import torch

from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.belief.encoding import (
    BeliefCheckpointBinding,
    BeliefEncodingSchema,
    belief_checkpoint_fields,
)
from manabot.env.observation import ObservationSpace
from manabot.infra.artifacts import (
    StoredArtifact,
    fetch_etude_checkpoint,
    materialize_artifact,
)
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.architecture import validate_architecture_receipt
from manabot.model.world import (
    checkpoint_source,
    checkpoint_world,
    deck_world,
    validate_checkpoint_world,
    validate_policy_input,
)
from manabot.policy.checkpoint import (
    NotAContainer,
    PolicyCheckpoint,
    canonical_name,
    checkpoint_metadata,
    from_torch_state,
    read_container,
    read_metadata,
    torch_name,
    write_container,
)
import managym


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


@overload
def load_checkpoint(
    path: str | StoredArtifact,
    *,
    bucketed_shapes: bool = False,
    live_tokens: bool = False,
) -> tuple[ActorCritic, ObservationSpace]: ...


@overload
def load_checkpoint(
    path: str | StoredArtifact,
    *,
    include_belief_binding: Literal[True],
    bucketed_shapes: bool = False,
    live_tokens: bool = False,
) -> tuple[ActorCritic, ObservationSpace, BeliefCheckpointBinding | None]: ...


def load_checkpoint(
    path: str | StoredArtifact,
    *,
    include_belief_binding: bool = False,
    bucketed_shapes: bool = False,
    live_tokens: bool = False,
) -> (
    tuple[ActorCritic, ObservationSpace]
    | tuple[ActorCritic, ObservationSpace, BeliefCheckpointBinding | None]
):
    """Load saved configuration and validate world/schema; verify S3 bytes before loading.

    A PyTorch `.pt` file (a zip archive) loads as written; any other file is
    read as the shared container. Raises `NotAContainer` for a file that is
    neither (a truncated download, say), `UnsupportedPolicy` for a
    configuration the container does not carry, and `UnsupportedPolicyInput`
    for a checkpoint of another policy input. `bucketed_shapes` and
    `live_tokens` select the forward's padding and attention layout (see
    `ActorCritic`); they are execution choices, not checkpoint fields.
    """

    if isinstance(path, StoredArtifact):
        path = str(materialize_artifact(path))
    if not _is_torch_file(Path(path)):
        try:
            agent, obs_space, binding = _load_container(
                Path(path), bucketed_shapes=bucketed_shapes, live_tokens=live_tokens
            )
        except NotAContainer as error:
            raise NotAContainer(
                f"{path} is neither a PyTorch checkpoint (a zip archive) nor a "
                "policy checkpoint container; it may be truncated or another "
                "kind of file"
            ) from error
        if include_belief_binding:
            return agent, obs_space, binding
        return agent, obs_space

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["hypers"]
    obs_space = ObservationSpace(
        ObservationConfig.from_checkpoint(config["observation_hypers"])
    )

    world_binding = validate_checkpoint_world(checkpoint, obs_space)
    agent_config = ModelConfig.from_checkpoint(config["agent_hypers"])
    agent = ActorCritic(
        obs_space,
        agent_config,
        bucketed_shapes=bucketed_shapes,
        live_tokens=live_tokens,
    )
    binding = _restore_metadata(agent, checkpoint, world_binding)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    if include_belief_binding:
        return agent, obs_space, binding
    return agent, obs_space


def load_metadata(path: str | Path) -> dict[str, Any]:
    """Metadata from either policy format; containers read only their header."""
    path = Path(path)
    if not _is_torch_file(path):
        return read_metadata(path)
    record = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(record, dict):
        raise ValueError("policy checkpoint requires a mapping")
    state = record.pop("model_state_dict", None)
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint holds no policy")
    record["parameter_count"] = sum(
        value.numel() for value in state.values() if isinstance(value, torch.Tensor)
    )
    # Old files have no explicit parameter order. Keep the stored state order
    # for the learning-state importer to filter against actual parameter names.
    record["parameter_names"] = list(state)
    return record


def _restore_metadata(
    agent: ActorCritic,
    checkpoint: Mapping[str, Any],
    world_binding: dict[str, Any],
) -> BeliefCheckpointBinding | None:
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
    if checkpoint.get("architecture") is not None:
        validate_architecture_receipt(
            agent.architecture_receipt(), checkpoint["architecture"]
        )
    return binding


def _is_torch_file(path: Path) -> bool:
    """`torch.save` writes a zip archive; the container starts with its
    8-byte header length."""
    with path.open("rb") as file:
        return file.read(4) == b"PK\x03\x04"


def _load_container(
    path: Path, *, bucketed_shapes: bool, live_tokens: bool
) -> tuple[ActorCritic, ObservationSpace, BeliefCheckpointBinding | None]:
    """Restore policy metadata with the same admission as a legacy checkpoint."""
    container = read_container(path)
    record = {
        key: json.loads(value)
        for key, value in container.metadata.items()
        if key != "converted_from"
    }
    agent = assemble(
        container, bucketed_shapes=bucketed_shapes, live_tokens=live_tokens
    )
    world_binding = validate_checkpoint_world(record, container.space)
    binding = _restore_metadata(agent, record, world_binding)
    agent.eval()
    return agent, container.space, binding


def assemble(
    checkpoint: PolicyCheckpoint,
    *,
    bucketed_shapes: bool = False,
    live_tokens: bool = False,
) -> ActorCritic:
    """An `ActorCritic` holding a container's parameters and buffers; binds
    no world (`load_checkpoint` validates the binding)."""
    agent = ActorCritic(
        checkpoint.space,
        checkpoint.model,
        bucketed_shapes=bucketed_shapes,
        live_tokens=live_tokens,
    )
    torch_names = {canonical_name(name): name for name in agent.state_dict()}
    arrays = {**checkpoint.parameters, **checkpoint.buffers}
    unknown = sorted(set(arrays) - set(torch_names))
    if unknown:
        raise KeyError(f"container tensors with no policy parameter: {unknown}")
    agent.load_state_dict(
        {
            torch_names[name]: torch.from_numpy(np.array(value))
            for name, value in arrays.items()
        }
    )
    return agent


def save_checkpoint(
    agent: ActorCritic,
    obs_space: ObservationSpace,
    path: str | Path,
    *,
    player_configs: list[managym.PlayerConfig] | None,
    extra: dict[str, Any] | None = None,
    belief_schema: BeliefEncodingSchema | None = None,
) -> None:
    """Save a portable policy with its explicit world, input and value semantics.

    `player_configs` binds those setups; `None` binds the world itself, so the
    policy may play any deck the world admits.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "world_binding": (
            deck_world(obs_space)
            if player_configs is None
            else checkpoint_world(player_configs, obs_space)
        ),
        "source": checkpoint_source(),
        "architecture": agent.architecture_receipt().model_dump(mode="json"),
        "model_state_dict": agent.state_dict(),
        "global_step": 0,
        "hypers": {
            "agent_hypers": agent.config.checkpoint_payload(),
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
    for name, parameter in agent.named_parameters():
        if parameter.dtype != torch.float32:
            raise TypeError(
                f"parameter {name}: expected float32, got {parameter.dtype}"
            )
    container = from_torch_state(
        agent.config,
        obs_space.encoder.config,
        {
            name: tensor.detach().cpu().numpy()
            for name, tensor in agent.state_dict().items()
        },
        checkpoint_metadata(checkpoint),
        parameter_names=dict(agent.named_parameters()),
    )
    write_container(path, container)


class _Torch:
    name = "torch"

    def load(self, checkpoint: PolicyCheckpoint) -> tuple[ActorCritic, Learner]:
        agent = assemble(checkpoint)
        return agent, Learner(agent, "cpu", minibatch_rows=1)

    def parameter_layout(
        self, canonical: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Torch tensors have the canonical shapes under torch's own names."""
        return {
            torch_name(name): np.asarray(value) for name, value in canonical.items()
        }

    gradient_layout = parameter_layout


BACKEND = _Torch()
