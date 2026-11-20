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

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import importlib
from pathlib import Path
import tempfile
from types import ModuleType
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
from manabot.model.interface import saved_readings, world_fields
from manabot.model.world import (
    checkpoint_source,
    validate_policy_input,
)
from manabot.policy.checkpoint import (
    NotAContainer,
    ParameterArrays,
    PolicyCheckpoint,
    admit_checkpoint,
    canonical_name,
    checkpoint_metadata,
    from_torch_state,
    read_metadata,
    read_saved_container,
    torch_name,
    write_container,
)
from manabot.policy.prepare import Tables
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
    agent, space = BACKEND.load_policy(
        path, bucketed_shapes=bucketed_shapes, live_tokens=live_tokens
    )
    if include_belief_binding:
        return agent, space, agent.belief_binding
    return agent, space


def read_policy(path: str) -> PolicyCheckpoint:
    """Read and bind either policy format without constructing a Torch module."""
    try:
        return admit_checkpoint(read_written(Path(path)))
    except NotAContainer as error:
        raise NotAContainer(
            f"{path} is neither a PyTorch checkpoint (a zip archive) nor a "
            "policy checkpoint container; it may be truncated or another kind of file"
        ) from error


def restore_policy(
    checkpoint: PolicyCheckpoint,
    *,
    bucketed_shapes: bool = False,
    live_tokens: bool = False,
) -> ActorCritic:
    """Assemble an admitted checkpoint with its saved decision and value semantics."""
    agent = assemble(
        checkpoint, bucketed_shapes=bucketed_shapes, live_tokens=live_tokens
    )
    record = checkpoint.record()
    value_semantic = record.get("bc", {}).get("value_semantic")
    if value_semantic is not None:
        if value_semantic not in ("signed_outcome", "win_logit"):
            raise ValueError("unsupported checkpoint value semantic")
        if agent.value_semantic is not None and value_semantic != agent.value_semantic:
            raise ValueError("checkpoint value semantic conflicts with its critic")
        agent.value_semantic = value_semantic
    world_binding = record["world_binding"]
    validate_policy_input(agent, world_binding)
    agent.world_binding = world_binding
    agent.belief_binding = (
        BeliefCheckpointBinding.from_checkpoint(record)
        if agent.belief_count_buckets > 0
        else None
    )
    agent.eval()
    return agent


def read_written(path: Path) -> PolicyCheckpoint:
    """The writer's tensors and metadata from either format, before admission."""
    if not _is_torch_file(path):
        return read_saved_container(path)
    record = torch.load(path, map_location="cpu", weights_only=True)
    config = record["hypers"]
    observation = ObservationConfig.from_checkpoint(config["observation_hypers"])
    model = ModelConfig.from_checkpoint(config["agent_hypers"])
    state = record["model_state_dict"]
    buffers = set(Tables.BUFFERS.values())
    return from_torch_state(
        model,
        observation,
        {name: tensor.detach().numpy() for name, tensor in state.items()},
        checkpoint_metadata(record),
        parameter_names=[name for name in state if name not in buffers],
        readings=saved_readings(record, ObservationSpace(observation)),
    )


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


def _is_torch_file(path: Path) -> bool:
    """`torch.save` writes a zip archive; the container starts with its
    8-byte header length."""
    with path.open("rb") as file:
        return file.read(4) == b"PK\x03\x04"


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
        readings=checkpoint.readings,
        bucketed_shapes=bucketed_shapes,
        live_tokens=live_tokens,
    )
    cards = agent.semantic_cards
    if cards is not None and checkpoint.report is not None:
        weight = checkpoint.parameters["semantic_cards.token_embedding.weight"]
        if cards.token_embedding.num_embeddings != weight.shape[0]:
            cards.token_embedding = torch.nn.Embedding(
                weight.shape[0], checkpoint.model.hidden_dim
            )
        program_shape = checkpoint.buffers["semantic_cards.tokens"].shape
        if cards.tokens.shape != program_shape:
            cards.tokens = torch.empty(program_shape, dtype=torch.long)
    torch_names = {canonical_name(name): name for name in agent.state_dict()}
    arrays = {**checkpoint.parameters, **checkpoint.buffers}
    unknown = sorted(set(arrays) - set(torch_names))
    if unknown:
        raise ValueError(f"container tensors with no policy parameter: {unknown}")
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
    player_configs: Sequence[managym.PlayerConfig] | None,
    extra: dict[str, Any] | None = None,
    belief_schema: BeliefEncodingSchema | None = None,
) -> None:
    """Save a portable policy with its explicit world, input and value semantics.

    Pass `player_configs` to record the exact setups in the binding. Passing
    `None` allows the policy to play any admitted deck.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        **world_fields(obs_space, agent.readings, player_configs),
        "source": checkpoint_source(),
        "architecture": agent.architecture_receipt().model_dump(mode="json"),
        "global_step": 0,
        "hypers": {
            "agent_hypers": agent.config.checkpoint_payload(),
            "observation_hypers": obs_space.config.model_dump(),
            "train_hypers": {},
            # The collector applies the policy's decision settings to both seats.
            # Training provenance is independent of strict world/ABI admission.
            "training_setup": {
                "mana_payment": [obs_space.config.mana_payment] * 2,
                "mana_floats": [obs_space.config.mana_floats] * 2,
                "skip_trivial": True,
            },
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
    # Check before NumPy conversion, which refuses some floating types outright.
    for name, parameter in agent.named_parameters():
        if parameter.dtype != torch.float32:
            raise TypeError(
                f"parameter {name}: expected float32, got {parameter.dtype}"
            )
    container = from_torch_state(
        agent.config,
        obs_space.config,
        {
            name: tensor.detach().cpu().numpy()
            for name, tensor in agent.state_dict().items()
        },
        checkpoint_metadata(checkpoint),
        parameter_names=dict(agent.named_parameters()),
        readings=agent.readings,
    )
    write_container(path, container)


class _Torch:
    name = "torch"

    @property
    def verification(self) -> ModuleType:
        return importlib.import_module("manabot.backends.torch.verification")

    @property
    def runtime(self) -> ModuleType:
        return importlib.import_module("manabot.backends.torch.runtime")

    save_policy = staticmethod(save_checkpoint)

    @property
    def state(self) -> ModuleType:
        return importlib.import_module("manabot.backends.torch.state")

    @property
    def belief_fit(self) -> ModuleType:
        return importlib.import_module("manabot.backends.torch.belief_fit")

    @property
    def belief(self) -> ModuleType:
        return importlib.import_module("manabot.backends.torch.belief")

    @property
    def collection(self) -> ModuleType:
        return importlib.import_module("manabot.backends.torch.collection")

    @staticmethod
    @contextmanager
    def seeded_inference(seed: int, *, cpu_only: bool = False) -> Iterator[None]:
        with (
            torch.random.fork_rng(devices=[] if cpu_only else None),
            torch.inference_mode(),
        ):
            torch.manual_seed(seed)
            yield

    @property
    def rl(self) -> ModuleType:
        return importlib.import_module("manabot.backends.torch.rl")

    @property
    def arrays(self) -> ModuleType:
        return importlib.import_module("manabot.backends.torch.arrays")

    @property
    def learner(self) -> ModuleType:
        return importlib.import_module("manabot.backends.torch.learner")

    @property
    def rollout(self) -> ModuleType:
        return importlib.import_module("manabot.backends.torch.rollout")

    @staticmethod
    def load_policy(
        path: str | Path | PolicyCheckpoint,
        *,
        bucketed_shapes: bool = False,
        live_tokens: bool = False,
    ) -> tuple[ActorCritic, ObservationSpace]:
        checkpoint = (
            path if isinstance(path, PolicyCheckpoint) else read_policy(str(path))
        )
        agent = restore_policy(
            checkpoint, bucketed_shapes=bucketed_shapes, live_tokens=live_tokens
        )
        return agent, checkpoint.space

    read_policy = staticmethod(read_policy)

    @staticmethod
    def metadata(path: str) -> dict[str, Any]:
        return load_metadata(path)

    @staticmethod
    def seed_training(seed: int, *, torch_deterministic: bool = True) -> None:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

    @staticmethod
    def seed(seed: int) -> None:
        torch.manual_seed(seed)

    def load(self, checkpoint: PolicyCheckpoint) -> tuple[ActorCritic, Learner]:
        agent = assemble(checkpoint)
        return agent, Learner(agent, "cpu", minibatch_rows=1)

    def parameter_layout(self, canonical: ParameterArrays) -> dict[str, np.ndarray]:
        """Torch tensors have the canonical shapes under torch's own names."""
        return {
            torch_name(name): np.asarray(value) for name, value in canonical.items()
        }

    gradient_layout = parameter_layout


BACKEND = _Torch()
