"""Portable ActorCritic checkpoints: exact weights, world and input admission.

save_checkpoint records the model architecture, observation configuration and
explicit native matchup in a PyTorch `.pt` file. load_checkpoint reads either
that file or the shared container (`manabot.policy.checkpoint`), choosing by
the file's bytes, verifies bindings before loading weights and returns an
eval-mode model. Optimizer/collector recovery belongs to training. Historical
serialized keys remain unchanged; these are not training snapshots.

`BACKEND` is the torch `manabot.policy.protocol.Backend`: NumPy in and out,
beside the native torch API that training and evaluation use. Its parameters
and gradients keep canonical names (torch's tensors have the canonical
shapes). Until `ActorCritic` reads `Prepared` indices itself, its `prepare`
builds the shared `Prepared` and retains the admitted torch observation on it
(`Retained`), which `evaluate` and the learner run; parity against the
retained-reference torch path then exercises the harness with torch alone.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Literal, overload
from urllib.parse import unquote, urlsplit

from etude.run.records import ArtifactRef
from etude.run.storage import storage_from_uri
from etude.train.records import CheckpointRecord
import numpy as np
import torch

from manabot.backends.torch.learner import gradient_step
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rl import selected_mean
from manabot.belief.encoding import (
    BeliefCheckpointBinding,
    BeliefEncodingSchema,
    belief_checkpoint_fields,
)
from manabot.env.observation import ObservationSpace
from manabot.infra.artifacts import StoredArtifact, materialize_artifact, verify_file
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.model.admission import AdmittedObservation, admit_observation
from manabot.model.architecture import (
    architecture_receipt,
    validate_architecture_receipt,
)
from manabot.model.policy_distribution import legal_mask
from manabot.model.world import (
    checkpoint_source,
    checkpoint_world,
    deck_world,
    validate_checkpoint_world,
    validate_policy_input,
)
from manabot.policy.checkpoint import (
    PolicyCheckpoint,
    canonical_name,
    canonical_parameters,
    read_container,
)
from manabot.policy.prepare import Prepared, prepare
from manabot.policy.protocol import Settings, Targets
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
    path: str | StoredArtifact, *, bucketed_shapes: bool = False
) -> tuple[ActorCritic, ObservationSpace]: ...


@overload
def load_checkpoint(
    path: str | StoredArtifact,
    *,
    include_belief_binding: Literal[True],
    bucketed_shapes: bool = False,
) -> tuple[ActorCritic, ObservationSpace, BeliefCheckpointBinding | None]: ...


def load_checkpoint(
    path: str | StoredArtifact,
    *,
    include_belief_binding: bool = False,
    bucketed_shapes: bool = False,
) -> (
    tuple[ActorCritic, ObservationSpace]
    | tuple[ActorCritic, ObservationSpace, BeliefCheckpointBinding | None]
):
    """Load saved configuration and validate world/schema; verify S3 bytes before loading.

    A PyTorch `.pt` file (a zip archive) loads as written; any other file is
    read as the shared container, which raises `NotAContainer` if it is not
    one and `UnsupportedPolicy` for a configuration the container does not
    carry. Raises `UnsupportedPolicyInput` for a checkpoint of another policy
    input. `bucketed_shapes` selects the forward's padding path (see
    `ActorCritic`); it is an execution choice, not a checkpoint field.
    """

    if isinstance(path, StoredArtifact):
        path = str(materialize_artifact(path))
    if not _is_torch_file(Path(path)):
        agent, obs_space = _load_container(Path(path), bucketed_shapes)
        if include_belief_binding:
            return agent, obs_space, None
        return agent, obs_space

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["hypers"]
    obs_space = ObservationSpace(
        ObservationConfig.from_checkpoint(config["observation_hypers"])
    )

    world_binding = validate_checkpoint_world(checkpoint, obs_space)
    agent_config = ModelConfig.from_checkpoint(config["agent_hypers"])
    agent = ActorCritic(obs_space, agent_config, bucketed_shapes=bucketed_shapes)
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


def _is_torch_file(path: Path) -> bool:
    """`torch.save` writes a zip archive; the container starts with its
    8-byte header length."""
    with path.open("rb") as file:
        return file.read(4) == b"PK\x03\x04"


def _load_container(
    path: Path, bucketed_shapes: bool
) -> tuple[ActorCritic, ObservationSpace]:
    """An eval-mode policy from a container. `read_container` has validated
    the world binding and refused unsupported configurations, so no belief
    binding or `bc` value semantic applies."""
    container = read_container(path)
    record = {
        key: json.loads(container.metadata[key])
        for key in ("world_binding", "source", "architecture")
    }
    agent = assemble(container, bucketed_shapes=bucketed_shapes)
    world_binding = validate_checkpoint_world(record, container.space)
    validate_policy_input(agent, world_binding)
    agent.world_binding = world_binding
    if record["architecture"] is not None:
        validate_architecture_receipt(agent, record["architecture"])
    agent.eval()
    return agent, container.space


def assemble(
    checkpoint: PolicyCheckpoint, *, bucketed_shapes: bool = False
) -> ActorCritic:
    """An `ActorCritic` holding a container's parameters and buffers; binds
    no world (`load_checkpoint` validates the binding)."""
    agent = ActorCritic(
        checkpoint.space, checkpoint.model, bucketed_shapes=bucketed_shapes
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
        "architecture": architecture_receipt(agent).model_dump(mode="json"),
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
    torch.save(checkpoint, path)


# -- the NumPy backend protocol ------------------------------------------------------


@dataclass(frozen=True)
class Retained(Prepared):
    """A `Prepared` batch that keeps the admitted torch observation the torch
    forward still reads. Removed when `ActorCritic` reads `Prepared` itself."""

    admitted: AdmittedObservation = field(kw_only=True)


class TorchPolicy:
    """`BackendPolicy` over an `ActorCritic`."""

    def __init__(self, agent: ActorCritic, checkpoint: PolicyCheckpoint) -> None:
        self.agent = agent
        self._tables = checkpoint.tables()

    def prepare(
        self, obs: Mapping[str, np.ndarray], *, inference: bool = False
    ) -> Retained:
        shared = prepare(obs, self._tables, inference=inference)
        admitted = admit_observation(
            self.agent, {key: torch.from_numpy(value) for key, value in obs.items()}
        )
        return Retained(
            **{item.name: getattr(shared, item.name) for item in fields(shared)},
            admitted=admitted,
        )

    def evaluate(self, batch: Prepared) -> tuple[np.ndarray, np.ndarray]:
        """The eval-mode forward, as collection runs it."""
        self.agent.eval()
        with torch.no_grad():
            logits, values = self.agent(_admitted(batch))
        return logits.numpy(), values.numpy()

    def cache_programs(self) -> None:
        """The torch forward encodes its programs per call; nothing to cache."""

    def clear_cache(self) -> None:
        pass


class TorchLearner:
    """`BackendLearner`: the damped PPO loss with a clipped Adam step, as
    `manabot.training.updates.damped_policy` computes it, over canonical names."""

    def __init__(self, agent: ActorCritic) -> None:
        self.agent = agent
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=0.0, eps=1e-5)

    def _loss(
        self, batch: Prepared, targets: Targets, settings: Settings
    ) -> torch.Tensor:
        # Imported here: the update module reaches collection and search,
        # which load checkpoints through this module.
        from manabot.training.updates.damped_policy import damped_policy_loss

        obs = _admitted(batch)
        self.agent.train()
        logits, values = self.agent.forward_distribution(obs)
        policy = damped_policy_loss(
            logits,
            torch.from_numpy(targets.actions),
            torch.from_numpy(targets.advantages),
            torch.from_numpy(targets.behavior_logs),
            torch.from_numpy(targets.reference),
            legal_mask(obs["actions_valid"]),
            clip=settings.clip,
            collection_kl=settings.collection_kl,
            tau=settings.tau,
        )
        value = (
            (values.squeeze(-1) - torch.from_numpy(targets.value_targets))
            .square()
            .mean()
        )
        return selected_mean(policy, torch.from_numpy(targets.actor_mask)) + value

    def gradients(
        self, batch: Prepared, targets: Targets, settings: Settings
    ) -> tuple[float, dict[str, np.ndarray]]:
        self.agent.zero_grad(set_to_none=True)
        loss = self._loss(batch, targets, settings)
        loss.backward()
        gradients = canonical_parameters(
            {
                name: (
                    np.zeros(tuple(parameter.shape), np.float32)
                    if parameter.grad is None
                    else parameter.grad.numpy().copy()
                )
                for name, parameter in self.agent.named_parameters()
            }
        )
        self.agent.zero_grad(set_to_none=True)
        return float(loss.detach()), gradients

    def step(
        self, batch: Prepared, targets: Targets, settings: Settings
    ) -> tuple[float, float]:
        for group in self.optimizer.param_groups:
            group["lr"] = settings.learning_rate
        loss = self._loss(batch, targets, settings)
        norm = gradient_step(loss, self.agent, self.optimizer, settings.max_grad_norm)
        return float(loss.detach()), float(norm)

    def parameters(self) -> dict[str, np.ndarray]:
        return canonical_parameters(
            {
                name: parameter.detach().numpy().copy()
                for name, parameter in self.agent.named_parameters()
            }
        )


def _admitted(batch: Prepared) -> AdmittedObservation:
    if not isinstance(batch, Retained):
        raise TypeError("the torch backend evaluates batches its own prepare built")
    return batch.admitted


class _Torch:
    name = "torch"

    def load(self, checkpoint: PolicyCheckpoint) -> tuple[TorchPolicy, TorchLearner]:
        agent = assemble(checkpoint)
        return TorchPolicy(agent, checkpoint), TorchLearner(agent)

    def parameter_layout(
        self, canonical: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        return {name: np.asarray(value) for name, value in canonical.items()}

    def gradient_layout(
        self, canonical: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        return {name: np.asarray(value) for name, value in canonical.items()}


BACKEND = _Torch()
