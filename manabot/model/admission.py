"""Admit tensor callers through the shared host policy contract.

An admitted batch carries raw buffers for collection diagnostics and Prepared
rows for the Torch forward. The configuration binding belongs to the batch,
not to an individual model instance; rollout owners can bind unchanged rows
without repeating validation. Preparation owns trimming and all input joins.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch

from manabot.backends.torch._numerics import buffer_on
from manabot.backends.torch.arrays import device_of, on_device, to_numpy
from manabot.backends.torch.rl import legal_mask
from manabot.policy import admission
from manabot.policy.buckets import BucketSizes, row_needs, sizes_of
from manabot.policy.checkpoint import layout
from manabot.policy.prepare import Prepared, Tables

if TYPE_CHECKING:
    from manabot.backends.torch.policy import ActorCritic


def admission_binding(agent: "ActorCritic") -> str:
    return (
        agent.config.model_dump_json()
        + "\n"
        + agent.observation_space.encoder.config.model_dump_json()
    )


def _spec(agent: "ActorCritic") -> admission.Spec:
    buffers: dict[str, np.ndarray] = {}
    for name, _ in agent.named_buffers():
        parent, _, field = name.rpartition(".")
        module = agent.get_submodule(parent) if parent else agent
        buffers[name] = to_numpy(buffer_on(module, field, torch.device("cpu")))
    tables = Tables.from_buffers(buffers, layout(agent.observation_space))
    return admission.Spec(
        agent.config,
        agent.observation_space,
        tables,
        agent.card_definitions.num_embeddings
        if agent.card_definitions is not None
        else 0,
    )


class AdmittedObservation(dict[str, torch.Tensor]):
    """Checked source buffers and prepared forward rows, read-only by contract."""

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        binding: str,
        legal: bool,
        buckets: BucketSizes | None = None,
        needs: np.ndarray | None = None,
        *,
        prepared: Prepared[torch.Tensor],
    ) -> None:
        super().__init__(tensors)
        self.binding = binding
        self.legal = legal
        self.buckets = buckets
        self.needs = needs
        self.prepared = prepared


def admit_observation(
    agent: "ActorCritic", obs: Mapping[str, torch.Tensor]
) -> AdmittedObservation:
    """Validate raw rows once, then prepare their host indexes and masks."""
    spec = _spec(agent)
    host = {key: to_numpy(value) for key, value in obs.items()}
    legal = to_numpy(legal_mask(torch.from_numpy(host["actions_valid"])))
    admission.check(host, spec, legal)
    return _bind(agent, obs, spec, measure_row_needs(agent, obs))


def check_observation(agent: "ActorCritic", obs: Mapping[str, torch.Tensor]) -> None:
    """Check rollout rows that their owner will prepare in later minibatches."""
    host = {key: to_numpy(value) for key, value in obs.items()}
    legal = to_numpy(legal_mask(torch.from_numpy(host["actions_valid"])))
    admission.check(host, _spec(agent), legal)


def bind_admitted(
    agent: "ActorCritic",
    obs: Mapping[str, torch.Tensor],
    *,
    needs: np.ndarray | None = None,
) -> AdmittedObservation:
    """Prepare unchanged rows of an already admitted source, without checks."""
    return _bind(
        agent, obs, _spec(agent), measure_row_needs(agent, obs, recorded=needs)
    )


def _bind(
    agent: "ActorCritic",
    obs: Mapping[str, torch.Tensor],
    spec: admission.Spec,
    needs: np.ndarray | None,
) -> AdmittedObservation:
    host = {key: to_numpy(value) for key, value in obs.items()}
    legal = bool(host["actions_valid"].any(-1).all())
    buckets = None if needs is None else _sizes(agent, needs)
    batch = agent.prepare_admitted(
        host,
        needs=needs,
        inference=device_of(agent).type == "cpu" and not torch.is_grad_enabled(),
    )
    return AdmittedObservation(
        obs,
        spec.binding,
        legal,
        buckets,
        needs,
        prepared=on_device(batch, obs["actions_valid"].device),
    )


def _sizes(agent: "ActorCritic", needs: np.ndarray) -> BucketSizes:
    return sizes_of(
        needs, agent.observation_space.encoder, objects=agent.object_trimming()
    )


def measure_row_needs(
    agent: "ActorCritic",
    obs: Mapping[str, torch.Tensor],
    *,
    recorded: np.ndarray | None = None,
) -> np.ndarray | None:
    """Per-row bucket needs; reuse those recorded by the collector when supplied."""
    if not agent.bucketed_shapes:
        return None
    if recorded is not None:
        return recorded
    return row_needs(
        {key: to_numpy(value) for key, value in obs.items()},
        agent.observation_space.encoder,
    )


def measure_buckets(
    agent: "ActorCritic", obs: Mapping[str, torch.Tensor]
) -> BucketSizes | None:
    needs = measure_row_needs(agent, obs)
    return None if needs is None else _sizes(agent, needs)
