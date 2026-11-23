"""Torch thread, precision and device setup, and policy construction.

The composition root has already chosen the device; a missing one is refused,
never replaced by CPU.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from manabot.backends.torch.learner import Learner, apply_matmul_precision
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy.checkpoint import pretrained_memory_branch
from manabot.training.learning import Precision
from manabot.training.mps import mps_torch_error


def require_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; CPU fallback is forbidden")
    if device == "mps" and (error := mps_torch_error()) is not None:
        raise ValueError(error)
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable; CPU fallback is forbidden")


def configure(threads: int, precision: Precision, device: str) -> None:
    torch.set_num_threads(threads)
    apply_matmul_precision(precision)
    require_device(device)


def threads() -> int:
    return torch.get_num_threads()


def create_policy(
    space: ObservationSpace,
    config: ModelConfig,
    *,
    bucketed_shapes: bool = False,
    live_tokens: bool = False,
    pretrained: ActorCritic | None = None,
    precision: Precision | None = None,
) -> ActorCritic:
    """Construct float32 master weights. Collection and learning scope autocast."""
    model = ActorCritic(
        space,
        config,
        readings=pretrained.readings if pretrained is not None else None,
        bucketed_shapes=bucketed_shapes,
        live_tokens=live_tokens,
    )
    if pretrained is not None:
        weights = pretrained.state_dict()
        if any(not torch.isfinite(v).all() for v in weights.values()):
            raise ValueError("pretrained policy contains nonfinite tensors")
        load_pretrained(model, weights, pretrained.config)
    return model


create_learner = Learner


@contextmanager
def scoped(threads: int, precision: Precision, device: str) -> Iterator[None]:
    require_device(device)
    previous_threads = torch.get_num_threads()
    previous_precision = torch.get_float32_matmul_precision()
    previous_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_num_threads(threads)
        apply_matmul_precision(precision)
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cudnn.allow_tf32 = previous_cudnn
        torch.set_num_threads(previous_threads)


@contextmanager
def preserve_rng(device: str) -> Iterator[None]:
    with torch.random.fork_rng(
        devices=[0] if device in {"cuda", "mps"} else [],
        device_type="mps" if device == "mps" else "cuda",
    ):
        yield


def load_pretrained(
    agent: ActorCritic, weights: dict[str, torch.Tensor], source: ModelConfig
) -> None:
    """Copy published weights into `agent` in place; strict unless a memory branch.

    A memory encoder started from a same-layout control keeps its newly
    initialized encoder, gates and projections. Every other tensor must come
    from the published policy, with its exact shape.
    """
    if not pretrained_memory_branch(
        source, agent.config, agent.observation_space.config
    ):
        agent.load_state_dict(weights, strict=True)
        return
    fresh = agent.state_dict()
    memory = {
        name
        for name in fresh
        if name.startswith("decision_memory.")
        or name.endswith((".memory_gate", ".memory_projection.weight"))
    }
    if set(fresh) - set(weights) != memory or set(weights) - set(fresh):
        raise ValueError(
            "pretrained copy would leave non-memory parameters unused or uninitialized"
        )
    agent.load_state_dict({**fresh, **weights}, strict=True)


def memory_metrics(device: str) -> dict[str, int]:
    if device == "cuda":
        return {
            "memory/peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
            "memory/peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(),
        }
    elif device == "mps":
        return {
            "memory/allocated_mps_bytes": torch.mps.current_allocated_memory(),
            "memory/driver_allocated_mps_bytes": torch.mps.driver_allocated_memory(),
        }
    return {}


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
