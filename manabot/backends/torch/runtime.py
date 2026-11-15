"""Torch execution setup, resource measurements and policy construction.

Host execution owns budgets and stage order; this module owns framework/device
choices already made by its composition root. No workload probes run here.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import cast

import torch

from manabot.backends.torch.learner import apply_matmul_precision
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy.protocol import BackendPolicy
from manabot.training.mps import mps_torch_error
from manabot.training.regime import Precision


def cuda_peak_metrics(device: str) -> dict[str, int]:
    """This process's CUDA high-water marks so far; absent off CUDA.

    Reserved bytes include the allocator's cache, so they are what a card must
    hold; allocated bytes are what the tensors themselves needed.
    """
    if device != "cuda":
        return {}
    return {
        "memory/peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
        "memory/peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(),
    }


def mps_memory_metrics(device: str) -> dict[str, int]:
    """Current MPS tensor/driver bytes, not peaks or additive memory totals."""
    if device != "mps":
        return {}
    return {
        "memory/allocated_mps_bytes": torch.mps.current_allocated_memory(),
        "memory/driver_allocated_mps_bytes": torch.mps.driver_allocated_memory(),
    }


def require_execution() -> None:
    """Torch supports the current regime and exact recovery contract."""


def identities(devices: Sequence[str]) -> dict[str, object]:
    result: dict[str, object] = {"torch": torch.__version__}
    if "cuda" in devices:
        require_device("cuda")
        gpu = torch.cuda.get_device_properties(0)
        result["cuda"] = {
            "device": gpu.name,
            "memory_bytes": gpu.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
            "runtime": torch.version.cuda,
        }
    if "mps" in devices:
        require_device("mps")
        if (error := mps_torch_error()) is not None:
            raise ValueError(error)
    return result


def require_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; CPU fallback is forbidden")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable; CPU fallback is forbidden")


def configure(threads: int, precision: Precision, device: str) -> None:
    torch.set_num_threads(threads)
    apply_matmul_precision(precision)
    require_device(device)


def threads() -> int:
    return torch.get_num_threads()


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def create_policy(
    space: ObservationSpace,
    config: ModelConfig,
    *,
    bucketed_shapes: bool = False,
    live_tokens: bool = False,
) -> ActorCritic:
    return ActorCritic(
        space, config, bucketed_shapes=bucketed_shapes, live_tokens=live_tokens
    )


def freeze(policy: BackendPolicy) -> None:
    cast(ActorCritic, policy).requires_grad_(False)


def policy_device(policy: BackendPolicy) -> str:
    return str(next(cast(ActorCritic, policy).parameters()).device)


@contextmanager
def preserve_policy_rng(policy: BackendPolicy) -> Iterator[None]:
    device = next(cast(ActorCritic, policy).parameters()).device.type
    with torch.random.fork_rng(
        devices=[0] if device in {"cuda", "mps"} else [],
        device_type="mps" if device == "mps" else "cuda",
    ):
        yield
