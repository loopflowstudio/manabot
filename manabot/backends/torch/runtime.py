"""Torch thread, precision and device setup, and policy construction.

The composition root has already chosen the device; a missing one is refused,
never replaced by CPU.
"""

from __future__ import annotations

import torch

from manabot.backends.torch.learner import apply_matmul_precision
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.training.learning import Precision


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
