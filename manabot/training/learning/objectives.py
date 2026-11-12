"""Deprecated: the updates moved to `manabot.training.updates`.

`update_iteration` here keeps its 0.2 behaviour, choosing the update from
the learning record; new code calls `updates.ppo.update_iteration` or
`updates.damped_policy.update_damped_iteration` directly.
"""

import warnings

import numpy as np

import manabot.backends.torch as torch_backend
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.infra.deprecation import moved
from manabot.training.regime import DampedPolicyLearning, Learning
from manabot.training.rollout.collection import RolloutBatch
from manabot.training.updates import ppo
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.training.updates.selection import UpdateDiagnostics


def update_iteration(
    learner: Learner,
    batch: RolloutBatch,
    learning: Learning | DampedPolicyLearning,
    progress: float,
    rng: np.random.Generator,
    *,
    iteration: int = 1,
    bootstrap_agent: ActorCritic | None = None,
) -> UpdateDiagnostics:
    """Deprecated dispatch to the PPO or damped update; removed in 0.3.0."""
    warnings.warn(
        f"{__name__}.update_iteration is deprecated; call "
        "manabot.training.updates.ppo.update_iteration or "
        "manabot.training.updates.damped_policy.update_damped_iteration. "
        "It will be removed in 0.3.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(learning, DampedPolicyLearning):
        return update_damped_iteration(
            torch_backend,
            learner,
            batch,
            learning,
            iteration,
            bootstrap_agent=bootstrap_agent,
        )
    return ppo.update_iteration(
        torch_backend,
        learner,
        batch,
        learning,
        progress,
        rng,
        iteration=iteration,
        bootstrap_agent=bootstrap_agent,
    )


__getattr__ = moved(__name__, {"update_ema": "manabot.backends.torch.learner"})
