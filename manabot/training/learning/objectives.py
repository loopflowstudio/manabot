"""Deprecated: the update rules moved to `manabot.training.loops`.

`update_iteration` here keeps its 0.2 behaviour, choosing the rule from the
learning record; new code calls `loops.ppo.update_iteration` or
`loops.damped_policy.update_damped_iteration` directly.
"""

import warnings

import numpy as np

from manabot.infra.deprecation import moved
from manabot.model.actor_critic import ActorCritic
from manabot.training.learning.learner import Learner
from manabot.training.learning.selection import UpdateDiagnostics
from manabot.training.loops import ppo
from manabot.training.loops.damped_policy import update_damped_iteration
from manabot.training.regime import DampedPolicyLearning, Learning
from manabot.training.rollout.collection import RolloutBatch


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
    """Deprecated dispatch to the PPO or damped rule; removed in 0.3.0."""
    warnings.warn(
        f"{__name__}.update_iteration is deprecated; call "
        "manabot.training.loops.ppo.update_iteration or "
        "manabot.training.loops.damped_policy.update_damped_iteration. "
        "It will be removed in 0.3.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(learning, DampedPolicyLearning):
        return update_damped_iteration(
            learner, batch, learning, iteration, bootstrap_agent=bootstrap_agent
        )
    return ppo.update_iteration(
        learner,
        batch,
        learning,
        progress,
        rng,
        iteration=iteration,
        bootstrap_agent=bootstrap_agent,
    )


__getattr__ = moved(__name__, {"update_ema": "manabot.rl.step"})
