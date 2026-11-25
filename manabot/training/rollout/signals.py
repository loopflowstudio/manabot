"""Detached learner measurements in packed collection order."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BatchSignals:
    """Read-only host arrays [N] in flat rollout order, including discovery rows."""

    critic_residual: np.ndarray
    positive_value_loss: np.ndarray
    policy_entropy: np.ndarray
    wdl_entropy: np.ndarray | None
    action_value_spread: np.ndarray | None
