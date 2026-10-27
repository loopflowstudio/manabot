"""Trainable policy and Adam state shared by every RL objective.

A Learner bundles the trainable actor/critic and its optimizer, enabling train
mode on admission. Learning-rule functions update this state; execute_regime
owns collection, clocks, tracking, exports and recovery. Native stream count
sets the damped rule's minibatch width. There is no independent training loop.
The learner's profiler times learning phases between existing device waits;
it adds no synchronization and draws no training randomness.
"""

import numpy as np
from numpy.typing import NDArray
import torch

from manabot.infra.profiler import Profiler
from manabot.model.actor_critic import ActorCritic


class Learner:
    def __init__(
        self,
        agent: ActorCritic,
        device: torch.device | str,
        *,
        minibatch_rows: int,
    ) -> None:
        if minibatch_rows < 1:
            raise ValueError("minibatch_rows must be positive")
        self.device = torch.device(device)
        # Checkpoint admission returns eval mode. Recurrent CUDA kernels need
        # a training-mode forward before backward, including on warm starts.
        self.agent = agent.to(self.device).train()
        self.minibatch_rows = minibatch_rows
        # The learning rule sets the rate before stepping; retain the initial
        # Adam configuration in snapshots even when selection skips all rows.
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=2.5e-4, eps=1e-5)
        self.profiler = Profiler()


def observation_tensors(
    obs: dict[str, NDArray[np.generic]], device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Preserve native dtypes except the floating action-focus embedding input."""
    return {
        key: torch.as_tensor(value, device=device).float()
        if key == "action_focus"
        else torch.as_tensor(value, device=device)
        for key, value in obs.items()
    }
