"""Trainable policy and Adam state shared by every RL objective.

A Learner bundles the trainable actor/critic and its optimizer, enabling train
mode on admission. Learning-rule functions update this state; execute_regime
owns collection, clocks, tracking, exports and recovery. Native stream count
sets the damped rule's minibatch width. There is no independent training loop.
The learner's profiler times learning phases between existing device waits;
it adds no synchronization and draws no training randomness.
"""

from typing import ContextManager

import numpy as np
from numpy.typing import NDArray
import torch

from manabot.infra.profiler import Profiler
from manabot.model.actor_critic import ActorCritic
from manabot.model.inference import transfer
from manabot.training.precision import forward_precision
from manabot.training.regime import Precision


class Learner:
    def __init__(
        self,
        agent: ActorCritic,
        device: torch.device | str,
        *,
        minibatch_rows: int,
        precision: Precision = "float32",
    ) -> None:
        if minibatch_rows < 1:
            raise ValueError("minibatch_rows must be positive")
        self.device = torch.device(device)
        # Checkpoint admission returns eval mode. Recurrent CUDA kernels need
        # a training-mode forward before backward, including on warm starts.
        self.agent = agent.to(self.device).train()
        self.minibatch_rows = minibatch_rows
        self.precision: Precision = precision
        # The learning rule sets the rate before stepping; retain the initial
        # Adam configuration in snapshots even when selection skips all rows.
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=2.5e-4, eps=1e-5)
        self.profiler = Profiler()

    def forward_precision(self) -> ContextManager[object]:
        """Wrap each policy forward; callers cast its outputs to float32."""
        return forward_precision(self.precision, self.device)


def observation_tensors(
    obs: dict[str, NDArray[np.generic]], device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Preserve native dtypes except the floating action-focus embedding input.

    Moves one copy per dtype rather than one per key, as collection does.
    """
    moved, _ = transfer(
        {
            key: torch.as_tensor(value).float()
            if key == "action_focus"
            else torch.as_tensor(value)
            for key, value in obs.items()
        },
        torch.device(device),
    )
    return moved
