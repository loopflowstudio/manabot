"""Rollout admission for the updates.

AdmittedRollout admits a collected batch's observations once before its
minibatches. The updates step a `Learner` (`manabot.backends.torch.learner`);
execute_regime owns collection, clocks, tracking, exports and recovery. Native
stream count sets the damped rule's minibatch width.
"""

import numpy as np
from numpy.typing import NDArray
import torch

from manabot.backends.torch.policy import ActorCritic
from manabot.model.admission import (
    AdmittedObservation,
    admit_observation,
    bind_admitted,
    check_observation,
    measure_row_needs,
)
from manabot.model.inference import transfer_admitted
from manabot.training.rollout.actions import ActionRows
from manabot.training.rollout.collection import RolloutBatch


def observation_tensors(obs: dict[str, NDArray[np.generic]]) -> dict[str, torch.Tensor]:
    """CPU tensors in native dtypes, except the floating action-focus input."""
    return {
        key: torch.as_tensor(value).float()
        if key == "action_focus"
        else torch.as_tensor(value)
        for key, value in obs.items()
    }


class AdmittedRollout:
    """A rollout batch whose observations are admitted once for one policy.

    Construction admits the bootstrap rows. `admit_rows` admits the learner
    rows an update will train, in chunks of at most one minibatch so host
    memory stays bounded; rows filtered out never enter a model and are not
    admitted. Minibatches then re-read admitted rows from the unchanged batch
    without checks, so epochs add no admission work. Any model with the
    admitting configuration (a target or bootstrap copy) reads them. The batch
    must not change afterwards.
    """

    def __init__(self, agent: ActorCritic, batch: RolloutBatch) -> None:
        self.agent = agent
        self.batch = batch
        self._admitted = np.zeros(batch.actions.size, dtype=bool)
        # Bucket needs of admitted rows, recorded at collection or measured
        # once, so every epoch's minibatches take their maximum instead of
        # measuring again.
        self._needs: np.ndarray | None = None
        self._measured = np.zeros(batch.actions.size, dtype=bool)
        self._next = admit_observation(agent, observation_tensors(batch.next_obs))

    def admit_rows(self, indexes: NDArray[np.int64], chunk_rows: int) -> None:
        """Admit flat rollout `indexes` not yet admitted."""
        pending = np.unique(indexes[~self._admitted[indexes]])
        for start in range(0, len(pending), chunk_rows):
            chunk = pending[start : start + chunk_rows]
            rows = self.batch.observations(chunk)
            obs = observation_tensors(rows.observation)
            check_observation(self.agent, obs)
            self._admitted[chunk] = True
            needs = measure_row_needs(
                self.agent, obs, recorded=self.batch.recorded.gather_needs(chunk)
            )
            if needs is not None:
                if self._needs is None:
                    self._needs = np.zeros(
                        (len(self._admitted), needs.shape[1]), np.int64
                    )
                self._needs[chunk] = needs
                self._measured[chunk] = True

    def next_observation(self, device: torch.device) -> AdmittedObservation:
        """Bootstrap observations [trajectory, ...] on `device`."""
        return transfer_admitted(self._next, device)[0]

    def rows(
        self, indexes: NDArray[np.int64], device: torch.device
    ) -> tuple[AdmittedObservation, ActionRows]:
        """Admitted observations of flat rollout `indexes` on `device`, with
        their behavior distributions."""
        if not self._admitted[indexes].all():
            raise ValueError("rollout rows were not admitted")
        rows = self.batch.observations(indexes)
        needs = (
            self._needs[indexes]
            if self._needs is not None and self._measured[indexes].all()
            else None
        )
        admitted = bind_admitted(
            self.agent, observation_tensors(rows.observation), needs=needs
        )
        return transfer_admitted(admitted, device)[0], rows
