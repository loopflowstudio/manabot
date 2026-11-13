"""Rollout admission for the updates: a collected batch's observations as the
torch policy reads them.

`AdmittedRollout` admits a batch's observations once before its minibatches.
The updates reach it as `backend.rollout.AdmittedRollout`; every backend has
one with these methods over its own policy input.
"""

import numpy as np
from numpy.typing import NDArray
import torch

from manabot.backends.torch.arrays import _array_fields, on_device
from manabot.backends.torch.policy import ActorCritic
from manabot.policy.buckets import row_needs
from manabot.policy.prepare import Prepared
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
        self.agent.check_observation(batch.next_obs)

    def admit_rows(self, indexes: NDArray[np.int64], chunk_rows: int) -> None:
        """Admit flat rollout `indexes` not yet admitted."""
        pending = np.unique(indexes[~self._admitted[indexes]])
        for start in range(0, len(pending), chunk_rows):
            chunk = pending[start : start + chunk_rows]
            rows = self.batch.observations(chunk)
            obs = rows.observation
            self.agent.check_observation(obs)
            self._admitted[chunk] = True
            needs = (
                self.batch.recorded.gather_needs(chunk)
                if self.agent.bucketed_shapes
                else None
            )
            if needs is None and self.agent.bucketed_shapes:
                needs = row_needs(obs, self.agent.observation_space.encoder)
            if needs is not None:
                if self._needs is None:
                    self._needs = np.zeros(
                        (len(self._admitted), needs.shape[1]), np.int64
                    )
                self._needs[chunk] = needs
                self._measured[chunk] = True

    def next_observation(self, device: torch.device) -> Prepared[torch.Tensor]:
        """Bootstrap observations [trajectory, ...] on `device`."""
        return on_device(
            self.agent.prepare_admitted(
                self.batch.next_obs, inference=not torch.is_grad_enabled()
            ),
            device,
        )

    def rows(
        self, indexes: NDArray[np.int64], device: torch.device
    ) -> tuple[Prepared[torch.Tensor], ActionRows]:
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
        prepared = self.agent.prepare_admitted(
            rows.observation, needs=needs, inference=not torch.is_grad_enabled()
        )
        return on_device(prepared, device), rows

    def legal(self, obs: Prepared[torch.Tensor]) -> torch.Tensor:
        """The 0/1 legal-action input [B,A] of rows this rollout returned."""
        return obs.legal

    def evidence(self, obs: Prepared[torch.Tensor]) -> dict[str, torch.Tensor]:
        """The arrays a rejected step retains for rows this rollout returned."""
        return _array_fields(obs, torch.Tensor)
