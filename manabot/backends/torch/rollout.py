"""Rollout admission for the updates: a collected batch's observations as the
torch policy reads them.

`AdmittedRollout` admits a batch's observations once before its minibatches.
The updates reach it as `backend.rollout.AdmittedRollout`; every backend has
one with these methods over its own policy input.
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
from numpy.typing import NDArray
import torch

from manabot.backends.torch.arrays import array_fields, on_device
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rl import (
    NumericalError,
    check_behavior,
    legal_mask,
    require,
)
from manabot.env.memory_store import PrefixObservation, select_rows
from manabot.policy.prepare import Prepared
from manabot.training.rollout.actions import ActionRows, PackedActions
from manabot.training.rollout.collection import RolloutBatch


class AdmittedRollout:
    """A rollout batch whose observations are admitted once for one policy.

    Bootstrap reads admit one execution chunk at a time. `admit_rows` admits the learner
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
        # Each epoch regroups the extents measured at learner admission.
        self._needs: np.ndarray | None = None
        self._measured = np.zeros(batch.actions.size, dtype=bool)

    def admit_rows(self, indexes: NDArray[np.int64], chunk_rows: int) -> None:
        """Admit flat rollout `indexes` not yet admitted."""
        pending = np.unique(indexes[~self._admitted[indexes]])
        for start in range(0, len(pending), chunk_rows):
            chunk = pending[start : start + chunk_rows]
            needs = self.agent.check_observation(
                self.batch.observations(chunk).observation
            )
            self._admitted[chunk] = True
            if needs is None:
                continue
            # Packing drops illegal offers. Keep their collection-time shape floor
            # while validating the current rollout, whose arrays are writable.
            recorded = self.batch.recorded.gather_needs(chunk)
            if recorded is not None:
                needs = np.maximum(needs, recorded)
            if self._needs is None:
                self._needs = np.zeros((len(self._admitted), needs.shape[1]), np.int64)
            self._needs[chunk] = needs
            self._measured[chunk] = True

    def next_observation(self, device: torch.device) -> Prepared[torch.Tensor] | None:
        """Inference-prepared bootstrap observations [trajectory, ...] on `device`."""
        return next(
            self.next_observations(device, max(1, len(self.batch.next_done))), None
        )

    def next_observations(
        self, device: torch.device, size: int
    ) -> Iterator[Prepared[torch.Tensor]]:
        """Admit, prepare and transfer one bootstrap chunk at a time."""
        if size < 1:
            raise ValueError("bootstrap rows must be positive")
        for start in range(0, len(self.batch.next_done), size):
            raw = select_rows(self.batch.next_obs, slice(start, start + size))
            prepared = self.agent.prepare(raw, inference=True)
            del raw
            yield on_device(prepared, device)
            del prepared

    def rows(
        self, indexes: NDArray[np.int64], device: torch.device
    ) -> tuple[Prepared[torch.Tensor], ActionRows]:
        """Learner-prepared observations of flat rollout `indexes` on `device`,
        with their behavior distributions, independent of autograd mode."""
        prepared, rows = self._prepare_rows(indexes)
        return on_device(prepared, device), rows

    def _prepare_rows(
        self, indexes: NDArray[np.int64]
    ) -> tuple[Prepared[np.ndarray], ActionRows]:
        if not self._admitted[indexes].all():
            raise ValueError("rollout rows were not admitted")
        rows = self.batch.observations(indexes)
        needs = (
            self._needs[indexes]
            if self._needs is not None and self._measured[indexes].all()
            else None
        )
        prepared = self.agent.prepare_admitted(
            rows.observation, needs=needs, inference=False
        )
        # Behavior/reference calculations need ordinary action fields, not a
        # second raw memory owner beside Prepared (including in the prefetch).
        if isinstance(rows.observation, PrefixObservation):
            raw = rows.observation
            rows.observation = PrefixObservation(
                raw, raw.prefixes, raw.memory_capacities
            )
        return prepared, rows

    @contextmanager
    def minibatches(
        self, order: NDArray[np.int64], size: int, device: torch.device
    ) -> Iterator[
        Iterator[tuple[NDArray[np.int64], Prepared[torch.Tensor], ActionRows]]
    ]:
        """Prepare learner rows one batch ahead, independent of autograd mode.

        Only immutable host input is read on the worker. Transfers and all
        model/optimizer work stay on the caller's device stream. At most two
        host minibatches are live; exiting joins the worker even on failure.
        """
        if size < 1:
            raise ValueError("minibatch size must be positive")
        chunks = [order[start : start + size] for start in range(0, len(order), size)]
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ppo-prepare"
        ) as worker:

            def batches() -> Iterator[
                tuple[NDArray[np.int64], Prepared[torch.Tensor], ActionRows]
            ]:
                # Lazy submission: the worker holds the batch being awaited
                # and at most the one after it.
                submitted = (
                    worker.submit(self._prepare_rows, chunk) for chunk in chunks
                )
                pending = next(submitted, None)
                for chunk in chunks:
                    ready, pending = pending, next(submitted, None)
                    assert ready is not None
                    prepared, rows = ready.result()
                    yield chunk, on_device(prepared, device), rows
                    del prepared, rows, ready

            yield batches()

    def legal(self, obs: Prepared[torch.Tensor]) -> torch.Tensor:
        """The 0/1 legal-action input [B,A] of rows this rollout returned."""
        return obs.legal

    def evidence(self, obs: Prepared[torch.Tensor]) -> dict[str, torch.Tensor]:
        """The arrays a rejected step retains for rows this rollout returned."""
        return array_fields(obs, torch.Tensor)


def packing_mask(
    mask: NDArray[np.generic], logs: NDArray[np.generic]
) -> NDArray[np.bool_]:
    """Validate legal support and padding before the host drops padding rows."""
    valid = legal_mask(torch.as_tensor(mask)).numpy()
    if valid.ndim != 2 or logs.shape != valid.shape:
        raise ValueError("packed actions require matching [B,A] distributions")
    if np.any(~np.isneginf(logs[~valid])):
        require(
            torch.as_tensor(np.isneginf(logs) | valid),
            "illegal_log_mass",
            logs=torch.as_tensor(logs),
            valid=torch.as_tensor(valid),
        )
    counts = valid.sum(-1)
    if np.any(counts == 0):
        require(
            torch.as_tensor(counts != 0),
            "empty_legal_support",
            valid=torch.as_tensor(valid),
        )
    return valid


def check_packed(
    packed: PackedActions,
    actions: NDArray[np.generic],
    selected_logs: NDArray[np.generic],
) -> None:
    """Admit all behavior evidence in bounded CPU chunks with original row ids."""
    if actions.size != len(packed.offsets) - 1 or selected_logs.shape != actions.shape:
        raise NumericalError(
            "behavior_shape",
            {
                "actions": torch.as_tensor(actions),
                "selected_logs": torch.as_tensor(selected_logs),
                "offsets": torch.as_tensor(packed.offsets),
            },
        )
    for start in range(0, len(packed.offsets) - 1, 32):
        indexes = np.arange(start, min(start + 32, len(packed.offsets) - 1))
        rows = packed.unpack(indexes)
        try:
            check_behavior(
                torch.as_tensor(rows.log_probabilities),
                torch.as_tensor(actions.reshape(-1)[indexes]),
                torch.as_tensor(selected_logs.reshape(-1)[indexes]),
                torch.as_tensor(rows.observation["actions_valid"]).bool(),
            )
        except NumericalError as error:
            # Preserve original rollout coordinates in incident evidence.
            error.tensors["minibatch_indexes"] = torch.as_tensor(indexes)
            raise
