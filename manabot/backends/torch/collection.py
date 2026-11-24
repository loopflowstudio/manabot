"""Torch collection numerics: frozen forwards and the collector's private RNG.

The same generator samples both learning rows and current-self opponents.
Scopes retain the established inference preparation and forward modes.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import numpy as np
import torch

from manabot.backends.torch.arrays import array_fields, device_of, on_device, to_numpy
from manabot.backends.torch.learner import forward_precision
from manabot.backends.torch.policy import ActorCritic, PolicyInference
from manabot.backends.torch.rl import NumericalError, policy_logs, require
from manabot.backends.torch.rollout import check_packed, packing_mask
from manabot.env.memory_store import select_rows
from manabot.env.observation import ObservationBuffers
from manabot.infra.profiler import Profiler
from manabot.policy.prepare import Prepared
from manabot.policy.protocol import BackendPolicy
from manabot.training.learning import Precision
from manabot.training.rollout.numerics import SampledRows
from manabot.training.rollout.selection_data import SelectionDraw


class CollectionSession:
    packing_mask = staticmethod(packing_mask)
    check_packed = staticmethod(check_packed)

    def __init__(
        self, *, device: str, seed: int, precision: Precision = "float32"
    ) -> None:
        self.device = torch.device(device)
        self.precision = precision
        self.rng = torch.Generator(device=self.device).manual_seed(seed + 1)

    @property
    def device_name(self) -> str:
        return str(self.device)

    def snapshot_rng(self) -> torch.Tensor:
        return self.rng.get_state().clone()

    def restore_rng(self, state: object) -> None:
        if not isinstance(state, torch.Tensor):
            raise ValueError("expected Torch generator state")
        self.rng.set_state(state)

    @contextmanager
    def scope(self, agent: BackendPolicy) -> Iterator["CollectionScope"]:
        with PolicyInference(cast(ActorCritic, agent)) as inference:
            if inference.device.type != self.device.type or (
                self.device.index is not None
                and inference.device.index != self.device.index
            ):
                raise ValueError("collector and inference policy devices differ")
            yield CollectionScope(self, cast(ActorCritic, agent), inference)


class CollectionScope:
    def __init__(
        self, session: CollectionSession, agent: ActorCritic, inference: PolicyInference
    ) -> None:
        self.session, self.agent, self.inference = session, agent, inference

    def opponent_actions(
        self, observation: ObservationBuffers, *, execution_rows: int | None = None
    ) -> np.ndarray:
        return self.sample(
            observation, Profiler(), execution_rows=execution_rows
        ).actions

    @staticmethod
    def _policy_logs(logits: torch.Tensor, obs: Prepared[torch.Tensor]) -> torch.Tensor:
        try:
            return policy_logs(logits, obs.legal)
        except NumericalError as error:
            error.tensors.update(
                {
                    f"observation/{key}": value.detach()
                    for key, value in array_fields(obs, torch.Tensor).items()
                }
            )
            raise

    def sample(
        self,
        observation: ObservationBuffers,
        profiler: Profiler,
        *,
        execution_rows: int | None = None,
    ) -> SampledRows:
        if execution_rows is not None and execution_rows < 1:
            raise ValueError("execution rows must be positive")
        count = len(observation["actions_valid"])
        width = execution_rows or count
        if width < 1:
            raise ValueError("execution rows must be positive")
        logs_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        outcome_parts: list[np.ndarray] = []
        needs_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, count, width):
                with profiler.track("Observation preparation"):
                    obs = self.inference.prepare(
                        select_rows(
                            observation, np.arange(start, min(start + width, count))
                        )
                    )
                with profiler.track("Policy inference and action selection"):
                    with forward_precision(self.session.precision, self.session.device):
                        logits, value = self.agent.forward_distribution(obs)
                    logits, value = logits.float(), value.float()
                    if self.agent.config.value_kind == "categorical_wdl":
                        outcomes = value.softmax(-1)
                        value = outcomes[:, 2] - outcomes[:, 0]
                        outcome_parts.append(outcomes.cpu().numpy())
                    logs_parts.append(self._policy_logs(logits, obs))
                    require(
                        torch.isfinite(value),
                        "nonfinite_collection_values",
                        values=value,
                        valid=obs.legal,
                    )
                    value_parts.append(value.reshape(-1))
                    if obs.bucket_needs is not None:
                        needs_parts.append(np.asarray(obs.bucket_needs, dtype=np.int64))
                # Only small outputs survive a chunk, never Prepared or activations.
                del obs, logits, value
            logs = torch.cat(logs_parts)
            values = torch.cat(value_parts)
            # One call in the original row order preserves the sampling stream.
            actions = torch.multinomial(
                logs.exp(), 1, generator=self.session.rng
            ).squeeze(-1)
            selected = logs.gather(-1, actions[:, None]).squeeze(-1)
        return SampledRows(
            actions.cpu().numpy().astype(np.int64),
            selected.cpu().numpy(),
            values.cpu().numpy(),
            logs.cpu().numpy(),
            np.concatenate(outcome_parts) if outcome_parts else None,
            np.concatenate(needs_parts) if needs_parts else None,
        )


class SelectionPredictor:
    """Frozen policy predictions and the retained selection-game sampling stream."""

    def __init__(self, agent: ActorCritic, seed: int) -> None:
        self.agent = agent
        self.generator = torch.Generator().manual_seed(seed)

    @torch.inference_mode()
    def predict(self, encoded: dict[str, np.ndarray]) -> SelectionDraw:
        agent, generator = self.agent, self.generator
        tensors = on_device(
            agent.prepare(
                {key: value[None] for key, value in encoded.items()},
                inference=True,
            ),
            device_of(agent),
        )
        logits, value = agent(tensors)
        probabilities = logits.softmax(-1)
        action = int(torch.multinomial(probabilities[0], 1, generator=generator))
        return SelectionDraw(action, float(value.item()), to_numpy(probabilities[0]))
