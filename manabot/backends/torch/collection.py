"""Torch collection numerics: frozen forwards and the collector's private RNG.

The same generator samples both learning rows and current-self opponents.
Scopes retain the established inference preparation and forward modes.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal, cast

import numpy as np
import torch

from manabot.backends.torch.arrays import array_fields, device_of, on_device
from manabot.backends.torch.learner import forward_precision
from manabot.backends.torch.policy import ActorCritic, PolicyInference
from manabot.backends.torch.rl import NumericalError, policy_logs, require
from manabot.backends.torch.rollout import check_packed, packing_mask
from manabot.env.observation import ObservationBuffers
from manabot.infra.profiler import Profiler
from manabot.policy.prepare import Prepared, reference_distribution
from manabot.policy.protocol import BackendPolicy
from manabot.training.learning import Precision
from manabot.training.rollout.numerics import SampledRows
from manabot.training.rollout.selection_data import SelectionPrediction


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

    def snapshot_rng(self) -> object:
        return self.rng.get_state().clone()

    def restore_rng(self, state: object) -> None:
        if not isinstance(state, torch.Tensor):
            raise TypeError("Torch collector RNG state requires a tensor")
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

    def opponent_actions(self, observation: ObservationBuffers) -> np.ndarray:
        obs = self.inference.prepare(observation)
        with torch.inference_mode():
            with forward_precision(self.session.precision, self.session.device):
                logits, _ = self.agent.forward(obs)
            logits = logits.float()
            probs = self._policy_logs(logits, obs).exp()
            actions = torch.multinomial(probs, 1, generator=self.session.rng).squeeze(
                -1
            )
        return actions.cpu().numpy().astype(np.int64)

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
        self, observation: ObservationBuffers, profiler: Profiler
    ) -> SampledRows:
        with profiler.track("Observation preparation"):
            obs_t = self.inference.prepare(observation)
        with profiler.track("Policy inference and action selection"):
            with torch.no_grad():
                outcomes: np.ndarray | None = None
                with forward_precision(self.session.precision, self.session.device):
                    if self.agent.config.value_kind == "categorical_wdl":
                        logits, value_t = self.agent.forward_distribution(obs_t)
                    else:
                        logits, value_t = self.agent(obs_t)
                logits, value_t = logits.float(), value_t.float()
                if self.agent.config.value_kind == "categorical_wdl":
                    outcome_t = value_t.softmax(dim=-1)
                    value_t = outcome_t[:, 2] - outcome_t[:, 0]
                    outcomes = outcome_t.cpu().numpy()
                logs = self._policy_logs(logits, obs_t)
                require(
                    torch.isfinite(value_t),
                    "nonfinite_collection_values",
                    values=value_t,
                    valid=obs_t.legal,
                )
                weights = logs.exp()
                action_t = torch.multinomial(
                    weights, 1, generator=self.session.rng
                ).squeeze(-1)
                logprob_t = logs.gather(-1, action_t[:, None]).squeeze(-1)
                log_probabilities = logs.cpu().numpy()
            acts = action_t.cpu().numpy().astype(np.int64)
            logprobs = logprob_t.cpu().numpy()
            values = value_t.view(-1).cpu().numpy()

        needs = (
            None
            if obs_t.bucket_needs is None
            else np.asarray(obs_t.bucket_needs, dtype=np.int64)
        )
        return SampledRows(acts, logprobs, values, log_probabilities, outcomes, needs)


class SelectionPredictor:
    """Frozen policy predictions and the retained selection-game sampling stream."""

    def __init__(self, agent: ActorCritic, seed: int) -> None:
        self.agent = agent
        self.generator = torch.Generator().manual_seed(seed)

    @torch.inference_mode()
    def predict(
        self,
        encoded: dict[str, np.ndarray],
        reference: Literal["uniform", "action_type_uniform"],
    ) -> SelectionPrediction:
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
        logs = probabilities.clamp_min(1e-12).log()
        ref = torch.from_numpy(
            reference_distribution(
                {k: encoded[k][None] for k in ("actions_valid", "actions")},
                reference,
            )
        )

        return SelectionPrediction(
            action,
            float(value.item()),
            float(probabilities[0, action]),
            float(-(probabilities * logs).sum()),
            float((probabilities * (logs - ref.clamp_min(1e-12).log())).sum()),
        )
