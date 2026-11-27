"""Torch collection numerics: frozen forwards and the collector's private RNG.

The same generator samples both learning rows and current-self opponents.
Scopes retain the established inference preparation and forward modes.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import cast

import numpy as np
import torch

from manabot.backends.torch.arrays import array_fields, device_of, on_device, to_numpy
from manabot.backends.torch.learner import forward_precision
from manabot.backends.torch.policy import (
    ActorCritic,
    PolicyInference,
    collection_memory,
    move_inference_prepared,
    prepare_inference_host,
)
from manabot.backends.torch.rl import NumericalError, policy_logs, require
from manabot.backends.torch.rollout import check_packed, packing_mask
from manabot.env.observation import ObservationBuffers
from manabot.infra.profiler import Profiler
from manabot.policy.collection_memory import CollectionMemory
from manabot.policy.prepare import Prepared
from manabot.policy.protocol import BackendPolicy
from manabot.policy.spans import ObservationSelection
from manabot.training.learning import Precision
from manabot.training.rollout.numerics import (
    InferredChunk,
    SampledRows,
    prepared_chunks,
)
from manabot.training.rollout.selection_data import SelectionDraw

__all__ = [
    "CollectionSession",
]


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

    @property
    def supports_pipeline(self) -> bool:
        return self.device.type in {"cpu", "cuda"}

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


@dataclass(frozen=True)
class ForwardRows:
    logs: torch.Tensor
    values: torch.Tensor
    outcomes: np.ndarray | None
    needs: np.ndarray | None


class CollectionScope:
    def __init__(
        self, session: CollectionSession, agent: ActorCritic, inference: PolicyInference
    ) -> None:
        self.session, self.agent, self.inference = session, agent, inference
        self._cohort_memory = (inference.memory, collection_memory())

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
        parts: list[ForwardRows] = []
        with torch.no_grad():
            for obs in prepared_chunks(
                observation, execution_rows, self.inference.prepare, profiler
            ):
                with profiler.track("Policy inference and action selection"):
                    parts.append(self.forward_step(obs))
                del obs
            return self.sample_steps(parts)

    def prepare_step(
        self,
        observation: ObservationSelection,
        profiler: Profiler,
        *,
        cohort: int,
        execution_rows: int | None,
    ) -> Iterator[Callable[[], InferredChunk]]:
        """Prepare one host chunk at a time; transfer and inference run on the worker."""
        memory = self._cohort_memory[cohort]
        stream = (
            torch.cuda.current_stream(self.inference.device)
            if self.inference.device.type == "cuda"
            else None
        )

        def prepare(
            rows: ObservationBuffers,
        ) -> tuple[ObservationBuffers, Prepared[np.ndarray], Mapping[str, int] | None]:
            batch, roots = prepare_inference_host(self.inference, rows, memory)
            return rows, batch, roots

        remaining = len(observation.coordinates)
        width = execution_rows or remaining
        parts: list[ForwardRows] = []
        for rows, batch, roots in prepared_chunks(
            observation, execution_rows, prepare, profiler
        ):
            remaining -= width
            yield partial(
                self.infer_step,
                rows,
                batch,
                roots,
                memory,
                stream,
                parts,
                last=remaining <= 0,
            )
            del rows, batch, roots

    @torch.no_grad()
    def infer_step(
        self,
        rows: ObservationBuffers,
        batch: Prepared[np.ndarray],
        roots: Mapping[str, int] | None,
        memory: CollectionMemory[torch.Tensor],
        stream: torch.cuda.Stream | None,
        parts: list[ForwardRows],
        *,
        last: bool,
    ) -> InferredChunk:
        start = perf_counter()
        # Use the caller's stream so weights and cached GRU states keep device order.
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            parts.append(
                self.forward_step(
                    move_inference_prepared(self.inference, rows, batch, roots, memory)
                )
            )
            sampled = self.sample_steps(parts) if last else None
        return InferredChunk(perf_counter() - start, sampled)

    def forward_step(self, obs: Prepared[torch.Tensor]) -> ForwardRows:
        with forward_precision(self.session.precision, self.session.device):
            logits, value = self.agent.forward_distribution(obs)
        logits, value = logits.float(), value.float()
        probabilities = None
        if self.agent.config.value_kind == "categorical_wdl":
            outcomes = value.softmax(-1)
            value = outcomes[:, 2] - outcomes[:, 0]
            probabilities = outcomes.cpu().numpy()
        logs = self._policy_logs(logits, obs)
        require(
            torch.isfinite(value),
            "nonfinite_collection_values",
            values=value,
            valid=obs.legal,
        )
        return ForwardRows(logs, value.reshape(-1), probabilities, obs.bucket_needs)

    def sample_steps(self, parts: list[ForwardRows]) -> SampledRows:
        logs = torch.cat([part.logs for part in parts])
        values = torch.cat([part.values for part in parts])
        # One call in the original row order preserves the sampling stream.
        actions = torch.multinomial(logs.exp(), 1, generator=self.session.rng).squeeze(
            -1
        )
        selected = logs.gather(-1, actions[:, None]).squeeze(-1)
        outcomes = [part.outcomes for part in parts if part.outcomes is not None]
        needs = [part.needs for part in parts if part.needs is not None]
        return SampledRows(
            actions.cpu().numpy().astype(np.int64),
            selected.cpu().numpy(),
            values.cpu().numpy(),
            logs.cpu().numpy(),
            np.concatenate(outcomes) if outcomes else None,
            np.concatenate(needs) if needs else None,
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
