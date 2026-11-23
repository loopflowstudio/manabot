"""MLX collection numerics: frozen forwards and a private split-key RNG."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
import numpy as np

from manabot.backends.mlx.arrays import mlx_device, to_numpy
from manabot.backends.mlx.policy import MLXActorCritic, PolicyInference
from manabot.backends.mlx.rl import (
    NumericalError,
    action_logs,
    policy_logs,
    require,
)
from manabot.backends.mlx.rollout import check_packed, observation_arrays, packing_mask
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
        self.device_name = device
        self.device = mlx_device(device)
        self.precision = precision
        self._key = mx.random.key(seed + 1)

    def next_key(self) -> mx.array:
        self._key, draw = mx.random.split(self._key)
        return draw

    def snapshot_rng(self) -> object:
        return np.array(self._key).copy()

    @contextmanager
    def scope(self, agent: BackendPolicy) -> Iterator["CollectionScope"]:
        mlx_agent = cast(MLXActorCritic, agent)
        expected = "bfloat16" if self.precision == "bf16" else self.precision
        if mlx_agent.precision != expected or mlx_agent.inference_device != self.device:
            raise ValueError(
                f"MLX collection requires {self.device_name}/{expected}, "
                f"policy uses {mlx_agent.device_type}/{mlx_agent.precision}"
            )
        with PolicyInference(mlx_agent) as inference:
            yield CollectionScope(self, inference)


@dataclass
class CollectionScope:
    session: CollectionSession
    inference: PolicyInference

    def opponent_actions(
        self, observation: ObservationBuffers, *, execution_rows: int | None = None
    ) -> np.ndarray:
        return self.sample(
            observation, Profiler(), execution_rows=execution_rows
        ).actions

    @staticmethod
    def _policy_logs(logits: mx.array, obs: Prepared[np.ndarray]) -> mx.array:
        try:
            return policy_logs(logits, mx.array(obs.legal))
        except NumericalError as error:
            error.tensors.update(
                {
                    f"observation/{name}": mx.array(value)
                    for name, value in observation_arrays(obs).items()
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
        count = len(observation["actions_valid"])
        width = count if execution_rows is None else execution_rows
        if width < 1:
            raise ValueError("execution rows must be positive")
        logs_parts: list[mx.array] = []
        value_parts: list[mx.array] = []
        needs_parts: list[np.ndarray] = []
        agent = self.inference.agent
        with mx.stream(self.session.device), agent.forward_precision():
            self.inference.cache_programs()
            for start in range(0, count, width):
                with profiler.track("Observation preparation"):
                    obs = self.inference.prepare(
                        select_rows(
                            observation, np.arange(start, min(start + width, count))
                        )
                    )
                with profiler.track("Policy inference and action selection"):
                    logits, value = agent.forward_distribution(obs)
                    logits = logits.astype(mx.float32)
                    value = value.astype(mx.float32)
                    logs_parts.append(self._policy_logs(logits, obs))
                    require(
                        mx.isfinite(value),
                        "nonfinite_collection_values",
                        values=value,
                        valid=mx.array(obs.legal),
                    )
                    value_parts.append(value.reshape(-1))
                    if obs.bucket_needs is not None:
                        needs_parts.append(np.asarray(obs.bucket_needs, dtype=np.int64))
                del obs, logits, value
            # Draw once for the original group, independent of execution chunk size.
            logs = mx.concatenate(logs_parts)
            values = mx.concatenate(value_parts)
            actions = mx.random.categorical(logs, key=self.session.next_key())
            selected = action_logs(logs, actions)
            mx.eval(actions, selected, values, logs)
        return SampledRows(
            to_numpy(actions).astype(np.int64),
            to_numpy(selected),
            to_numpy(values),
            to_numpy(logs),
            None,
            np.concatenate(needs_parts) if needs_parts else None,
        )


class SelectionPredictor:
    """Frozen MLX policy predictions with a private split-key sampling stream."""

    def __init__(self, agent: MLXActorCritic, seed: int) -> None:
        self.agent = agent
        self._key = mx.random.key(seed)

    def predict(self, encoded: dict[str, np.ndarray]) -> SelectionDraw:
        batch = {key: value[None] for key, value in encoded.items()}
        prepared = self.agent.prepare(batch, inference=True)
        logits_np, value_np = self.agent.evaluate(prepared)
        logits = mx.array(logits_np)
        probabilities = mx.softmax(logits, axis=-1)
        self._key, draw = mx.random.split(self._key)
        action = int(mx.random.categorical(logits[0], key=draw))
        return SelectionDraw(action, float(value_np[0]), to_numpy(probabilities[0]))
