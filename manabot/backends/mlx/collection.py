"""MLX collection numerics: frozen forwards and a private split-key RNG."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from typing import Literal, cast

import mlx.core as mx
import numpy as np

from manabot.backends.mlx.arrays import to_numpy
from manabot.backends.mlx.policy import MLXActorCritic, PolicyInference
from manabot.backends.mlx.rl import (
    NumericalError,
    action_logs,
    policy_logs,
    require,
)
from manabot.backends.mlx.rollout import check_packed, packing_mask
from manabot.env.memory_store import select_rows
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
        if device == "cuda":
            raise ValueError("MLX does not support CUDA")
        if device not in {"cpu", "mps", "gpu"}:
            raise ValueError(f"MLX does not support device {device!r}")
        self._device_name = device
        self._stream = mx.cpu if device == "cpu" else mx.gpu
        self.precision = precision
        self._key = mx.random.key(seed + 1)

    @property
    def device_name(self) -> str:
        return self._device_name

    def next_key(self) -> mx.array:
        self._key, draw = mx.random.split(self._key)
        return draw

    def snapshot_rng(self) -> object:
        return np.array(self._key).copy()

    @contextmanager
    def scope(self, agent: BackendPolicy) -> Iterator["CollectionScope"]:
        mlx_agent = cast(MLXActorCritic, agent)
        with PolicyInference(mlx_agent) as inference:
            yield CollectionScope(self, mlx_agent, inference)


class CollectionScope:
    def __init__(
        self,
        session: CollectionSession,
        agent: MLXActorCritic,
        inference: PolicyInference,
    ) -> None:
        self.session, self.agent, self.inference = session, agent, inference

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
                    f"observation/{item.name}": mx.array(value)
                    for item in fields(obs)
                    if isinstance(value := getattr(obs, item.name), np.ndarray)
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
        logs_parts: list[mx.array] = []
        value_parts: list[mx.array] = []
        needs_parts: list[np.ndarray] = []
        with mx.stream(self.session._stream), self.agent.forward_precision():
            self.inference.cache_programs()
            for start in range(0, count, width):
                with profiler.track("Observation preparation"):
                    obs = self.inference.prepare(
                        select_rows(
                            observation, np.arange(start, min(start + width, count))
                        )
                    )
                with profiler.track("Policy inference and action selection"):
                    logits, value = self.agent.forward_distribution(obs)
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
            logs = logs_parts[0] if len(logs_parts) == 1 else mx.concatenate(logs_parts)
            values = (
                value_parts[0] if len(value_parts) == 1 else mx.concatenate(value_parts)
            )
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

    def predict(
        self,
        encoded: dict[str, np.ndarray],
        reference: Literal["uniform", "action_type_uniform"],
    ) -> SelectionPrediction:
        prepared = self.agent.prepare(
            {key: value[None] for key, value in encoded.items()},
            inference=True,
        )
        logits_np, value_np = self.agent.evaluate(prepared)
        logits = mx.array(logits_np)
        probabilities = mx.softmax(logits, axis=-1)
        self._key, draw = mx.random.split(self._key)
        action = int(mx.random.categorical(logits[0], key=draw))
        probs_np = to_numpy(probabilities)
        logs = np.log(np.maximum(probs_np, 1e-12))
        ref = reference_distribution(
            {k: encoded[k][None] for k in ("actions_valid", "actions")},
            reference,
        )
        return SelectionPrediction(
            action,
            float(value_np[0]),
            float(probs_np[0, action]),
            float(-(probs_np * logs).sum()),
            float((probs_np * (logs - np.log(np.maximum(ref, 1e-12)))).sum()),
        )
