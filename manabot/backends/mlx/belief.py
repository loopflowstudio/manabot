"""Joint hidden-hand probabilities and seeded draws for MLX search and fitting.

The recurrent model corrects the physical odds in vocabulary order. Public
card counts and combinatorics stay on the host in double precision. Only
relative log odds enter MLX, where gradients flow through the learned logits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from manabot.backends.mlx.arrays import flat_numpy
from manabot.backends.mlx.gru import PackedGRU
from manabot.belief.sampling_types import SamplerInput, SamplerSchema
from manabot.sim.search.sampling_receipt import (
    HandBatch,
    PreparedHands,
    draw_hand_batch,
)


@dataclass(frozen=True)
class PhysicalDeals:
    """Per-row unknown pool, guaranteed cards, and slots left to fill."""

    residual: np.ndarray
    minima: np.ndarray
    slots: np.ndarray
    log_factorials: np.ndarray

    @classmethod
    def from_inputs(cls, inputs: Sequence[SamplerInput]) -> PhysicalDeals:
        if not inputs or len({len(row.pool_counts) for row in inputs}) != 1:
            raise ValueError("a nonempty batch with one vocabulary width is required")
        if len({row.vocabulary_identity for row in inputs}) != 1:
            raise ValueError("batch mixes vocabulary identities")
        minima = np.array([row.known_minima for row in inputs], np.int32)
        residual = np.array([row.pool_counts for row in inputs], np.int32) - minima
        slots = np.array([row.hand_size for row in inputs]) - minima.sum(1)
        limit = residual.sum(1).max()
        factorials = np.r_[0.0, np.log(np.arange(1, limit + 1)).cumsum()]
        return cls(residual, minima, slots, factorials)

    def unknown(self, targets: np.ndarray) -> np.ndarray:
        if targets.shape != self.minima.shape or targets.dtype.kind not in "iu":
            raise ValueError("hand labels must be integer [batch, vocabulary] counts")
        unknown = targets.astype(np.int64) - self.minima
        if (
            np.any(unknown < 0)
            or np.any(unknown > self.residual)
            or not np.array_equal(unknown.sum(1), self.slots)
        ):
            raise ValueError(
                "hand label violates the public compatible-deal constraints"
            )
        return unknown

    def log_choose(self, n: np.ndarray, k: np.ndarray) -> np.ndarray:
        safe = np.clip(k, 0, n)
        f = self.log_factorials
        return np.where(k == safe, f[n] - f[safe] - f[n - safe], -np.inf)

    def log_prob(self, counts: np.ndarray) -> np.ndarray:
        return self.log_choose(self.residual, self.unknown(counts)).sum(
            1
        ) - self.log_choose(self.residual.sum(1), self.slots)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        counts = self.minima.copy()
        for i, slots in enumerate(self.slots):
            if slots:
                counts[i] += rng.multivariate_hypergeometric(
                    self.residual[i], slots, method="marginals"
                )
        return counts


class AutoregressiveBeliefSampler(nn.Module):
    """Teacher-force preceding counts for joint NLL, or draw them for search."""

    def __init__(
        self, schema: SamplerSchema, hidden_size: int = 64, history_dropout: float = 0.0
    ) -> None:
        super().__init__()
        if hidden_size < 1 or not 0 <= history_dropout <= 1:
            raise ValueError("invalid hidden size or history dropout")
        self.schema = schema
        self.hidden_size = hidden_size
        self.history_dropout = history_dropout
        width = len(schema.card_names)
        self.context = nn.Linear(2 * width + 1 + schema.history_size, hidden_size)
        self.card = nn.Embedding(width, hidden_size)
        self.recurrent = PackedGRU(hidden_size + 1, hidden_size)
        self.correction = nn.Linear(hidden_size, schema.max_count + 1)
        self.correction.weight = mx.zeros_like(self.correction.weight)
        self.correction.bias = mx.zeros_like(self.correction.bias)

    def decode(
        self,
        inputs: Sequence[SamplerInput],
        targets: np.ndarray | None,
        key: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        for row in inputs:
            row.validate(self.schema)
        deals = PhysicalDeals.from_inputs(inputs)
        unknown = deals.unknown(targets) if targets is not None else None
        history = mx.array([row.history_features for row in inputs], mx.float32)
        if self.training and self.history_dropout:
            history = history * (
                mx.random.uniform(shape=(len(inputs), 1)) >= self.history_dropout
            )
        remaining = deals.slots
        context = np.column_stack((deals.residual, deals.minima, remaining))
        hidden = mx.tanh(
            self.context(
                mx.concatenate([mx.array(context, mx.float32), history], axis=1)
            )
        )
        previous = mx.zeros((len(inputs), 1))
        later = deals.residual.sum(1)
        candidates = np.arange(self.schema.max_count + 1)
        log_prob = mx.zeros((len(inputs),))
        counts = []
        for column in range(len(self.schema.card_names)):
            token = mx.broadcast_to(self.card.weight[column], hidden.shape)
            hidden = self.recurrent.step(
                mx.concatenate([token, previous], axis=1), hidden
            )
            available = deals.residual[:, column, None]
            later = later - available[:, 0]
            base = deals.log_choose(available, candidates) + deals.log_choose(
                later[:, None], remaining[:, None] - candidates
            )
            # Subtract the common offset before narrowing physical odds to float32.
            logits = mx.array(
                base - base.max(1, keepdims=True), mx.float32
            ) + self.correction(hidden)
            if unknown is None:
                assert key is not None
                key, draw_key = mx.random.split(key)
                count = mx.random.categorical(logits, key=draw_key)
                host_count = np.asarray(count)
            else:
                host_count = unknown[:, column]
                count = mx.array(host_count, mx.int32)
            conditional = nn.log_softmax(logits, axis=1)
            log_prob = (
                log_prob + mx.take_along_axis(conditional, count[:, None], axis=1)[:, 0]
            )
            counts.append(count)
            remaining = remaining - host_count
            previous = count[:, None].astype(mx.float32) / self.schema.max_count
        return mx.stack(counts, axis=1) + mx.array(deals.minima), log_prob

    def log_prob(
        self, inputs: Sequence[SamplerInput], hand_counts: np.ndarray | mx.array
    ) -> mx.array:
        return self.decode(inputs, np.asarray(hand_counts), None)[1]

    def draw(
        self, inputs: Sequence[SamplerInput], rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        key = mx.random.key(int(rng.bit_generator.random_raw()))
        counts, logs = self.decode(inputs, None, key)
        mx.eval(counts, logs)
        return np.asarray(counts), np.asarray(logs)

    def sample_hands(
        self,
        prepared: PreparedHands,
        *,
        count: int,
        seed: int,
        check: Callable[[], None],
    ) -> HandBatch:
        rng = np.random.default_rng(seed)

        def draw() -> tuple[list[int], float]:
            counts, logs = self.draw([prepared.inputs], rng)
            return counts[0].tolist(), float(logs[0])

        return draw_hand_batch(prepared, count, seed, check, draw)


class PhysicalDistribution:
    def sample_hands(
        self,
        prepared: PreparedHands,
        *,
        count: int,
        seed: int,
        check: Callable[[], None],
    ) -> HandBatch:
        rng = np.random.default_rng(seed)
        deals = PhysicalDeals.from_inputs([prepared.inputs])

        def draw() -> tuple[list[int], float]:
            counts = deals.sample(rng)
            return counts[0].tolist(), float(deals.log_prob(counts)[0])

        return draw_hand_batch(prepared, count, seed, check, draw)


def sampler_parameters(model: object) -> dict[str, np.ndarray]:
    state = flat_numpy(cast(AutoregressiveBeliefSampler, model).parameters())
    state["recurrent.input_weight"] = state.pop("recurrent.Wx")
    state["recurrent.hidden_weight"] = state.pop("recurrent.Wh")
    return state


def restore_sampler(
    schema: SamplerSchema,
    hidden_size: int,
    history_dropout: float,
    parameters: Mapping[str, np.ndarray],
) -> AutoregressiveBeliefSampler:
    model = AutoregressiveBeliefSampler(schema, hidden_size, history_dropout)
    state = dict(parameters)
    state["recurrent.Wx"] = state.pop("recurrent.input_weight")
    state["recurrent.Wh"] = state.pop("recurrent.hidden_weight")
    model.load_weights([(k, mx.array(v)) for k, v in state.items()], strict=True)
    model.eval()
    return model


def freeze_sampler(model: object) -> None:
    cast(AutoregressiveBeliefSampler, model).freeze()
