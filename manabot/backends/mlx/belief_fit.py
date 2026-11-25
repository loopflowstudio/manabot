"""Fit MLX recurrent corrections on training games and score the frozen sampler."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

from manabot.backends.mlx.belief import AutoregressiveBeliefSampler, PhysicalDeals
from manabot.backends.mlx.runtime import preserve_rng
from manabot.belief.sampling_data import SamplerDataset, SamplerExample
from manabot.belief.sampling_fit import (
    SamplerArmMetrics,
    SamplerEvaluation,
    SamplerFitResult,
    fit_splits,
    score_fit,
    score_sampler_arm,
)
from manabot.belief.sampling_types import SamplerInput


def evaluate_physical_sampler(
    examples: Sequence[SamplerExample],
    *,
    samples: int = 32,
    seed: int = 0,
    check: Callable[[], None] | None = None,
) -> SamplerArmMetrics:
    if not examples or samples < 1:
        raise ValueError("evaluation needs examples and a positive sample count")
    rng = np.random.default_rng(seed)

    def log_prob(example: SamplerExample) -> float:
        deals = PhysicalDeals.from_inputs([example.inputs])
        return float(deals.log_prob(np.array([example.target_hand]))[0])

    def sample(inputs: SamplerInput) -> np.ndarray:
        return PhysicalDeals.from_inputs([inputs] * samples).sample(rng)

    return score_sampler_arm(examples, log_prob=log_prob, sample=sample, check=check)


def evaluate_sampler(
    model: AutoregressiveBeliefSampler,
    examples: Sequence[SamplerExample],
    *,
    samples: int = 32,
    seed: int = 0,
    check: Callable[[], None] | None = None,
) -> SamplerEvaluation:
    if not examples or samples < 1:
        raise ValueError("evaluation needs examples and a positive sample count")
    rng = np.random.default_rng(seed)

    def log_prob(example: SamplerExample) -> float:
        return float(
            model.log_prob([example.inputs], np.array([example.target_hand]))[0]
        )

    def sample(inputs: SamplerInput) -> np.ndarray:
        return model.draw([inputs] * samples, rng)[0]

    previous = model.training
    model.eval()
    try:
        return SamplerEvaluation(
            len(examples),
            score_sampler_arm(examples, log_prob=log_prob, sample=sample, check=check),
            evaluate_physical_sampler(
                examples, samples=samples, seed=seed, check=check
            ),
        )
    finally:
        model.train(previous)


def fit_belief_sampler(
    dataset: SamplerDataset,
    *,
    steps: int,
    batch_size: int = 32,
    hidden_size: int = 64,
    learning_rate: float = 0.001,
    history_dropout: float = 0.0,
    evaluation_samples: int = 32,
    seed: int = 0,
    check: Callable[[], None] | None = None,
) -> SamplerFitResult:
    """Minimize teacher-forced joint NLL on train rows sampled with replacement."""
    splits = fit_splits(dataset, steps, batch_size, evaluation_samples, learning_rate)
    if check:
        check()
    rng = np.random.default_rng(seed)
    with preserve_rng():
        mx.random.seed(seed)
        model = AutoregressiveBeliefSampler(
            dataset.schema, hidden_size, history_dropout
        )
        optimizer = optim.Adam(learning_rate)

        def objective(rows: Sequence[SamplerExample]) -> mx.array:
            return -model.log_prob(
                [row.inputs for row in rows],
                np.array([row.target_hand for row in rows]),
            ).mean()

        loss_and_grad = nn.value_and_grad(model, objective)
        training = splits["train"]
        started = time.perf_counter()
        for _ in range(steps):
            if check:
                check()
            rows = [
                training[index]
                for index in rng.integers(len(training), size=batch_size)
            ]
            loss, gradients = loss_and_grad(rows)
            if not bool(mx.isfinite(loss)):
                raise ValueError("sampler produced a non-finite training loss")
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
        fit_seconds = time.perf_counter() - started
        model.eval()
        return score_fit(
            model,
            dataset,
            splits,
            evaluate_sampler,
            steps=steps,
            batch_size=batch_size,
            seed=seed,
            evaluation_samples=evaluation_samples,
            check=check,
            fit_seconds=fit_seconds,
            parameter_bytes=sum(
                value.nbytes for _, value in tree_flatten(model.parameters())
            ),
        )
