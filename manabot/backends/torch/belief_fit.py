"""Torch autoregressive fitting, seeded hand sampling and checkpoint encoding.

`fit_belief_sampler` optimizes joint NLL on training games; `evaluate_sampler`
supplies hand scores and draws to the host sampler metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
import io
import math
import time
from typing import cast

import numpy as np
import torch

from manabot.backends.torch.belief import (
    AutoregressiveBeliefSampler,
    physical_deal_log_prob,
    sample_physical_deal,
)
from manabot.belief.sampling_data import SamplerDataset, SamplerExample
from manabot.belief.sampling_fit import (
    SamplerArmMetrics,
    SamplerEvaluation,
    SamplerFitMetrics,
    SamplerFitResult,
    score_sampler_arm,
)
from manabot.belief.sampling_types import SamplerInput
from manabot.infra.torch import deterministic_cpu


def _check(check: Callable[[], None] | None) -> None:
    if check is not None:
        check()


@torch.no_grad()
def _evaluate_arm(
    model: AutoregressiveBeliefSampler | None,
    examples: Sequence[SamplerExample],
    *,
    samples: int,
    seed: int,
    check: Callable[[], None] | None,
) -> SamplerArmMetrics:
    score_hand = physical_deal_log_prob if model is None else model.log_prob
    draw_hand = sample_physical_deal if model is None else model.sample
    generator = torch.Generator().manual_seed(seed)

    def log_prob(example: SamplerExample) -> float:
        target = torch.tensor([example.target_hand], dtype=torch.int64)
        return float(score_hand([example.inputs], target)[0])

    def sample(inputs: SamplerInput) -> np.ndarray:
        return draw_hand([inputs] * samples, generator=generator).cpu().numpy()

    return score_sampler_arm(examples, log_prob=log_prob, sample=sample, check=check)


def evaluate_physical_sampler(
    examples: Sequence[SamplerExample], *, samples: int = 32, seed: int = 0
) -> SamplerArmMetrics:
    """Measure saved labels against the physical prior without a model artifact."""
    if not examples or samples < 1:
        raise ValueError("evaluation needs examples and a positive sample count")
    return _evaluate_arm(None, examples, samples=samples, seed=seed, check=None)


def evaluate_sampler(
    model: AutoregressiveBeliefSampler,
    examples: Sequence[SamplerExample],
    *,
    samples: int = 32,
    seed: int = 0,
    check: Callable[[], None] | None = None,
) -> SamplerEvaluation:
    """Score untouched labels, with history dropout disabled in both arms.

    A caller collecting foreign-policy/adversarial histories must retain that
    cohort's own policy identity; these scores do not certify transfer from a
    frozen self-play policy. The physical-deal arm is the safe fallback.
    """
    if not examples or samples < 1:
        raise ValueError("evaluation needs examples and a positive sample count")
    previous = model.training
    model.eval()
    try:
        return SamplerEvaluation(
            examples=len(examples),
            learned=_evaluate_arm(
                model, examples, samples=samples, seed=seed, check=check
            ),
            physical_baseline=_evaluate_arm(
                None, examples, samples=samples, seed=seed, check=check
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
    """Optimize minibatch mean joint NLL; each step draws train rows with replacement.

    Gradients pass through teacher-forced conditional logits only. Full private
    labels are never features. History dropout is applied only while fitting;
    it changes the treatment and cannot promise robustness to opponent baiting.
    Callback failures abort immediately so the run owner can retain the attempt.
    """
    if (
        steps < 1
        or batch_size < 1
        or evaluation_samples < 1
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError(
            "fitting requires positive steps, batch size, evaluation samples and learning rate"
        )
    split_rows = {
        split: tuple(
            example
            for game in dataset.games
            if game.split == split
            for example in game.examples
        )
        for split in ("train", "validation", "test")
    }
    if any(not rows for rows in split_rows.values()):
        raise ValueError("all three whole-game splits need examples")
    game_ids = [game.game_id for game in dataset.games]
    if len(set(game_ids)) != len(game_ids):
        raise ValueError("game identities must be unique across dataset splits")
    _check(check)
    with deterministic_cpu(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AutoregressiveBeliefSampler(
            dataset.schema, hidden_size, history_dropout
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        training = split_rows["train"]
        started = time.perf_counter()
        model.train()
        for _ in range(steps):
            _check(check)
            indexes = torch.randint(len(training), (batch_size,)).tolist()
            rows = [training[index] for index in indexes]
            targets = torch.tensor([row.target_hand for row in rows], dtype=torch.int64)
            loss = -model.log_prob([row.inputs for row in rows], targets).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("sampler produced a non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        fit_seconds = time.perf_counter() - started
        model.eval()
        evaluations = {
            split: evaluate_sampler(
                model, rows, samples=evaluation_samples, seed=seed + index, check=check
            )
            for index, (split, rows) in enumerate(split_rows.items())
        }
    metrics = SamplerFitMetrics(
        train=evaluations["train"],
        validation=evaluations["validation"],
        test=evaluations["test"],
        optimizer_exposures=steps * batch_size,
        optimizer_steps=steps,
        history_dropout=history_dropout,
        seed=seed,
        fit_seconds=fit_seconds,
        parameter_bytes=sum(
            value.numel() * value.element_size() for value in model.parameters()
        ),
    )
    return SamplerFitResult(
        model,
        metrics,
        dataset.identity,
        dataset.policy_identity,
        dataset.world_identity,
    )


def encode_sampler(result: SamplerFitResult, dataset: SamplerDataset) -> bytes:
    model = cast(AutoregressiveBeliefSampler, result.model)
    payload = {
        "format": "manabot.autoregressive-belief-sampler/v1",
        "schema": asdict(dataset.schema),
        "schema_identity": dataset.schema.identity,
        "dataset_identity": dataset.identity,
        "policy_identity": dataset.policy_identity,
        "world_identity": dataset.world_identity,
        "hidden_size": model.hidden_size,
        "history_dropout": model.history_dropout,
        "state_dict": model.state_dict(),
        "metrics": asdict(result.metrics),
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()
