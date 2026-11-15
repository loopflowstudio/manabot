"""Torch autoregressive fitting, calibration and checkpoint encoding.

`fit_belief_sampler` optimizes joint NLL on training games; `evaluate_sampler`
scores held-out games. Metric records and artifact publication remain in the
host `belief.sampling_fit` module, independent of this numerical implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
import io
import math
import time
import tracemalloc
from typing import cast

import torch

from manabot.backends.torch.belief import (
    AutoregressiveBeliefSampler,
    physical_deal_log_prob,
    sample_physical_deal,
)
from manabot.belief.sampling_data import SamplerDataset, SamplerExample
from manabot.belief.sampling_fit import (
    CalibrationBin,
    SamplerArmMetrics,
    SamplerEvaluation,
    SamplerFitMetrics,
    SamplerFitResult,
    calibration_error,
)
from manabot.infra.torch import deterministic_cpu


def _check(check: Callable[[], None] | None) -> None:
    if check is not None:
        check()


def _calibration_bins(
    probabilities: torch.Tensor, truths: torch.Tensor
) -> tuple[CalibrationBin, ...]:
    # Ten fixed equal-width bins; the last includes probability one.
    buckets = (probabilities * 10).long().clamp(max=9)
    return tuple(
        CalibrationBin(
            int((buckets == index).sum()),
            float(probabilities[buckets == index].sum()),
            float(truths[buckets == index].sum()),
        )
        for index in range(10)
    )


@torch.no_grad()
def _evaluate_arm(
    model: AutoregressiveBeliefSampler | None,
    examples: Sequence[SamplerExample],
    *,
    physical: bool,
    samples: int,
    seed: int,
    check: Callable[[], None] | None,
) -> SamplerArmMetrics:
    if not physical and model is None:
        raise ValueError("learned evaluation requires a sampler")
    score_hand = physical_deal_log_prob if model is None or physical else model.log_prob
    draw_hand = sample_physical_deal if model is None or physical else model.sample
    generator = torch.Generator().manual_seed(seed)
    nll = 0.0
    forecasts: list[torch.Tensor] = []
    outcomes: list[torch.Tensor] = []
    conjunction_errors: list[torch.Tensor] = []
    violations = 0
    elapsed = 0.0
    largest_tensor = 0
    peak_python = 0
    owns_trace = not tracemalloc.is_tracing()
    if owns_trace:
        tracemalloc.start()
    try:
        for example in examples:
            _check(check)
            inputs = example.inputs
            target = torch.tensor([example.target_hand], dtype=torch.int64)
            score = score_hand([inputs], target)
            nll -= float(score[0])
            started = time.perf_counter()
            batch = [inputs] * samples
            draws = draw_hand(batch, generator=generator)
            elapsed += time.perf_counter() - started
            largest_tensor = max(largest_tensor, draws.numel() * draws.element_size())
            present = draws > 0
            probabilities = present.double().mean(dim=0)
            truth = (target[0] > 0).double()
            forecasts.append(probabilities)
            outcomes.append(truth)
            if draws.shape[1] > 1:
                joint = (present[:, :-1] & present[:, 1:]).double().mean(dim=0)
                conjunction_errors.append((joint - truth[:-1] * truth[1:]).square())
            low = torch.tensor(inputs.known_minima)
            high = torch.tensor(inputs.pool_counts)
            invalid = ((draws < low) | (draws > high)).any(dim=1) | (
                draws.sum(dim=1) != inputs.hand_size
            )
            violations += int(invalid.sum())
            peak_python = max(peak_python, tracemalloc.get_traced_memory()[1])
    finally:
        if owns_trace:
            tracemalloc.stop()
    probabilities = torch.cat(forecasts)
    truths = torch.cat(outcomes)
    bins = _calibration_bins(probabilities, truths)
    return SamplerArmMetrics(
        joint_nll=nll / len(examples),
        inclusion_brier=float((probabilities - truths).square().mean()),
        inclusion_ece=calibration_error(bins),
        conjunction_brier=float(torch.cat(conjunction_errors).mean())
        if conjunction_errors
        else None,
        support_violations=violations,
        sampled_hands=len(examples) * samples,
        sampling_seconds=elapsed,
        peak_python_bytes=peak_python,
        largest_sample_tensor_bytes=largest_tensor,
        calibration_bins=bins,
    )


def evaluate_physical_sampler(
    examples: Sequence[SamplerExample], *, samples: int = 32, seed: int = 0
) -> SamplerArmMetrics:
    """Measure saved labels against the physical prior without a model artifact."""
    if not examples or samples < 1:
        raise ValueError("evaluation needs examples and a positive sample count")
    return _evaluate_arm(
        None, examples, physical=True, samples=samples, seed=seed, check=None
    )


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
                model, examples, physical=False, samples=samples, seed=seed, check=check
            ),
            physical_baseline=_evaluate_arm(
                model, examples, physical=True, samples=samples, seed=seed, check=check
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
