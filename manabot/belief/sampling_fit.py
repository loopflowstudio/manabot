"""Fit, measure and persist a sampler after its behavior dataset is frozen.

Private hand labels only enter the proper joint log score and held-out metrics.
``fit_belief_sampler`` fits train games; validation/test games never update the
model. ``evaluate_sampler`` also accepts separately collected foreign-policy
histories without treating that cohort as self-play calibration. Checkpoints
bind exact dataset, behavior policy, schema and world identities, and are
immutable files. These diagnostics do not establish search or playing strength.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import math
import os
from pathlib import Path
import tempfile
import time
import tracemalloc
from typing import Any, Callable, Sequence

import torch

from manabot.belief.learning import _deterministic_torch_cpu
from manabot.belief.range import BeliefState
from manabot.belief.sampling import (
    AutoregressiveBeliefSampler,
    SamplerInput,
    SamplerSchema,
    physical_deal_log_prob,
    sample_physical_deal,
)
from manabot.belief.sampling_data import SamplerDataset, SamplerExample


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """Sufficient statistics for one fixed inclusion-probability bin."""

    count: int
    probability_sum: float
    truth_sum: float


@dataclass(frozen=True, slots=True)
class SamplerArmMetrics:
    """Per-decision means; calibration pools card-presence forecasts, not games.

    Brier and ECE use Monte Carlo inclusion probabilities. Conjunction Brier
    averages adjacent vocabulary pairs, retaining a fixed O(C) query panel.
    Python peak bytes excludes native Torch allocations; tensor bytes records
    the largest returned count tensor separately. Neither is process peak RSS.
    """

    joint_nll: float
    inclusion_brier: float
    inclusion_ece: float
    conjunction_brier: float | None
    support_violations: int
    sampled_hands: int
    sampling_seconds: float
    peak_python_bytes: int
    largest_sample_tensor_bytes: int
    calibration_bins: tuple[CalibrationBin, ...] = ()


@dataclass(frozen=True, slots=True)
class SamplerEvaluation:
    examples: int
    learned: SamplerArmMetrics
    physical_baseline: SamplerArmMetrics


@dataclass(frozen=True, slots=True)
class SamplerFitMetrics:
    train: SamplerEvaluation
    validation: SamplerEvaluation
    test: SamplerEvaluation
    optimizer_exposures: int
    optimizer_steps: int
    history_dropout: float
    seed: int
    fit_seconds: float
    parameter_bytes: int
    exact_posterior_comparison: str = (
        "unavailable: requires an externally computed exact posterior"
    )
    foreign_policy_evaluation: str = (
        "unavailable: requires a separately identified held-out policy cohort"
    )
    matched_time_search_strength: str = "unavailable: requires a frozen arena protocol"


@dataclass(frozen=True, slots=True)
class SamplerFitResult:
    model: AutoregressiveBeliefSampler
    metrics: SamplerFitMetrics
    dataset_identity: str
    policy_identity: str
    world_identity: str

    @property
    def optimizer_exposures(self) -> int:
        return self.metrics.optimizer_exposures


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


def calibration_error(bins: Sequence[CalibrationBin]) -> float:
    """Recompute pooled ECE; averaging subgroup ECE is a different estimand."""
    total = sum(bin.count for bin in bins)
    if total == 0:
        raise ValueError("calibration needs forecasts")
    return sum(abs(bin.probability_sum - bin.truth_sum) for bin in bins) / total


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
    with _deterministic_torch_cpu(), torch.random.fork_rng(devices=[]):
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


def save_belief_sampler(
    path: Path, result: SamplerFitResult, dataset: SamplerDataset
) -> str:
    """Create one immutable checkpoint; return SHA-256 of its exact bytes."""
    if (result.dataset_identity, result.policy_identity, result.world_identity) != (
        dataset.identity,
        dataset.policy_identity,
        dataset.world_identity,
    ):
        raise ValueError(
            "sampler fit is bound to a different dataset or policy/world identity"
        )
    if result.model.schema != dataset.schema:
        raise ValueError("sampler and dataset schemas differ")
    payload = {
        "format": "manabot.autoregressive-belief-sampler/v1",
        "schema": asdict(dataset.schema),
        "schema_identity": dataset.schema.identity,
        "dataset_identity": dataset.identity,
        "policy_identity": dataset.policy_identity,
        "world_identity": dataset.world_identity,
        "hidden_size": result.model.hidden_size,
        "history_dropout": result.model.history_dropout,
        "state_dict": result.model.state_dict(),
        "metrics": asdict(result.metrics),
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    data = stream.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".sampler-", delete=False
    ) as output:
        temporary = Path(output.name)
        try:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
            os.link(temporary, path)  # Atomic publication, refusing an existing target.
        finally:
            temporary.unlink()
    return hashlib.sha256(data).hexdigest()


def load_belief_sampler(
    path: Path,
    *,
    expected_dataset_identity: str,
    expected_policy_identity: str,
    expected_schema_identity: str,
    expected_world_identity: str,
    expected_checkpoint_identity: str,
) -> AutoregressiveBeliefSampler:
    """Reject mismatched bytes or semantic bindings before admitting parameters."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_checkpoint_identity:
        raise ValueError("sampler checkpoint byte identity mismatch")
    # Torch's weights-only decoder is an untyped serialization boundary.
    payload: Any = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "manabot.autoregressive-belief-sampler/v1"
    ):
        raise ValueError("unknown sampler checkpoint format")
    expected = {
        "dataset_identity": expected_dataset_identity,
        "policy_identity": expected_policy_identity,
        "schema_identity": expected_schema_identity,
        "world_identity": expected_world_identity,
    }
    for key, value in expected.items():
        if not value or payload.get(key) != value:
            raise ValueError(f"sampler checkpoint {key} mismatch")
    schema_payload = payload["schema"]
    if not isinstance(schema_payload, dict):
        raise ValueError("invalid sampler checkpoint schema")
    schema = SamplerSchema(**schema_payload)
    if schema.identity != expected_schema_identity:
        raise ValueError("sampler checkpoint schema contents mismatch")
    model = AutoregressiveBeliefSampler(
        schema, payload["hidden_size"], payload["history_dropout"]
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


@dataclass(frozen=True, slots=True)
class ExactReferenceMetrics:
    """Expected joint log scores under an externally computed exact posterior."""

    reference_identity: str
    learned_cross_entropy: float
    physical_cross_entropy: float
    learned_kl: float
    physical_kl: float
    learned_reference_support_mass: float


@torch.no_grad()
def compare_exact_reference(
    model: AutoregressiveBeliefSampler,
    inputs: SamplerInput,
    reference: BeliefState,
) -> ExactReferenceMetrics:
    """Consume the existing exact tracker's result on a tractable root.

    The caller must provide the posterior at this same viewer history; this
    adapter never recomputes action likelihoods or runs its own Bayes tracker.
    It rejects incompatible physical constraints and missing vocabulary rows.
    """
    names = model.schema.card_names
    pool = dict(reference.space.pool)
    known = dict(reference.space.known_hand)
    if (
        any(name not in names for name in (*pool, *known))
        or tuple(pool.get(name, 0) for name in names) != inputs.pool_counts
        or tuple(known.get(name, 0) for name in names) != inputs.known_minima
        or reference.space.hand_size != inputs.hand_size
    ):
        raise ValueError("exact reference physical domain differs from sampler input")
    if any(
        name not in names for world in reference.space.worlds for name, _ in world.hand
    ):
        raise ValueError(
            "exact reference contains cards outside the sampler vocabulary"
        )
    rows = [
        tuple(dict(world.hand).get(name, 0) for name in names)
        for world in reference.space.worlds
    ]
    targets = torch.tensor(rows, dtype=torch.int64)
    batch = [inputs] * len(rows)
    physical = physical_deal_log_prob(batch, targets)
    prior_training = model.training
    model.eval()
    try:
        learned = model.log_prob(batch, targets)
    finally:
        model.train(prior_training)
    probabilities = torch.tensor(reference.probabilities, dtype=torch.float64)
    entropy = reference.entropy
    learned_ce = float(-(probabilities * learned).sum())
    physical_ce = float(-(probabilities * physical).sum())
    return ExactReferenceMetrics(
        reference_identity=reference.digest,
        learned_cross_entropy=learned_ce,
        physical_cross_entropy=physical_ce,
        learned_kl=learned_ce - entropy,
        physical_kl=physical_ce - entropy,
        learned_reference_support_mass=float(learned.exp().sum()),
    )


@dataclass(frozen=True, slots=True)
class SamplerCohortEvaluation:
    """Named foreign/adversarial evidence, separate from self-play calibration."""

    training_policy_identity: str
    evaluation_policy_identity: str
    cohort_identity: str
    distribution_shift: bool
    adversarial_selection: bool
    metrics: SamplerEvaluation


def evaluate_sampler_cohort(
    result: SamplerFitResult,
    examples: Sequence[SamplerExample],
    *,
    evaluation_policy_identity: str,
    cohort_identity: str,
    world_identity: str,
    adversarial_selection: bool = False,
    samples: int = 32,
    seed: int = 0,
    check: Callable[[], None] | None = None,
) -> SamplerCohortEvaluation:
    """Record opponent-assumption mismatch while retaining the safe baseline.

    Cohort selection and label provenance belong to the collection protocol;
    this method cannot turn synthetic or human-selected rows into human-play
    evidence. World changes require a newly pinned training/evaluation protocol.
    """
    if not evaluation_policy_identity or not cohort_identity:
        raise ValueError("cohort and evaluation policy identities are required")
    if world_identity != result.world_identity:
        raise ValueError("evaluation cohort world differs from fitted sampler")
    return SamplerCohortEvaluation(
        training_policy_identity=result.policy_identity,
        evaluation_policy_identity=evaluation_policy_identity,
        cohort_identity=cohort_identity,
        distribution_shift=evaluation_policy_identity != result.policy_identity
        or adversarial_selection,
        adversarial_selection=adversarial_selection,
        metrics=evaluate_sampler(
            result.model, examples, samples=samples, seed=seed, check=check
        ),
    )
