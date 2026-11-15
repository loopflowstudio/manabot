"""Host sampler metrics and immutable artifacts; numerical fitting is injected."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Protocol

from manabot.belief.sampling_data import SamplerDataset, SamplerExample
from manabot.sim.search.sampling import LearnedDistribution


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
    model: LearnedDistribution
    metrics: SamplerFitMetrics
    dataset_identity: str
    policy_identity: str
    world_identity: str

    @property
    def optimizer_exposures(self) -> int:
        return self.metrics.optimizer_exposures


def calibration_error(bins: Sequence[CalibrationBin]) -> float:
    """Recompute pooled ECE; averaging subgroup ECE is a different estimand."""
    total = sum(bin.count for bin in bins)
    if total == 0:
        raise ValueError("calibration needs forecasts")
    return sum(abs(bin.probability_sum - bin.truth_sum) for bin in bins) / total


class BeliefTrainer(Protocol):
    def encode_sampler(
        self, result: SamplerFitResult, dataset: SamplerDataset
    ) -> bytes: ...

    def evaluate_physical_sampler(
        self, examples: Sequence[SamplerExample], *, samples: int = 32, seed: int = 0
    ) -> SamplerArmMetrics: ...
    def evaluate_sampler(
        self,
        model: LearnedDistribution,
        examples: Sequence[SamplerExample],
        *,
        samples: int = 32,
        seed: int = 0,
        check: Callable[[], None] | None = None,
    ) -> SamplerEvaluation: ...
    def fit_belief_sampler(
        self,
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
    ) -> SamplerFitResult: ...


class SamplerEncoder(Protocol):
    def encode_sampler(
        self, result: SamplerFitResult, dataset: SamplerDataset
    ) -> bytes: ...


def evaluate_physical_sampler(
    examples: Sequence[SamplerExample],
    *,
    backend: BeliefTrainer,
    samples: int = 32,
    seed: int = 0,
) -> SamplerArmMetrics:
    """Score saved labels against the selected physical-deal implementation."""
    return backend.evaluate_physical_sampler(examples, samples=samples, seed=seed)


def evaluate_sampler(
    model: LearnedDistribution,
    examples: Sequence[SamplerExample],
    *,
    backend: BeliefTrainer,
    samples: int = 32,
    seed: int = 0,
    check: Callable[[], None] | None = None,
) -> SamplerEvaluation:
    """Score held-out labels with the selected backend, without fitting."""
    return backend.evaluate_sampler(
        model, examples, samples=samples, seed=seed, check=check
    )


def fit_belief_sampler(
    dataset: SamplerDataset,
    *,
    backend: BeliefTrainer,
    steps: int,
    batch_size: int = 32,
    hidden_size: int = 64,
    learning_rate: float = 0.001,
    history_dropout: float = 0.0,
    evaluation_samples: int = 32,
    seed: int = 0,
    check: Callable[[], None] | None = None,
) -> SamplerFitResult:
    """Fit training games through the selected numerical implementation."""
    return backend.fit_belief_sampler(
        dataset,
        steps=steps,
        batch_size=batch_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        history_dropout=history_dropout,
        evaluation_samples=evaluation_samples,
        seed=seed,
        check=check,
    )


def save_belief_sampler(
    path: Path,
    result: SamplerFitResult,
    dataset: SamplerDataset,
    *,
    codec: SamplerEncoder,
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
    data = codec.encode_sampler(result, dataset)
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
