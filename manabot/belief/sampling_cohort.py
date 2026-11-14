"""Offline uncertainty over saved sampler reports, conditional on one producer.

CohortSpec declares every fit attempt and one fixed evaluation panel. Analysis
never samples hands or loads models. Fits are the independent training unit;
games (all decisions/viewers together) are the conditional evaluation unit.
Separate percentile bootstraps vary one axis at a time, never individual rows.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import platform
from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from manabot.belief.sampling_data import Split
from manabot.belief.sampling_fit import SamplerArmMetrics
from manabot.reports.belief import SamplerQualityReport, SplitMeasurement


class Declaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FitAttempt(Declaration):
    """Run/config identities and training seed are declarations from fit receipts.

    Repeated evaluations belong in separate panels, not additional fit attempts.
    Failed attempts retain a reason and cannot be replaced by a successful retry.
    """

    attempt_id: str = Field(min_length=1)
    treatment: str = Field(min_length=1)
    training_seed: int
    fit_receipt_identity: str = Field(min_length=1)
    configuration_identity: str = Field(min_length=1)
    training_dataset_identity: str = Field(min_length=1)
    status: Literal["complete", "failed", "missing"]
    report: Path | None = None
    report_sha256: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def check_status(self) -> FitAttempt:
        if self.status == "complete":
            if self.report is None or not self.report_sha256 or self.reason:
                raise ValueError(
                    "complete attempt requires report/digest and no failure"
                )
        elif (
            not self.reason or self.report is not None or self.report_sha256 is not None
        ):
            raise ValueError("failed/missing attempt requires reason and no report")
        return self


class Contrast(Declaration):
    left: str
    right: str


class EvaluationGroup(Declaration):
    """Predeclared dependent games, including legs with no retained rows."""

    group_id: str = Field(min_length=1)
    game_ids: tuple[str, ...] = Field(min_length=1)


class CohortSpec(Declaration):
    format: Literal["manabot.sampler-cohort/v1"] = "manabot.sampler-cohort/v1"
    description: str = Field(min_length=1)
    producer_policy_identity: str = Field(min_length=1)
    evaluation_policy_identity: str = Field(min_length=1)
    evaluation_dataset_identity: str = Field(min_length=1)
    world_identity: str = Field(min_length=1)
    schema_identity: str = Field(min_length=1)
    evaluator_identity: str = Field(min_length=1)
    split: Split = "test"
    samples_per_example: int = Field(gt=0)
    sampling_seed: int
    bootstrap_seed: int = 0
    bootstrap_replicates: int = Field(default=2000, ge=100, le=100000)
    attempts: tuple[FitAttempt, ...] = Field(min_length=1)
    contrasts: tuple[Contrast, ...] = ()
    groups: tuple[EvaluationGroup, ...] = ()
    panel_game_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def check_design(self) -> CohortSpec:
        if len(set(self.panel_game_ids)) != len(self.panel_game_ids):
            raise ValueError("panel game identities must be unique")
        grouped_games = [game for group in self.groups for game in group.game_ids]
        if len({g.group_id for g in self.groups}) != len(self.groups) or len(
            set(grouped_games)
        ) != len(grouped_games):
            raise ValueError("evaluation groups and game membership must be unique")
        if len({a.attempt_id for a in self.attempts}) != len(self.attempts):
            raise ValueError("attempt IDs must be unique")
        if len({(a.treatment, a.training_seed) for a in self.attempts}) != len(
            self.attempts
        ):
            raise ValueError("repeated training seeds are not independent fits")
        complete = [a for a in self.attempts if a.status == "complete"]
        if len({a.fit_receipt_identity for a in complete}) != len(complete):
            raise ValueError("one fit receipt cannot count as multiple fits")
        if len({a.training_dataset_identity for a in self.attempts}) != 1:
            raise ValueError("fit-seed analysis requires one fixed training dataset")
        treatments = {a.treatment for a in self.attempts}
        for treatment in treatments:
            if (
                len(
                    {
                        a.configuration_identity
                        for a in self.attempts
                        if a.treatment == treatment
                    }
                )
                != 1
            ):
                raise ValueError("one treatment must bind one fit configuration")
        for contrast in self.contrasts:
            if (
                contrast.left == contrast.right
                or not {contrast.left, contrast.right} <= treatments
            ):
                raise ValueError("contrast needs two declared treatments")
            left = {
                a.training_seed for a in self.attempts if a.treatment == contrast.left
            }
            right = {
                a.training_seed for a in self.attempts if a.treatment == contrast.right
            }
            if left != right:
                raise ValueError(
                    "paired treatments require identical declared fit seeds"
                )
        return self


Metric = Literal[
    "joint_nll",
    "inclusion_brier",
    "inclusion_ece",
    "conjunction_brier",
    "violation_rate",
]
METRICS: tuple[Metric, ...] = (
    "joint_nll",
    "inclusion_brier",
    "inclusion_ece",
    "conjunction_brier",
    "violation_rate",
)


@dataclass(frozen=True, slots=True)
class Interval:
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class Estimate:
    comparison: str
    metric: Metric
    fit_seeds: tuple[int, ...]
    per_fit: tuple[float, ...]
    mean: float | None
    fit_interval: Interval | None
    game_interval: Interval | None
    unavailable: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CohortReport:
    format: str
    manifest: CohortSpec
    manifest_identity: str
    analysis_identity: str
    numpy_version: str
    python_version: str
    complete: bool
    estimates: tuple[Estimate, ...]
    group_coverage: tuple[GroupCoverage, ...]


@dataclass(frozen=True, slots=True)
class GroupCoverage:
    group_id: str
    declared_games: tuple[str, ...]
    retained_games: tuple[str, ...]
    missing_games: tuple[str, ...]


def _value(arm: SamplerArmMetrics, metric: Metric) -> float | None:
    if metric == "violation_rate":
        return arm.support_violations / arm.sampled_hands
    values: dict[Metric, float | None] = {
        "joint_nll": arm.joint_nll,
        "inclusion_brier": arm.inclusion_brier,
        "inclusion_ece": arm.inclusion_ece,
        "conjunction_brier": arm.conjunction_brier,
    }
    return values[metric]


def _admit_report(
    spec: CohortSpec, attempt: FitAttempt, root: Path
) -> SamplerQualityReport:
    assert attempt.report is not None
    data = (root / attempt.report).read_bytes()
    if hashlib.sha256(data).hexdigest() != attempt.report_sha256:
        raise ValueError(f"{attempt.attempt_id}: report byte identity mismatch")
    report = TypeAdapter(SamplerQualityReport).validate_json(data)
    expected = {
        "format": "manabot.sampler-quality/v2",
        "evaluation_dataset_identity": spec.evaluation_dataset_identity,
        "evaluation_policy_identity": spec.evaluation_policy_identity,
        "training_dataset_identity": attempt.training_dataset_identity,
        "training_policy_identity": spec.producer_policy_identity,
        "world_identity": spec.world_identity,
        "schema_identity": spec.schema_identity,
        "evaluator_identity": spec.evaluator_identity,
        "seed": spec.sampling_seed,
        "training_seed": attempt.training_seed,
    }
    for name, value in expected.items():
        if getattr(report, name) != value:
            raise ValueError(f"{attempt.attempt_id}: {name} mismatch")
    if report.checkpoint_identity is None:
        raise ValueError("cohort requires learned sampler reports")
    panels = [
        m
        for m in report.measurements
        if m.split == spec.split and m.samples_per_example == spec.samples_per_example
    ]
    if len(panels) != 1 or not panels[0].games:
        raise ValueError("one whole-game panel required")
    panel = panels[0]
    if tuple(g.game_id for g in panel.games) != panel.game_ids or len(
        set(panel.game_ids)
    ) != len(panel.games):
        raise ValueError("game membership mismatch")
    if sum(g.examples for g in panel.games) != panel.examples:
        raise ValueError("game example count mismatch")
    for game in panel.games:
        if game.examples < 1 or game.learned is None:
            raise ValueError("complete learned game evidence required")
        for arm in (game.learned, game.physical_prior):
            if arm.sampled_hands != game.examples * spec.samples_per_example:
                raise ValueError("sample count mismatch")
            if not 0 <= arm.support_violations <= arm.sampled_hands:
                raise ValueError("invalid violation count")
            values = [_value(arm, metric) for metric in METRICS]
            if any(v is not None and (not np.isfinite(v) or v < 0) for v in values):
                raise ValueError("non-finite or negative metric")
    return report


def _estimate(
    label: str,
    metric: Metric,
    seeds: tuple[int, ...],
    rows: list[list[float]],
    *,
    complete: bool,
    spec: CohortSpec,
    groups: tuple[tuple[int, ...], ...],
) -> Estimate:
    if not rows or not rows[0]:
        return Estimate(
            label, metric, (), (), None, None, None, ("no supported complete fits",)
        )
    values = np.asarray(rows, dtype=np.float64)  # [independent fits, whole games]
    per_fit = tuple(float(v) for v in values.mean(axis=1))
    reasons: list[str] = []
    if not complete:
        return Estimate(
            label,
            metric,
            seeds,
            per_fit,
            None,
            None,
            None,
            ("declared cohort incomplete; no complete-case cohort estimate",),
        )
    # Separate axes avoid counting shared games again for each fit. Game draws
    # use identical indexes for every fit and both sides of each contrast.
    rng = np.random.default_rng(spec.bootstrap_seed)

    def interval(axis: int) -> Interval | None:
        size = values.shape[0] if axis == 0 else len(groups)
        if size < 2:
            reasons.append(
                "independent-fit uncertainty unavailable"
                if axis == 0
                else "game uncertainty unavailable: fewer than two retained groups"
            )
            return None
        means = values.mean(axis=1 - axis)
        draws = np.empty(spec.bootstrap_replicates)
        sums = (
            np.asarray([means[list(group)].sum() for group in groups])
            if axis == 1
            else None
        )
        counts = np.asarray([len(group) for group in groups])
        for i in range(spec.bootstrap_replicates):
            indexes = rng.integers(size, size=size)
            # Keep every available leg/viewer together. Weight by retained games,
            # preserving the equal-game estimand when groups have missing legs.
            draws[i] = (
                means[indexes].mean()
                if sums is None
                else sums[indexes].sum() / counts[indexes].sum()
            )
        low, high = np.quantile(draws, [0.025, 0.975])
        return Interval(float(low), float(high))

    fit_ci, game_ci = interval(0), interval(1)
    return Estimate(
        label,
        metric,
        seeds,
        per_fit,
        float(values.mean()),
        fit_ci,
        game_ci,
        tuple(reasons),
    )


def report_sampler_cohort(spec: CohortSpec, *, root: Path = Path(".")) -> CohortReport:
    """Analyze hashed saved reports; identity errors fail rather than drop a fit.

    All fits share one producer and evaluation dataset. Unequal game membership
    fails; unequal decision counts are allowed and games still have equal weight.
    Missing/failed declared fits suppress cohort estimates, retaining per-fit means.
    ECE here is mean game ECE; pooled forecast ECE stays in each saved report.
    """
    panels: dict[str, SplitMeasurement] = {}
    checkpoint_ids: set[str] = set()
    membership: tuple[tuple[str, int, int, int], ...] | None = None
    runtime: tuple[str, str, str, int] | None = None
    for attempt in spec.attempts:
        if attempt.status != "complete":
            continue
        report = _admit_report(spec, attempt, root)
        panel = next(
            m
            for m in report.measurements
            if m.split == spec.split
            and m.samples_per_example == spec.samples_per_example
        )
        if spec.panel_game_ids:
            selected = set(spec.panel_game_ids)
            games = tuple(game for game in panel.games if game.game_id in selected)
            panel = replace(
                panel,
                games=games,
                game_ids=tuple(g.game_id for g in games),
                examples=sum(g.examples for g in games),
            )
        assert report.checkpoint_identity is not None
        if report.checkpoint_identity in checkpoint_ids:
            raise ValueError("repeated checkpoint bytes are not independent fits")
        checkpoint_ids.add(report.checkpoint_identity)
        current_runtime = (
            report.torch_version,
            report.python_version,
            report.device,
            report.threads,
        )
        if runtime is not None and runtime != current_runtime:
            raise ValueError("report runtime mismatch")
        runtime = current_runtime
        current = tuple(
            (g.game_id, g.seed, g.assignment, g.examples) for g in panel.games
        )
        if membership is not None and membership != current:
            raise ValueError("paired dataset game membership mismatch")
        membership = current
        panels[attempt.attempt_id] = panel
    complete = len(panels) == len(spec.attempts)
    game_ids = tuple(row[0] for row in membership) if membership is not None else ()
    if (
        membership is not None
        and not spec.groups
        and len({row[1] for row in membership}) != len(membership)
    ):
        raise ValueError("shared deal seeds require declared evaluation groups")
    declared_groups = spec.groups or tuple(
        EvaluationGroup(group_id=game, game_ids=(game,)) for game in game_ids
    )
    if set(game_ids) - {game for group in declared_groups for game in group.game_ids}:
        raise ValueError("retained game is outside declared evaluation groups")
    seed_groups: dict[int, str] = {}
    for group in declared_groups:
        for game_id, seed, _, _ in membership or ():
            if (
                game_id in group.game_ids
                and seed_groups.setdefault(seed, group.group_id) != group.group_id
            ):
                raise ValueError("shared deal seed crosses evaluation groups")
    coverage = tuple(
        GroupCoverage(
            group.group_id,
            group.game_ids,
            tuple(game for game in group.game_ids if game in game_ids),
            tuple(game for game in group.game_ids if game not in game_ids),
        )
        for group in declared_groups
    )
    groups = tuple(
        tuple(i for i, game in enumerate(game_ids) if game in group.game_ids)
        for group in declared_groups
        if any(game in game_ids for game in group.game_ids)
    )
    treatments = sorted({a.treatment for a in spec.attempts})
    estimates: list[Estimate] = []
    comparisons = [
        (t, None, mode)
        for t in treatments
        for mode in ("learned", "physical_prior", "difference")
    ]
    comparisons.extend((c.left, c.right, "difference") for c in spec.contrasts)
    for left, right, mode in comparisons:
        attempts = sorted(
            (a for a in spec.attempts if a.treatment == left),
            key=lambda a: a.training_seed,
        )
        partners = {a.training_seed: a for a in spec.attempts if a.treatment == right}
        for metric in METRICS:
            rows: list[list[float]] = []
            seeds: list[int] = []
            unsupported = False
            for attempt in attempts:
                panel = panels.get(attempt.attempt_id)
                other = (
                    panels.get(partners[attempt.training_seed].attempt_id)
                    if right
                    else None
                )
                if panel is None or (right and other is None):
                    continue
                row: list[float] = []
                for i, game in enumerate(panel.games):
                    assert game.learned is not None
                    baseline = (
                        game.physical_prior if other is None else other.games[i].learned
                    )
                    assert baseline is not None
                    a, b = _value(game.learned, metric), _value(baseline, metric)
                    value = (
                        a
                        if mode == "learned"
                        else b
                        if mode == "physical_prior"
                        else None
                        if a is None or b is None
                        else a - b
                    )
                    if value is None:
                        unsupported = True
                        break
                    row.append(value)
                if len(row) == len(panel.games):
                    rows.append(row)
                    seeds.append(attempt.training_seed)
            label = (
                f"{left} minus {right or 'physical_prior'}"
                if mode == "difference"
                else f"{left} {mode}"
            )
            estimate = _estimate(
                label,
                metric,
                tuple(seeds),
                rows,
                complete=complete and not unsupported,
                spec=spec,
                groups=groups,
            )
            if unsupported:
                estimate = Estimate(
                    label,
                    metric,
                    tuple(seeds),
                    estimate.per_fit,
                    None,
                    None,
                    None,
                    ("metric unsupported in declared cohort",),
                )
            estimates.append(estimate)
    manifest_json = spec.model_dump_json()
    return CohortReport(
        "manabot.sampler-cohort-report/v1",
        spec,
        hashlib.sha256(manifest_json.encode()).hexdigest(),
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        str(np.__version__),
        platform.python_version(),
        complete,
        tuple(estimates),
        coverage,
    )


def render_cohort_report(report: CohortReport) -> str:
    """Small reviewable Markdown companion; JSON retains declarations and fit effects."""
    lines = [
        "# Saved sampler cohort",
        "",
        report.manifest.description,
        "",
        f"Complete: {report.complete}. Declared attempts: {len(report.manifest.attempts)}.",
        "",
        "Equal game and fit weights; differences are left minus right (lower is better).",
        "95% percentile intervals vary fit seeds or declared game groups separately; neither is a joint interval.",
        "Conditional on this producer and saved population; small fit cohorts are exploratory.",
        "",
        "| Comparison | Metric | Mean | Fit interval | Game interval | Unavailable |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    def display(value: Interval | None) -> str:
        return (
            "unavailable" if value is None else f"[{value.low:.6g}, {value.high:.6g}]"
        )

    for estimate in report.estimates:
        mean = "unavailable" if estimate.mean is None else f"{estimate.mean:.6g}"
        lines.append(
            f"| {estimate.comparison} | {estimate.metric} | {mean} | {display(estimate.fit_interval)} | {display(estimate.game_interval)} | {'; '.join(estimate.unavailable)} |"
        )
    lines.extend(["", "## Declared attempts", ""])
    missing = sum(len(group.missing_games) for group in report.group_coverage)
    lines.append(
        f"Evaluation groups: {len(report.group_coverage)}; missing legs: {missing}. Intervals condition on retained games."
    )
    lines.extend(
        f"- {a.attempt_id}: {a.status}; training seed {a.training_seed}; {a.reason or 'saved report admitted'}"
        for a in report.manifest.attempts
    )
    lines.extend(
        [
            "",
            "Sampling costs remain in the hashed input reports; regeneration performs no sampling.",
            "No calibration, transfer or strength acceptance follows from this report.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="new JSON path; also writes a .md companion",
    )
    args = parser.parse_args(argv)
    markdown = args.out.with_suffix(".md")
    if args.out == markdown or args.out.exists() or markdown.exists():
        raise FileExistsError("choose new JSON and Markdown output paths")
    spec = CohortSpec.model_validate_json(args.manifest.read_bytes())
    report = report_sampler_cohort(spec, root=args.manifest.parent)
    payload = {**asdict(report), "manifest": spec.model_dump(mode="json")}
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(encoded)
    with markdown.open("x") as output:
        output.write(render_cohort_report(report))


if __name__ == "__main__":
    main()
