"""Read-only questions for normally trained baseline checkpoints.

Frozen native choices describe policy sensitivity; paired arena outcomes supply
learning evidence. Neither changes the learner or labels an arbitrary action as
correct. Game uncertainty resamples complete deal blocks; cohort uncertainty also
resamples training seeds. Incomplete evidence is never a passing learning result.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import time
from typing import Literal

import numpy as np
from pydantic import Field
import torch

from manabot.arena.match import selected_decks
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.arena.monitor import MonitorResult
from manabot.env.observation import ActionEnum
from manabot.experiment.evaluation_queue import Attempt, checkpoints
from manabot.experiment.execution import ExperimentRun
from manabot.infra.files import atomic_json
from manabot.model.checkpoint import load_checkpoint
from manabot.model.world import validate_agent_setup
from manabot.sim.search.flat_mc import ACTION_SPACE_KIND_NAMES
from manabot.training.records import TrainingRun
from manabot.training.regime import Strict, TrainSelfPlay
from manabot.verify.action_cases import reorder_objects, reorder_offers
from manabot.verify.choice_corpus import PACK, Case, collect_choices, input_collisions
import managym

Verdict = Literal["learning", "regression", "inconclusive", "error"]


class PairedDeal(Strict):
    deal: int
    initial: float = Field(ge=0, le=1)
    final: float = Field(ge=0, le=1)
    # Four seat/deck legs are one block, never four independent samples.
    legs: list[float] = Field(min_length=4, max_length=4)


class LearningEvidence(Strict):
    verdict: Verdict
    seeds: int
    deals: int
    gain: float | None = None
    lower: float | None = None
    upper: float | None = None
    per_seed: list[float] = Field(default_factory=list)
    reason: str


def learning_evidence(cohorts: list[list[PairedDeal]]) -> LearningEvidence:
    """Paired hierarchical bootstrap with a frozen 5-point practical threshold.

    Same evaluation deals are shared by seeds. Resample deal indexes jointly,
    and training seeds as independent units; never bootstrap individual actions
    or treat four deck/seat legs as independent. A single seed describes that
    trained instance, not the reliability of a training method.
    """
    if not cohorts or any(not rows for rows in cohorts):
        return LearningEvidence(
            verdict="error",
            seeds=len(cohorts),
            deals=0,
            reason="missing paired outcome evidence",
        )
    ordered = [sorted(rows, key=lambda row: row.deal) for rows in cohorts]
    deals = [row.deal for row in ordered[0]]
    if len(set(deals)) != len(deals) or any(
        [row.deal for row in rows] != deals for rows in ordered
    ):
        raise ValueError("seed cohorts must contain the same unique deal blocks")
    gains = np.asarray([[row.final - row.initial for row in rows] for rows in ordered])
    if not np.isfinite(gains).all():
        raise ValueError("nonfinite paired gains")
    rng = np.random.default_rng(150_197)
    samples = np.empty(10_000)
    for i in range(len(samples)):
        seeds = rng.integers(len(cohorts), size=len(cohorts))
        blocks = rng.integers(len(deals), size=len(deals))
        samples[i] = gains[np.ix_(seeds, blocks)].mean()
    lower, upper = np.quantile(samples, [0.025, 0.975])
    gain = float(gains.mean())
    verdict: Verdict = "inconclusive"
    if len(deals) >= 24 and lower > 0 and gain >= 0.05:
        verdict = "learning"
    elif len(deals) >= 24 and upper < 0 and gain <= -0.05:
        verdict = "regression"
    return LearningEvidence(
        verdict=verdict,
        seeds=len(cohorts),
        deals=len(deals),
        gain=gain,
        lower=float(lower),
        upper=float(upper),
        per_seed=gains.mean(axis=1).tolist(),
        reason=(
            "paired outcome improvement against random; conditional on these seeds/deals"
            if verdict == "learning"
            else "paired outcome deterioration against random"
            if verdict == "regression"
            else "insufficient evidence of a practically meaningful outcome change"
        ),
    )


class ChoiceScore(Strict):
    case: str
    family: int
    log_odds: float
    value: float


class PolicySnapshot(Strict):
    updates: int
    checkpoint_sha256: str
    choices: list[ChoiceScore]
    maximum_invariance_error: float


@torch.no_grad()
def inspect_policy(path: Path, cases: list[Case], updates: int) -> PolicySnapshot:
    """Use the ordinary world-bound loader and exact setup; never refit weights."""
    if not cases:
        raise ValueError("empty frozen choice corpus")
    if len(cases) > 32:
        chunks = [
            inspect_policy(path, cases[i : i + 32], updates)
            for i in range(0, len(cases), 32)
        ]
        return PolicySnapshot(
            updates=updates,
            checkpoint_sha256=chunks[0].checkpoint_sha256,
            choices=[choice for chunk in chunks for choice in chunk.choices],
            maximum_invariance_error=max(
                chunk.maximum_invariance_error for chunk in chunks
            ),
        )
    agent, _ = load_checkpoint(str(path))
    configs = [
        managym.authored_deck_setup(PACK, key) for key in ("ur_lessons", "gw_allies")
    ]
    validate_agent_setup(agent, configs)
    batch = {key: torch.cat([case.obs[key] for case in cases]) for key in cases[0].obs}
    logits, values = agent(batch)
    probabilities = logits.softmax(-1)
    if not torch.isfinite(probabilities).all() or not torch.isfinite(values).all():
        raise ValueError("nonfinite frozen checkpoint output")
    restored, _ = load_checkpoint(str(path))
    loaded_logits, loaded_values = restored(batch)
    errors = [
        float((loaded_logits.softmax(-1) - probabilities).abs().max()),
        float((loaded_values - values).abs().max()),
    ]
    scores = []
    for row, case in enumerate(cases):
        obs = case.obs
        single, value = agent(obs)
        p = single.softmax(-1)
        count = int(obs["actions_valid"].sum())
        order = torch.arange(count - 1, -1, -1)
        offers, offer_value = agent(reorder_offers(obs, order))
        objects, object_value = agent(reorder_objects(obs))
        errors.extend(
            [
                float((p[0] - probabilities[row]).abs().max()),
                float((value[0] - values[row]).abs().max()),
                float((offers.softmax(-1)[0, :count] - p[0, order]).abs().max()),
                float((objects.softmax(-1) - p).abs().max()),
                float((offer_value - value).abs().max()),
                float((object_value - value).abs().max()),
                float(p[obs["actions_valid"] == 0].sum()),
            ]
        )
        # Log-sum-exp avoids underflow and gives identical public copies shared
        # probability mass. Its sign has no prescribed strategic interpretation.
        left, right = case.groups
        log_odds = float(single[0, left].logsumexp(0) - single[0, right].logsumexp(0))
        scores.append(
            ChoiceScore(
                case=case.name,
                family=case.family,
                log_odds=log_odds,
                value=float(value[0]),
            )
        )
    maximum = max(errors)
    if maximum > 1e-4:
        raise ValueError(f"frozen-policy invariance error {maximum} exceeds 1e-4")
    return PolicySnapshot(
        updates=updates,
        checkpoint_sha256=file_sha256(path),
        choices=scores,
        maximum_invariance_error=maximum,
    )


class FamilyChange(Strict):
    family: int
    cases: int
    mean_absolute_log_odds_change: float
    std_log_odds_change: float
    mean_absolute_value_change: float


def family_changes(
    initial: PolicySnapshot, final: PolicySnapshot
) -> list[FamilyChange]:
    if [(r.case, r.family) for r in initial.choices] != [
        (r.case, r.family) for r in final.choices
    ]:
        raise ValueError("choice corpus changed between checkpoints")
    groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for before, after in zip(initial.choices, final.choices, strict=True):
        groups[before.family].append(
            (after.log_odds - before.log_odds, after.value - before.value)
        )
    return [
        FamilyChange(
            family=family,
            cases=len(rows),
            mean_absolute_log_odds_change=float(np.abs(np.asarray(rows)[:, 0]).mean()),
            std_log_odds_change=float(np.asarray(rows)[:, 0].std()),
            mean_absolute_value_change=float(np.abs(np.asarray(rows)[:, 1]).mean()),
        )
        for family, rows in sorted(groups.items())
    ]


class ActionExposure(Strict):
    action_type: int
    collected: int = 0
    retained: int = 0


class BaselineReport(Strict):
    seed: int
    regime_digest: str
    source: str = "unavailable"
    corpus_sha256: str = "unavailable"
    world: str
    steps: int
    software: Literal["passed", "error"] = "error"
    learning: LearningEvidence
    paired: list[PairedDeal] = Field(default_factory=list)
    policies: list[PolicySnapshot] = Field(default_factory=list)
    families: list[FamilyChange] = Field(default_factory=list)
    exposure: list[ActionExposure] = Field(default_factory=list)
    error: str | None = None


def _paired(initial: MonitorResult, final: MonitorResult) -> list[PairedDeal]:
    if initial.status != "completed" or final.status != "completed":
        raise ValueError("arena evaluation incomplete")
    if (
        initial.protocol != final.protocol
        or initial.training_seed != final.training_seed
    ):
        raise ValueError("arena protocol/seed changed between checkpoints")
    before = {(row.deal_seed, row.leg): row for row in initial.rows}
    after = {(row.deal_seed, row.leg): row for row in final.rows}
    expected = {(deal, leg) for deal in initial.protocol.deal_seeds for leg in range(4)}
    if (
        set(before) != expected
        or set(after) != expected
        or len(before) != len(initial.rows)
        or len(after) != len(final.rows)
    ):
        raise ValueError("missing or duplicate arena legs")
    result = []
    for deal in initial.protocol.deal_seeds:
        left, right = [], []
        for leg in range(4):
            a, b = before[deal, leg], after[deal, leg]
            if not a.valid or not b.valid or a.score_a is None or b.score_a is None:
                raise ValueError("invalid arena outcome/replay")
            left.append(a.score_a)
            right.append(b.score_a)
        result.append(
            PairedDeal(
                deal=deal,
                initial=sum(left) / 4,
                final=sum(right) / 4,
                legs=[b - a for a, b in zip(left, right, strict=True)],
            )
        )
    return result


def report_baseline(directory: Path) -> BaselineReport:
    """Retain an honest report even when training, evaluation or probes failed."""
    experiment = ExperimentRun.model_validate_json(
        (directory / "experiment.json").read_text()
    )
    (attempt,) = experiment.attempts
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    report = BaselineReport(
        seed=attempt.seed,
        regime_digest="unavailable",
        world=managym.WORLD_VERSION,
        steps=0,
        learning=learning_evidence([]),
    )
    try:
        run = TrainingRun.model_validate_json(
            (Path(attempt.path) / "run.json").read_text()
        )
        report.regime_digest = run.regime_digest
        report.steps = run.updates_through(run.stages[-1].id)
        report.source = experiment.runtime["source"]
        if experiment.status != "completed" or attempt.status != "completed":
            evaluations = [
                Attempt.model_validate_json(path.read_text())
                for path in (directory / "monitoring").glob("attempt-*/attempt.json")
            ]
            reasons = [item.error for item in evaluations if item.status != "completed"]
            raise ValueError(
                f"experiment incomplete: {experiment.error or attempt.error or reasons}"
            )
        (declared,) = run.regime.stages
        if (
            run.status != "completed"
            or not isinstance(declared, TrainSelfPlay)
            or report.steps != declared.updates
        ):
            raise ValueError("training did not reach its declared horizon")
        if not sum(stage.optimizer_updates or 0 for stage in run.stages):
            raise ValueError("no optimizer updates were executed")
        exported = checkpoints(run, include_initial=True)
        if (
            not exported
            or exported[0].coordinates.updates != 0
            or exported[-1].coordinates.updates != report.steps
        ):
            raise ValueError("missing initial or endpoint checkpoint")
        monitors = [
            MonitorResult.model_validate_json(path.read_text())
            for path in (directory / "monitoring").rglob("monitor.json")
        ]
        (initial,) = [item for item in monitors if item.coordinates.updates == 0]
        (final,) = [
            item for item in monitors if item.coordinates.updates == report.steps
        ]
        if (
            initial.run_id != run.id
            or final.run_id != run.id
            or initial.artifact != exported[0].artifact
            or final.artifact != exported[-1].artifact
        ):
            raise ValueError(
                "arena evidence does not bind this run's initial/final checkpoints"
            )
        report.paired = _paired(initial, final)
        report.learning = learning_evidence([report.paired])
        cases: list[Case] = []
        corpus_hashes: list[str] = []
        deadline = time.perf_counter() + 60
        for round_index in range(3):
            target = directory / "choices" / str(round_index)
            rows, _, findings = collect_choices(
                target, 4000 + round_index, deadline, closest=True
            )
            if findings or not rows or time.perf_counter() >= deadline:
                raise ValueError(
                    f"native corpus audit incomplete or failed: {findings}"
                )
            collisions = {
                case.name: input_collisions(case)
                for case in rows
                if input_collisions(case)
            }
            if collisions:
                raise ValueError(f"public choices lost in encoded inputs: {collisions}")
            cases.extend(rows)
            corpus_hashes.append(file_sha256(target / "cases.json"))
        report.corpus_sha256 = canonical_sha256(corpus_hashes)
        with torch.random.fork_rng():
            for checkpoint in exported:
                if (
                    file_sha256(checkpoint.artifact["path"])
                    != checkpoint.artifact["sha256"]
                ):
                    raise ValueError("checkpoint bytes changed")
                report.policies.append(
                    inspect_policy(
                        Path(checkpoint.artifact["path"]),
                        cases,
                        checkpoint.coordinates.updates,
                    )
                )
        report.families = family_changes(report.policies[0], report.policies[-1])
        exposure: dict[int, ActionExposure] = {}
        for stage in run.stages:
            for diagnostic in stage.diagnostics:
                for group in diagnostic.get("selection_groups", []):
                    kind = group["action_type"]
                    total = exposure.setdefault(kind, ActionExposure(action_type=kind))
                    total.collected += group["rows"]
                    total.retained += group["retained"]
        report.exposure = [exposure[key] for key in sorted(exposure)]
        if sum(item.retained for item in report.exposure) == 0:
            raise ValueError("no observed training rows retained by learner")
        report.software = "passed"
    except Exception as error:
        report.error = f"{type(error).__name__}: {error}"
        report.learning = LearningEvidence(
            verdict="error", seeds=1, deals=0, reason=report.error
        )
    torch.set_num_threads(previous_threads)
    atomic_json(directory / "learning.json", report.model_dump(mode="json"))
    lines = [
        f"# Baseline learning: {report.learning.verdict}",
        "",
        f"Seed {report.seed}; {report.steps} updates; software: {report.software}.",
        report.learning.reason,
        "",
        f"Paired score gain: {report.learning.gain}; 95% interval: [{report.learning.lower}, {report.learning.upper}].",
        "",
        "Choice changes are descriptive, not strategic correctness or a pass criterion.",
        "",
        "| Family | States | Mean absolute log-odds change | Across-state SD of change |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {ACTION_SPACE_KIND_NAMES[r.family]} | {r.cases} | {r.mean_absolute_log_odds_change:.5f} | {r.std_log_odds_change:.5f} |"
        for r in report.families
    ]
    lines += [
        "",
        "Training exposure (chosen action type, not decision family):",
        "",
        "| Action type | Collected rows | Retained rows |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {ActionEnum(r.action_type).name.lower()} | {r.collected} | {r.retained} |"
        for r in report.exposure
    ]
    if report.paired:
        leg_gains = np.asarray([row.legs for row in report.paired]).mean(axis=0)
        lines += [
            "",
            "Paired gain by deck and seat (descriptive):",
            "",
            "| Deck | Seat | Gain |",
            "|---|---:|---:|",
        ]
        lines += [
            f"| {selected_decks(leg)[leg % 2]} | {leg % 2} | {100 * gain:+.1f} pp |"
            for leg, gain in enumerate(leg_gains)
        ]
    if report.error:
        lines += ["", report.error]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return report


def aggregate_reports(
    paths: list[Path], out: Path, *, expected_seeds: tuple[int, ...]
) -> LearningEvidence:
    """Require the complete declared seed cohort; partial success cannot turn green."""
    reports: list[BaselineReport] = []
    errors: list[str] = []
    for path in paths:
        try:
            reports.append(BaselineReport.model_validate_json(path.read_text()))
        except (OSError, ValueError) as error:
            errors.append(f"cannot read {path}: {error}")
    if len(expected_seeds) < 3 or len(set(expected_seeds)) != len(expected_seeds):
        errors.append(
            "nightly requires at least three distinct declared training seeds"
        )
    if sorted(report.seed for report in reports) != sorted(expected_seeds):
        errors.append("missing, duplicate or unexpected training seeds")
    if any(report.software != "passed" for report in reports):
        errors.append("a seed has failed or incomplete software/evaluation checks")
    if any(
        r.source == "unavailable" or r.corpus_sha256 == "unavailable" for r in reports
    ):
        errors.append("missing source or frozen-corpus identity")
    if (
        len(
            {
                (r.regime_digest, r.world, r.steps, r.source, r.corpus_sha256)
                for r in reports
            }
        )
        != 1
    ):
        errors.append("cohort recipe/world/horizon/source/corpus differs")
    evidence = (
        LearningEvidence(
            verdict="error", seeds=len(reports), deals=0, reason="; ".join(errors)
        )
        if errors
        else learning_evidence([r.paired for r in reports])
    )
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "result.json", evidence.model_dump(mode="json"))
    (out / "report.md").write_text(
        f"# Baseline nightly: {evidence.verdict}\n\n{evidence.reason}.\n\n"
        f"Training seeds: {len(reports)}/{len(expected_seeds)}. "
        f"Paired deal blocks per seed: {evidence.deals}.\n\n"
        f"Mean gain: {evidence.gain}; 95% hierarchical interval: "
        f"[{evidence.lower}, {evidence.upper}]. Per-seed gains: {evidence.per_seed}.\n\n"
        "The opponent is uniform legal random. This checks early outcome learning, "
        "not tactical mastery or general strength. Choice sensitivity is descriptive. "
        "Inconclusive is not a passing learning claim.\n"
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    evidence = aggregate_reports(
        args.reports, args.out, expected_seeds=tuple(args.seeds)
    )
    print((args.out / "report.md").read_text())
    raise SystemExit(
        0
        if evidence.verdict == "learning"
        else 2
        if evidence.verdict == "inconclusive"
        else 1
    )


if __name__ == "__main__":
    main()
