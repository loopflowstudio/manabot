"""Read-only questions for normally trained baseline checkpoints.

Frozen native choices describe policy sensitivity; paired arena outcomes supply
learning evidence. Neither changes the learner or labels an arbitrary action as
correct. Game uncertainty resamples complete deal blocks; cohort uncertainty also
resamples training seeds. Incomplete evidence is never a passing learning result.
"""

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Literal

from etude.train.read import read_all
import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field

from manabot.env.observation import ActionEnum
from manabot.eval.evaluation import EvaluationResult, evaluate_local
from manabot.eval.local import checkpoint_request
from manabot.eval.results import EvaluationRow
from manabot.infra.files import atomic_json
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.schema import Strict
from manabot.model.world import validate_agent_setup
from manabot.policy.protocol import BackendPolicy, PolicyLoader
from manabot.sim.matchup import selected_decks
from manabot.sim.search.flat_mc import ACTION_SPACE_KIND_NAMES
from manabot.training.loops.self_play import Counts, SelfPlay, learner_digest
from manabot.training.state.backend import VerificationBackend
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


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _logsumexp(values: np.ndarray) -> float:
    peak = float(values.max())
    return peak + float(np.log(np.exp(values - peak).sum()))


def inspect_policy(
    path: Path, cases: list[Case], updates: int, *, backend: PolicyLoader
) -> PolicySnapshot:
    """Use the checkpoint loader and exact setup; never refit weights."""
    if not cases:
        raise ValueError("empty frozen choice corpus")
    agent, _ = backend.load_policy(path)
    configs = [
        managym.authored_deck_setup(PACK, key) for key in ("ur_lessons", "gw_allies")
    ]
    validate_agent_setup(agent, configs)
    restored, _ = backend.load_policy(path)
    choices: list[ChoiceScore] = []
    maximum = 0.0
    for start in range(0, len(cases), 32):
        scores, error = score_choices(agent, restored, cases[start : start + 32])
        choices.extend(scores)
        maximum = max(maximum, error)
    return PolicySnapshot(
        updates=updates,
        checkpoint_sha256=file_sha256(path),
        choices=choices,
        maximum_invariance_error=maximum,
    )


def score_choices(
    agent: BackendPolicy, restored: BackendPolicy, cases: list[Case]
) -> tuple[list[ChoiceScore], float]:
    """Score a bounded batch and check reload, batching and permutation invariance."""

    def evaluate(
        policy: BackendPolicy, observation: Mapping[str, ArrayLike]
    ) -> tuple[np.ndarray, np.ndarray]:
        host = {key: np.asarray(value) for key, value in observation.items()}
        return policy.evaluate(policy.prepare(host, inference=True))

    stacked = {
        key: np.concatenate([case.obs[key] for case in cases], axis=0)
        for key in cases[0].obs
    }
    logits, values = evaluate(agent, stacked)
    probabilities = _softmax(logits)
    if not np.isfinite(probabilities).all() or not np.isfinite(values).all():
        raise ValueError("nonfinite frozen checkpoint output")
    loaded_logits, loaded_values = evaluate(restored, stacked)
    errors = [
        float(np.abs(_softmax(loaded_logits) - probabilities).max()),
        float(np.abs(loaded_values - values).max()),
    ]
    scores: list[ChoiceScore] = []
    for row, case in enumerate(cases):
        obs = case.obs
        single, value = evaluate(agent, obs)
        p = _softmax(single)
        valid = np.asarray(obs["actions_valid"])
        count = int(valid.sum())
        order = list(reversed(range(count)))
        offers, offer_value = evaluate(agent, reorder_offers(obs, order))
        objects, object_value = evaluate(agent, reorder_objects(obs))
        errors.extend(
            [
                float(np.abs(p[0] - probabilities[row]).max()),
                float(np.abs(value[0] - values[row]).max()),
                float(np.abs(_softmax(offers)[0, :count] - p[0, order]).max()),
                float(np.abs(_softmax(objects) - p).max()),
                float(np.abs(offer_value - value).max()),
                float(np.abs(object_value - value).max()),
                float(p[valid == 0].sum()),
            ]
        )
        # Log-sum-exp avoids underflow and gives identical public copies shared
        # probability mass. Its sign has no prescribed strategic interpretation.
        left, right = case.groups
        log_odds = _logsumexp(single[0, left]) - _logsumexp(single[0, right])
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
    return scores, maximum


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
    # Reports written before the world label was removed carry it; ignored.
    world: str | None = Field(default=None, exclude_if=lambda value: value is None)
    steps: int
    software: Literal["passed", "error"] = "error"
    learning: LearningEvidence
    paired: list[PairedDeal] = Field(default_factory=list)
    policies: list[PolicySnapshot] = Field(default_factory=list)
    families: list[FamilyChange] = Field(default_factory=list)
    exposure: list[ActionExposure] = Field(default_factory=list)
    error: str | None = None


def paired_rows(
    initial_rows: Sequence[EvaluationRow],
    final_rows: Sequence[EvaluationRow],
    deals: tuple[int, ...],
) -> list[PairedDeal]:
    """Admit every replayed leg before reducing the four legs to one deal block."""
    before = {(row.deal_seed, row.leg): row for row in initial_rows}
    after = {(row.deal_seed, row.leg): row for row in final_rows}
    expected = {(deal, leg) for deal in deals for leg in range(4)}
    if (
        set(before) != expected
        or set(after) != expected
        or len(before) != len(initial_rows)
        or len(after) != len(final_rows)
    ):
        raise ValueError("missing or duplicate arena legs")
    result = []
    for deal in deals:
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


@dataclass(frozen=True)
class PolicyPoint:
    path: Path
    updates: int
    sha256: str


def inspect_baseline(
    report: BaselineReport,
    directory: Path,
    exported: Sequence[PolicyPoint],
    metric_rows: Sequence[Mapping[str, object]],
    *,
    backend: PolicyLoader,
) -> None:
    """Fill policy and exposure diagnostics in place, using the frozen native corpus."""
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
            raise ValueError(f"native corpus audit incomplete or failed: {findings}")
        collisions = {
            case.name: input_collisions(case) for case in rows if input_collisions(case)
        }
        if collisions:
            raise ValueError(f"public choices lost in encoded inputs: {collisions}")
        cases.extend(rows)
        corpus_hashes.append(file_sha256(target / "cases.json"))
    report.corpus_sha256 = canonical_sha256(corpus_hashes)
    for checkpoint in exported:
        if file_sha256(checkpoint.path) != checkpoint.sha256:
            raise ValueError("checkpoint bytes changed")
        report.policies.append(
            inspect_policy(
                Path(checkpoint.path),
                cases,
                checkpoint.updates,
                backend=backend,
            )
        )
    report.families = family_changes(report.policies[0], report.policies[-1])
    exposure: dict[int, ActionExposure] = {}
    for row in metric_rows:
        for name, count in row.items():
            parts = name.split("/")
            if len(parts) != 4 or parts[0] != "selection_groups":
                continue
            if parts[3] not in ("rows", "retained") or not isinstance(count, int):
                continue
            kind = int(parts[1])
            total = exposure.setdefault(kind, ActionExposure(action_type=kind))
            if parts[3] == "rows":
                total.collected += count
            else:
                total.retained += count
    report.exposure = [exposure[key] for key in sorted(exposure)]
    if sum(item.retained for item in report.exposure) == 0:
        raise ValueError("no observed training rows retained by learner")
    report.software = "passed"


def write_baseline_report(report: BaselineReport, directory: Path) -> BaselineReport:
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


def report_recipe(
    directory: Path,
    settings: SelfPlay,
    *,
    seed: int,
    updates: int,
    deals: tuple[int, ...],
    seconds: float,
    backend: VerificationBackend,
) -> BaselineReport:
    """Evaluate the loop's initial/final raw policies and retain the nightly evidence.

    This is an explicit post-training evaluation; all four deck/seat legs share
    the same deals. Etude remains the owner of training attempts and metric rows:
    `directory` is the `out` folder of the Etude run directory they are read from.
    """
    report = BaselineReport(
        seed=seed,
        regime_digest=learner_digest(settings),
        steps=updates,
        learning=learning_evidence([]),
    )
    try:
        with backend.runtime.scoped(1, "float32", "cpu"):
            exported = [
                PolicyPoint(path, int(path.stem.split("-")[0]), file_sha256(path))
                for path in sorted((directory / "policies").glob("*-raw.safetensors"))
            ]
            if (
                not exported
                or exported[0].updates != 0
                or exported[-1].updates != updates
            ):
                raise ValueError("missing initial or endpoint checkpoint")
            payload = backend.metadata(str(exported[-1].path))
            counts = Counts.model_validate(payload.get("bc", {}).get("counts"))
            if counts.updates != updates or counts.optimizer_updates == 0:
                raise ValueError(
                    "training did not complete the requested optimizer updates"
                )
            source = payload.get("source", {}).get("commit")
            if not isinstance(source, str):
                raise ValueError("checkpoint has no source revision")
            report.source = source
            results: list[EvaluationResult] = []
            deadline = time.time() + seconds
            for point in (exported[0], exported[-1]):
                if time.time() >= deadline:
                    raise TimeoutError("verification evaluation budget exhausted")
                request = checkpoint_request(
                    point.path, opponent="random", deal_seeds=deals, backend=backend
                )
                target = directory / "evaluation" / f"{point.updates:08d}"
                if (target / "evaluation.json").exists():
                    result = EvaluationResult.model_validate_json(
                        (target / "evaluation.json").read_bytes()
                    )
                    if result.execution is not None or result.request != request:
                        raise ValueError("evaluation belongs to another checkpoint")
                else:
                    result = evaluate_local(
                        request,
                        deadline=deadline,
                        out_dir=target,
                    )
                result.validate_cohort()
                if result.status != "completed" or result.request.deal_seeds != deals:
                    raise ValueError(
                        "paired arena evaluation incomplete or from other deals"
                    )
                results.append(result)
            report.paired = paired_rows(results[0].rows, results[1].rows, deals)
            run_directory = directory.parent
            (tracked,) = [
                run
                for run in read_all(run_directory.parent)
                if run.name == run_directory.name
            ]
            exposure = {
                name: int(np.nansum(tracked.points(name, "updates").y))
                for name in tracked.metrics()
                if name.startswith("selection_groups/")
            }
            with backend.runtime.preserve_rng("cpu"):
                inspect_baseline(
                    report, directory, exported, [exposure], backend=backend
                )
    except Exception as error:
        report.error = f"{type(error).__name__}: {error}"
        report.learning = LearningEvidence(
            verdict="error", seeds=1, deals=0, reason=report.error
        )
    return write_baseline_report(report, directory)


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
        len({(r.regime_digest, r.steps, r.source, r.corpus_sha256) for r in reports})
        != 1
    ):
        errors.append("cohort recipe/horizon/source/corpus differs")
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
