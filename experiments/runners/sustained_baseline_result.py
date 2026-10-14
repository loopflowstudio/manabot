"""Retained endpoint evaluation and analysis for the frozen ETU-118 experiment.

The shared queue/arena owns games, replay and failed attempts. This module selects
the predeclared initial/800/1240/endpoint raw artifacts and interprets complete paired
seed cohorts. It does not choose a daily regression test from early gains.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Literal

from nbclient import NotebookClient
import nbformat
import numpy as np
from pydantic import Field

from manabot.arena.models import file_sha256
from manabot.training.checkpoint_queue import CheckpointQueue, MonitoringBudget
from manabot.training.clock import boot_identity
from manabot.training.execution import atomic_json
from manabot.training.experiment_execution import ExperimentRun
from manabot.training.models import Strict
from manabot.training.monitor_evaluation import MonitorProtocol, MonitorResult
from manabot.training.recovery import attempt_lock

if TYPE_CHECKING:
    from experiments.runners.sustained_baseline import SustainedPlan


class FinalReceipt(Strict):
    status: Literal["running", "paused", "completed", "incomplete"] = "running"
    plan_sha256: str
    active_seconds: float = Field(default=0, ge=0)
    process_seconds: float = Field(default=0, ge=0)
    calendar_seconds: float = Field(default=0, ge=0)
    last_seen_unix: float
    last_seen_active: float | None = None
    boot_identity: str | None = None
    downtime_seconds: float = 0
    uncertain_seconds: float = 0
    error: str | None = None


class Comparison(Strict):
    opponent: str
    reference_updates: int
    endpoint_updates: int
    seed_gains: list[float]
    mean_gain: float
    seed_interval: tuple[float, float]
    deal_interval: tuple[float, float]
    seed_deal_intervals: list[tuple[float, float]]


class Conclusion(Strict):
    status: Literal["incomplete", "criteria-not-met", "criteria-met"]
    comparisons: list[Comparison] = []
    limits: str = "Three training seeds; fixed anchors; no human-challenger or daily-test acceptance."


def analyze(root: Path, plan: SustainedPlan, binding: str) -> Conclusion:
    """Never impute failed cells or select a favorable checkpoint after scoring."""
    endpoint = plan.milestones[-1]
    results = [
        MonitorResult.model_validate_json(p.read_text())
        for p in sorted((root / "final").rglob("monitor.json"))
    ]
    if len(results) != 24 or any(
        r.status != "completed" or len(r.rows) != 400 for r in results
    ):
        return Conclusion(status="incomplete")
    if any(
        r.protocol.comparison_sha256 != binding or r.regime_digest != plan.regime_digest
        for r in results
    ):
        raise ValueError("final evidence differs from frozen comparison identity")
    comparisons: list[Comparison] = []
    for opponent in ("scripted_greedy", "random"):
        for reference in (0, 800, 1240):
            gains: list[float] = []
            deal_gains: list[np.ndarray] = []
            for seed in plan.schedule.seeds:
                matched = [
                    r
                    for r in results
                    if r.training_seed == seed and r.protocol.opponent == opponent
                ]
                if {r.coordinates.updates for r in matched} != {
                    0,
                    800,
                    1240,
                    endpoint,
                } or len(matched) != 4:
                    raise ValueError(
                        "endpoint cohort does not match declared seed/checkpoint cross"
                    )
                pairs = [
                    next(r for r in matched if r.coordinates.updates == n)
                    for n in (reference, endpoint)
                ]
                expected = [
                    (d, leg)
                    for d in plan.schedule.scientific_deal_seeds
                    for leg in range(4)
                ]
                if any(
                    [(row.deal_seed, row.leg) for row in r.rows] != expected
                    or not all(
                        row.replay_passed and row.terminated and row.failure is None
                        for row in r.rows
                    )
                    for r in pairs
                ):
                    raise ValueError("endpoint cohort is incomplete or non-replayed")
                scores = [
                    sum(float(row.score_a) for row in r.rows) / 400 for r in pairs
                ]
                gains.append(scores[1] - scores[0])
                deal_gains.append(
                    np.array(
                        [
                            float(b.score_a) - float(a.score_a)
                            for a, b in zip(pairs[0].rows, pairs[1].rows, strict=True)
                        ]
                    )
                    .reshape(-1, 4)
                    .mean(axis=1)
                )
            means = np.random.default_rng(118).choice(gains, (10000, 3)).mean(axis=1)
            lo, hi = np.quantile(means, [0.025, 0.975])
            # Shared deal blocks preserve all four legs and paired checkpoint/seed
            # comparisons. This interval conditions on the three trained seeds.
            matrix = np.stack(deal_gains)
            indexes = np.random.default_rng(119).integers(
                0, matrix.shape[1], size=(10000, matrix.shape[1])
            )
            sampled = matrix[:, indexes].mean(axis=2)
            deal_lo, deal_hi = np.quantile(sampled.mean(axis=0), [0.025, 0.975])
            comparisons.append(
                Comparison(
                    opponent=opponent,
                    reference_updates=reference,
                    endpoint_updates=endpoint,
                    seed_gains=gains,
                    mean_gain=float(np.mean(gains)),
                    seed_interval=(float(lo), float(hi)),
                    deal_interval=(float(deal_lo), float(deal_hi)),
                    seed_deal_intervals=[
                        tuple(float(v) for v in np.quantile(row, [0.025, 0.975]))
                        for row in sampled
                    ],
                )
            )
    initial, short, early, random_initial, _, _ = comparisons
    passed = (
        initial.mean_gain >= 0.10
        and min(initial.seed_gains) > 0
        and initial.seed_interval[0] > 0
        and short.mean_gain >= 0.05
        and short.seed_interval[0] > 0
        and early.mean_gain >= 0.05
        and early.seed_interval[0] > 0
        and random_initial.mean_gain >= -0.05
    )
    return Conclusion(
        status="criteria-met" if passed else "criteria-not-met", comparisons=comparisons
    )


def refresh(root: Path) -> Path:
    """Execute the existing editable notebook; never replace its cells or outputs."""
    path = root / "comparison.ipynb"
    book = nbformat.read(path, as_version=4)
    NotebookClient(
        book,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(root.resolve())}},
    ).execute()
    return root / "comparison.html"


def finalize(
    plan: SustainedPlan, plan_path: Path, root: Path, *, resume: bool = False
) -> FinalReceipt:
    plan.admit()
    training = ExperimentRun.model_validate_json((root / "experiment.json").read_text())
    if training.status != "completed":
        raise ValueError("final comparison requires a completed sustained cohort")
    target = root / "final-result.json"
    binding = file_sha256(plan_path)
    with attempt_lock(root / "final.lock"):
        if target.exists():
            if not resume:
                raise ValueError("explicit final-evaluation resume required")
            receipt = FinalReceipt.model_validate_json(target.read_text())
            if receipt.plan_sha256 != binding:
                raise ValueError("frozen final plan changed")
            if receipt.status == "completed":
                return receipt
            gap = max(0, time.time() - receipt.last_seen_unix)
            receipt.calendar_seconds += gap
            if receipt.status == "running":
                unknown = gap
                if (
                    receipt.boot_identity == boot_identity()
                    and receipt.last_seen_active is not None
                ):
                    unknown = max(0, time.monotonic() - receipt.last_seen_active)
                receipt.active_seconds += unknown
                receipt.uncertain_seconds += unknown
                receipt.downtime_seconds += max(0, gap - unknown)
            else:
                receipt.downtime_seconds += gap
        else:
            receipt = FinalReceipt(plan_sha256=binding, last_seen_unix=time.time())
        (root / "pause.request").unlink(missing_ok=True)
        active, wall = time.monotonic(), time.time()
        prior_active, prior_calendar = receipt.active_seconds, receipt.calendar_seconds
        prior_downtime = receipt.downtime_seconds
        receipt.status, receipt.error = "running", None

        def save() -> None:
            receipt.active_seconds = prior_active + time.monotonic() - active
            receipt.calendar_seconds = prior_calendar + time.time() - wall
            receipt.last_seen_unix = time.time()
            receipt.last_seen_active = time.monotonic()
            receipt.boot_identity = boot_identity()
            receipt.downtime_seconds = prior_downtime + max(
                0, time.time() - wall - (time.monotonic() - active)
            )
            atomic_json(target, receipt.model_dump(mode="json"))

        sources = [
            Path(a.path) / "run.json"
            for a in training.attempts
            if (Path(a.path) / "run.json").exists()
        ]
        try:
            for opponent in ("scripted_greedy", "random"):
                config = MonitoringBudget(
                    seconds=(plan.final_reserve_seconds - plan.report_reserve_seconds)
                    / 2,
                    attempt_seconds=3600,
                    active_runtime=True,
                    include_initial=True,
                    include_monitoring=True,
                    updates=(0, 800, 1240, plan.milestones[-1]),
                    protocol=MonitorProtocol(
                        deal_seeds=plan.schedule.scientific_deal_seeds,
                        game_seconds=60,
                        opponent=opponent,
                        comparison_sha256=binding,
                    ),
                )
                queue = CheckpointQueue(root / "final" / opponent, config)
                try:
                    while True:
                        pausing = (root / "pause.request").exists()
                        queue.tick(sources, launch=not pausing)
                        save()
                        if (
                            receipt.active_seconds
                            >= plan.final_reserve_seconds - plan.report_reserve_seconds
                        ):
                            raise TimeoutError(
                                "final evaluation/report reserve exhausted"
                            )
                        if queue.process is None:
                            if pausing:
                                receipt.status = "paused"
                                return receipt
                            break
                        time.sleep(1)
                finally:
                    queue.close()
            conclusion = analyze(root, plan, binding)
            atomic_json(root / "conclusion.json", conclusion.model_dump(mode="json"))
            receipt.status = (
                "incomplete" if conclusion.status == "incomplete" else "completed"
            )
            refresh(root)
        except BaseException as error:
            receipt.status, receipt.error = (
                "incomplete",
                f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            receipt.process_seconds = sum(
                float(json.loads(p.read_text())["charged_evaluator_process_seconds"])
                for p in (root / "final").glob("*/dashboard.json")
            )
            save()
    return receipt
