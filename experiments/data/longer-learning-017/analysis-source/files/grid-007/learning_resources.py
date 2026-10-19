"""Separate learner, worker, setup and rental clocks for protocol017.

This reads completed, audited artifacts and existing SDK records. It never ticks
the scheduler or admits work. Worker phase intervals include dispatch overhead;
they are not substituted for the learner's measured active/inclusive clocks.
"""

from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

from etude.launch.client import default
from pydantic import BaseModel, ConfigDict, Field

from manabot.training.models import TrainingRun

# This task-local helper lives beside the launcher state, outside the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu-validation"))
from cli import configure  # noqa: E402

ROOT = Path(".runs/grid-007/learning-017")
LEDGER = Path(".runs/gpu-bottleneck/budget.json")
ALLOCATION = "grid-007-longer-learning-017"


class Event(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    phase: str
    observed_unix_seconds: float = Field(ge=0)
    cell: str | None = None


class CellInterval(BaseModel):
    cell: str
    wrapper_seconds: float = Field(ge=0)


class GroupIntervals(BaseModel):
    start_to_first_cell_seconds: float = Field(ge=0)
    setup_phase_seconds: float = Field(ge=0)
    cuda_admission_phase_seconds: float = Field(ge=0)
    cells: list[CellInterval]


class Summary(BaseModel):
    parameters: int
    peak_cuda_allocated: int
    learner_transitions: int
    environment_decisions: int
    optimizer_exposures: int
    updates: int
    collection_seconds: float
    learning_seconds: float
    run_seconds: float


class DiagnosticCounts(BaseModel):
    optimizer_steps: int
    skipped_steps: int
    nonfinite_count: int
    rejected_steps: int


class NumericalAudit(BaseModel):
    diagnostic_counts: DiagnosticCounts


class RetainedCell(BaseModel):
    name: str
    cell: str
    seed: int
    size: str
    streams: int
    directory: Path
    summary: Summary
    numerical_audit: NumericalAudit


class Retention(BaseModel):
    complete_cohort: bool
    provider_absent: bool
    allocation_sha256: str
    cells: list[RetainedCell]


class Group(BaseModel):
    name: str
    pool: str
    seed: int
    id: str
    cells: list[str]


class Launch(BaseModel):
    allocation_sha256: str
    groups: list[Group]


class RentalEvidence(BaseModel):
    machine: str
    pool: str
    provider_id: str | None
    requested_at: datetime
    deleted_at: datetime
    usd_per_hour: Decimal
    estimated_charge_usd: Decimal
    request_to_deletion_seconds: Decimal
    quoted_rate_cost_floor_usd: Decimal


class CellResources(BaseModel):
    name: str
    cell: str
    seed: int
    size: str
    streams: int
    parameters: int
    learner_transitions: int
    iterations: int
    adam_steps: int
    skipped_empty_groups: int
    optimizer_exposures: int
    environment_decisions: int
    games: int
    inclusive_training_seconds: float
    active_training_seconds: float
    training_setup_seconds: float
    wrapper_seconds: float
    peak_cuda_bytes: int
    sampled_peak_rss_bytes: int | None
    run_sha256: str


class GroupResources(BaseModel):
    seed: int
    run: str
    pool: str
    log_sha256: str
    intervals: GroupIntervals
    cells: list[CellResources]
    rentals: list[RentalEvidence]


class BudgetAllocation(BaseModel):
    name: str
    status: str
    conservative_charge_usd: Decimal | None = None


class Budget(BaseModel):
    authorized_total_usd: Decimal
    conservative_rental_total_usd: Decimal
    committed_caps_usd: Decimal
    ancillary_reserve_usd: Decimal
    allocations: list[BudgetAllocation]


class Settlement(BaseModel):
    provider_absent: bool
    quoted_rate_cost_floor_usd: Decimal
    machines: list[str]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_intervals(path: Path, completed_cells: list[str]) -> GroupIntervals:
    """Extract wall intervals from worker events, preserving their clock's meaning."""
    events = [
        Event.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.startswith("{")
    ]
    if any(
        b.observed_unix_seconds < a.observed_unix_seconds
        for a, b in zip(events, events[1:])
    ):
        raise ValueError("Worker event clock moved backwards")

    def timestamp(phase: str, cell: str | None = None) -> float:
        matches = [e for e in events if e.phase == phase and e.cell == cell]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one event for {phase}/{cell}, got {len(matches)}"
            )
        return matches[0].observed_unix_seconds

    if not completed_cells:
        raise ValueError("No completed cell for setup boundary")
    first = timestamp(completed_cells[0])
    intervals = [
        CellInterval(
            cell=cell,
            wrapper_seconds=timestamp("cell_completed", cell) - timestamp(cell),
        )
        for cell in completed_cells
    ]
    return GroupIntervals(
        start_to_first_cell_seconds=first - timestamp("starting"),
        setup_phase_seconds=timestamp("cuda-admission") - timestamp("setup"),
        cuda_admission_phase_seconds=first - timestamp("cuda-admission"),
        cells=intervals,
    )


def cell_resources(cell: RetainedCell, interval: CellInterval) -> CellResources:
    path = cell.directory / "study/training/run/run.json"
    run = TrainingRun.model_validate_json(path.read_text())
    if run.status != "completed" or len(run.stages) != 1 or run.seed != cell.seed:
        raise ValueError("Resource record requires the exact completed trajectory")
    stage = run.stages[0]
    learning = run.regime.stages[0].learning
    if learning.gradient != "ataraxos_move" or learning.filter_scope != "actor_critic":
        raise ValueError(
            "Empty-filter counter interpretation requires the frozen learning rule"
        )
    adam_steps = stage.observed_optimizer_updates()
    counts = cell.numerical_audit.diagnostic_counts
    if adam_steps is None or adam_steps != counts.optimizer_steps:
        raise ValueError("Saved Adam counters differ from the retained numerical audit")
    if counts.nonfinite_count or counts.rejected_steps:
        raise ValueError("Numerical health no longer matches the retention audit")
    summary = cell.summary
    if (
        run.seconds != summary.run_seconds
        or stage.learner_transitions != summary.learner_transitions
        or stage.optimizer_exposures != summary.optimizer_exposures
        or stage.environment_decisions != summary.environment_decisions
        or stage.collection_seconds != summary.collection_seconds
        or stage.learning_seconds != summary.learning_seconds
        or run.updates_through() != summary.updates
        or interval.cell != cell.cell
    ):
        raise ValueError("Resource counters differ from their audited summary")
    return CellResources(
        name=cell.name,
        cell=cell.cell,
        seed=cell.seed,
        size=cell.size,
        streams=cell.streams,
        parameters=summary.parameters,
        learner_transitions=stage.learner_transitions,
        iterations=summary.updates,
        adam_steps=adam_steps,
        skipped_empty_groups=counts.skipped_steps,
        optimizer_exposures=stage.optimizer_exposures,
        environment_decisions=stage.environment_decisions,
        games=stage.games,
        inclusive_training_seconds=run.seconds,
        active_training_seconds=stage.collection_seconds + stage.learning_seconds,
        training_setup_seconds=run.setup_seconds,
        wrapper_seconds=interval.wrapper_seconds,
        peak_cuda_bytes=summary.peak_cuda_allocated,
        sampled_peak_rss_bytes=stage.sampled_peak_rss_bytes,
        run_sha256=digest(path),
    )


def main() -> None:
    retained_path = ROOT / "verified-results.json"
    retained = Retention.model_validate_json(retained_path.read_text())
    if (
        not retained.provider_absent
        or not retained.complete_cohort
        or len(retained.cells) != 22
    ):
        raise ValueError(
            "Wait for the full retained comparison and confirmed provider deletion"
        )
    launch = Launch.model_validate_json((ROOT / "launch.json").read_text())
    if not (
        launch.allocation_sha256
        == retained.allocation_sha256
        == digest(ROOT / "allocation.json")
    ):
        raise ValueError("Allocation changed")
    settlement = Settlement.model_validate_json(
        (ROOT / "rental-settlement.json").read_text()
    )
    if not settlement.provider_absent:
        raise ValueError("Rental settlement lacks deletion evidence")
    budget = Budget.model_validate_json(LEDGER.read_text())
    allocation = next(a for a in budget.allocations if a.name == ALLOCATION)
    if (
        allocation.status != "terminal-deletion-confirmed"
        or allocation.conservative_charge_usd is None
    ):
        raise ValueError("Study budget has not been settled")

    # Existing launcher configuration owns credentials. These SDK methods read
    # records only; no wait/status/tick/terminate/admission method is used here.
    configure()
    scheduler = default()
    rentals = scheduler.rentals()
    groups: list[GroupResources] = []
    for group in launch.groups:
        cells = [c for c in retained.cells if c.seed == group.seed]
        ordered = [name.replace(":", "-s") for name in group.cells]
        if {c.cell for c in cells} != set(ordered) or len(cells) != len(ordered):
            raise ValueError("Resource cohort differs from admitted cells")
        by_cell = {c.cell: c for c in cells}
        log = ROOT / group.name / "job.log"
        intervals = read_intervals(log, ordered)
        machines = {m.name: m for m in scheduler.machines(group.pool, rented=True)}
        rental_rows: list[RentalEvidence] = []
        for rental, charge in rentals:
            if rental.machine not in machines:
                continue
            machine = machines[rental.machine]
            if machine.requested_at is None or machine.deleted_at is None:
                raise ValueError("Rental lacks request/deletion timestamps")
            seconds = Decimal(
                str((machine.deleted_at - machine.requested_at).total_seconds())
            )
            if seconds < 0:
                raise ValueError("Rental deletion predates request")
            floor = max(charge, rental.usd_per_hour * seconds / Decimal(3600))
            rental_rows.append(
                RentalEvidence(
                    machine=machine.name,
                    pool=group.pool,
                    provider_id=machine.provider_id,
                    requested_at=machine.requested_at,
                    deleted_at=machine.deleted_at,
                    usd_per_hour=rental.usd_per_hour,
                    estimated_charge_usd=charge,
                    request_to_deletion_seconds=seconds,
                    quoted_rate_cost_floor_usd=floor,
                )
            )
        if len(rental_rows) != 1:
            raise ValueError("Expected exactly one recorded rental per frozen group")
        groups.append(
            GroupResources(
                seed=group.seed,
                run=group.id,
                pool=group.pool,
                log_sha256=digest(log),
                intervals=intervals,
                cells=[cell_resources(by_cell[i.cell], i) for i in intervals.cells],
                rentals=rental_rows,
            )
        )
    rows = [r for group in groups for r in group.rentals]
    if {r.machine for r in rows} != set(settlement.machines) or sum(
        r.quoted_rate_cost_floor_usd for r in rows
    ) != settlement.quoted_rate_cost_floor_usd:
        raise ValueError("Rental records disagree with controller settlement")
    output = {
        "status": "verified",
        "groups": [g.model_dump(mode="json") for g in groups],
        "study_conservative_charge_usd": str(allocation.conservative_charge_usd),
        "campaign_budget": budget.model_dump(mode="json"),
        "input_sha256": {
            str(p): digest(p)
            for p in (
                retained_path,
                ROOT / "allocation.json",
                ROOT / "launch.json",
                ROOT / "rental-settlement.json",
                LEDGER,
            )
        },
        "collector_sha256": digest(Path(__file__)),
        "limits": "Worker phase intervals include dispatch overhead. Request-to-deletion occupancy is conservative; shared setup and retention are not assigned to individual recipes. SDK estimates are not final provider invoices.",
    }
    (ROOT / "resources.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "verified",
                "cells": sum(len(g.cells) for g in groups),
                "rentals": len(rows),
                "study_conservative_charge_usd": str(
                    allocation.conservative_charge_usd
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
