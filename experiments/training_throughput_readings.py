"""C1 throughput evidence, extending the retained step-3 span reader.

Historical readers/results remain frozen artifacts. This is the maintained reader:
`uv run python -m experiments.training_throughput_readings PATH/step-3-c1 --source COMMIT` reads
cells.json and runs/<id>/{run.json,updates.json,gpu-samples.csv}. cells.json maps
*every* C1 leg to a Disposition (runs, status, reason, host, billed_usd). Skips and
failures remain visible; retries are separate cells, never pooled or speed-picked.
The freeze's source commit is required as --source. JSON output includes exact
ordinals, per-seed latency/rates and matched-solo comparisons. Cost admission is
pure arithmetic over reconciled spend and reserved liability, not a rental API.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics as st
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter

from experiments.training_throughput_sizes import (
    C1_LEGS,
    C1_SEEDS,
    Leg,
    Limits,
    leg_recipe,
)
from manabot.infra.identity import canonical_sha256
from manabot.training.records import TrainingRun
from manabot.training.regime import TrainingRegime

MIN_UPDATES = 3
PHASES = {
    "native": "time/collection/environment_step_and_encoding/native_stepping",
    "encoding": "time/collection/environment_step_and_encoding/observation_encoding",
    "inference": "time/collection/policy_inference_and_action_selection",
    "learning": "time/learning",
}


@dataclass(frozen=True)
class Update:
    number: int
    start: float
    end: float
    decisions: float
    values: dict[str, float]

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Run:
    id: str
    seed: int
    origin: float
    updates: tuple[Update, ...]
    source: str
    world: str
    host: str
    control: str
    resources: dict[str, float] = field(default_factory=dict)
    exposure: dict[str, float] = field(default_factory=dict)


class SeedReading(BaseModel):
    run: str
    seed: int
    ordinals: list[int]
    seconds: float
    decisions_per_hour: float
    phases: dict[str, float]
    window_seconds: tuple[float, float]
    resources: dict[str, float]


class Reading(BaseModel):
    status: Literal["read", "insufficient_overlap"]
    ordinals: dict[str, list[int]]
    seeds: list[SeedReading] = Field(default_factory=list)
    seconds: float | None = None
    decisions_per_hour: float | None = None


def read_cell(leg: Leg, runs: list[Run]) -> Reading:
    """Summarize complete concurrent updates, excluding warmup update 1."""
    if leg.admission:
        raise ValueError("admission is not a speed cell")
    if len(runs) != leg.packed or {r.seed for r in runs} != set(C1_SEEDS[: leg.packed]):
        raise ValueError("cell must contain every declared seed exactly once")
    if len({(r.source, r.world, r.host, r.control) for r in runs}) != 1:
        raise ValueError("cell mixes source/world/host/control identities")
    if any(not r.updates for r in runs):
        return Reading(status="insufficient_overlap", ordinals={r.id: [] for r in runs})
    low = max(r.origin for r in runs)
    high = min(r.updates[-1].end for r in runs)
    chosen = [
        [
            u
            for u in r.updates
            if 2 <= u.number <= leg.updates and u.start >= low and u.end <= high
        ]
        for r in runs
    ]
    ordinals = {
        r.id: [u.number for u in rows] for r, rows in zip(runs, chosen, strict=True)
    }
    if any(len(rows) < MIN_UPDATES for rows in chosen):
        return Reading(status="insufficient_overlap", ordinals=ordinals)
    seeds = []
    for r, rows in zip(runs, chosen, strict=True):
        seconds = st.median(u.seconds for u in rows)
        middle = len(rows) // 2
        seeds.append(
            SeedReading(
                run=r.id,
                resources=r.resources,
                seed=r.seed,
                ordinals=[u.number for u in rows],
                seconds=seconds,
                decisions_per_hour=3600
                * st.median(u.decisions for u in rows)
                / seconds,
                window_seconds=(
                    st.median(u.seconds for u in rows[:middle]),
                    st.median(u.seconds for u in rows[middle:]),
                ),
                phases={
                    name: st.median(u.values[key] for u in rows)
                    for name, key in PHASES.items()
                },
            )
        )
    return Reading(
        status="read",
        ordinals=ordinals,
        seeds=seeds,
        seconds=st.mean(s.seconds for s in seeds),
        decisions_per_hour=sum(s.decisions_per_hour for s in seeds),
    )


class Comparison(BaseModel):
    packed: Reading
    solo_ordinals: dict[str, list[int]]
    latency_gain: float
    decision_gain: float
    per_seed_latency_gain: list[float]


def compare(solo: Run, packed: Reading, runs: list[Run]) -> Comparison:
    """Match each packed seed's exact ordinals in the same-host solo control.

    The seed-26401 contrast is paired; other seeds/window contrasts are sensitivity,
    not independent repeated estimates or confidence intervals.
    """
    if packed.status != "read" or not runs or solo.seed != C1_SEEDS[0]:
        raise ValueError("comparison requires an admitted cell and declared solo seed")
    if any(
        (r.source, r.world, r.host, r.control)
        != (solo.source, solo.world, solo.host, solo.control)
        for r in runs
    ):
        raise ValueError("comparison mixes source/world/host/control identities")
    by_number = {u.number: u for u in solo.updates}
    matches: dict[str, list[int]] = {}
    solo_seconds: list[float] = []
    solo_rates: list[float] = []
    per_seed: list[float] = []
    for seed in packed.seeds:
        if not set(seed.ordinals) <= by_number.keys():
            raise ValueError(
                "solo endpoint lacks matching ordinals; run the declared longer control"
            )
        rows = [by_number[n] for n in seed.ordinals]
        seconds = st.median(u.seconds for u in rows)
        solo_seconds.append(seconds)
        solo_rates.append(3600 * st.median(u.decisions for u in rows) / seconds)
        per_seed.append(seconds / seed.seconds)
        matches[seed.run] = seed.ordinals
    assert packed.seconds is not None and packed.decisions_per_hour is not None
    return Comparison(
        packed=packed,
        solo_ordinals=matches,
        latency_gain=len(runs) * st.mean(solo_seconds) / packed.seconds,
        decision_gain=packed.decisions_per_hour / st.mean(solo_rates),
        per_seed_latency_gain=per_seed,
    )


@dataclass(frozen=True)
class Budget:
    prior_step3: float
    prior_total: float
    committed: float  # C1 bills plus outstanding rental/cleanup upper bounds
    gpu_hours: float
    updates: int


def admit_cost(
    budget: Budget,
    limit: Limits,
    *,
    rate: float,
    remaining_seconds: float,
    line_remaining_usd: float,
    packed: int,
    updates: int,
    cleanup_seconds: float = 300,
    prepaid_usd: float = 0,
) -> float:
    """Reserve one shared machine window, not K copies of its rate.

    Caller includes unposted bills in prior spend and existing liability in
    committed. A held rental already reserved in committed passes only its
    *additional* liability here using prepaid_usd; time/update caps hold.
    No optimistic pace estimate replaces the declared watchdog/setup window.
    """
    numbers = (
        budget.prior_step3,
        budget.prior_total,
        budget.committed,
        budget.gpu_hours,
        rate,
        remaining_seconds,
        line_remaining_usd,
        cleanup_seconds,
        prepaid_usd,
    )
    if (
        any(not math.isfinite(x) or x < 0 for x in numbers)
        or budget.prior_total < budget.prior_step3
    ):
        raise ValueError(
            "cost admission requires finite, reconciled nonnegative bounds"
        )
    if packed not in (1, 2, 4) or updates < 1 or budget.updates < 0:
        raise ValueError("invalid update allocation")
    hours = limit.max_hours + cleanup_seconds / 3600
    if rate <= 0 or prepaid_usd > min(budget.committed, hours * rate):
        raise ValueError(
            "prepaid liability must already be reserved; rate must be positive"
        )
    liability = hours * rate - prepaid_usd
    available = (
        min(10 - budget.prior_step3, 50 - budget.prior_total, 8) - budget.committed
    )
    if (
        hours * 3600 > remaining_seconds
        or budget.gpu_hours + hours > 15
        or budget.updates + packed * updates > 1600
        or liability > min(available, line_remaining_usd)
    ):
        raise ValueError(
            "cell watchdog and cleanup do not fit the remaining C1 budget/window"
        )
    return liability


class Disposition(BaseModel):
    status: Literal[
        "completed",
        "conditional_skip",
        "unavailable",
        "budget_skip",
        "failed",
        "not_needed",
    ]
    reason: str = Field(min_length=1)
    runs: list[str] = Field(default_factory=list)
    failed_runs: list[str] = Field(default_factory=list)
    host: str = ""
    billed_usd: float = Field(ge=0, allow_inf_nan=False)


class Rental(BaseModel):
    gpu: Literal["a4000", "l4", "fast"]
    usd_per_hour: float = Field(gt=0, allow_inf_nan=False)
    billed_usd: float = Field(ge=0, allow_inf_nan=False)
    outstanding_usd: float = Field(ge=0, allow_inf_nan=False)
    hours: float = Field(ge=0, allow_inf_nan=False)
    bill_reference: str = Field(min_length=1)
    deletion_reference: str = Field(min_length=1)


def _control(regime: TrainingRegime) -> str:
    """Complete resolved recipe except endpoint, id and watchdog."""
    payload = regime.model_dump(mode="json")
    payload.pop("id")
    payload.pop("wall_seconds", None)
    for stage in payload["stages"]:
        stage.pop("updates", None)
        stage["execution"].pop("wall_seconds", None)
    return json.dumps(payload, sort_keys=True)


def load_run(root: Path, run_id: str, leg: Leg, host: str, source: str) -> Run:
    """Read copied receipts, rejecting historical inputs in the C1 cohort."""
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run id must name one evidence directory")
    folder = root / "runs" / run_id
    receipt = TrainingRun.model_validate_json((folder / "run.json").read_bytes())
    if (
        receipt.identities.get("source_commit") != source
        or receipt.status != "completed"
    ):
        raise ValueError("run is not completed on the frozen C1 source")
    regime = receipt.regime
    expected = leg_recipe(leg)
    if (
        regime.id != expected.id
        or regime.observation.policy_input_version != 6
        or regime.observation.max_facts != 512
        or regime.agent.catalog_records(6)
    ):
        raise ValueError("run does not carry the C1 hashed v6 declaration")
    if receipt.regime_digest != canonical_sha256(regime.model_dump(mode="json")):
        raise ValueError("receipt regime digest disagrees with its bytes")
    control = _control(regime)
    if control != _control(expected):
        raise ValueError("run changed the declared C1 recipe")
    samples = (folder / "gpu-samples.csv").read_text().splitlines()
    origin = (
        datetime.strptime(samples[0].split(",")[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    raw = TypeAdapter(dict[int, dict[str, float]]).validate_json(
        (folder / "updates.json").read_bytes()
    )
    if sorted(raw) != list(range(1, leg.updates + 1)):
        raise ValueError("run did not reach its declared endpoint")
    for row in raw.values():
        if any(not math.isfinite(v) for v in row.values()):
            raise ValueError("nonfinite update evidence")
        if row.get("numerical/nonfinite_count", 0) or row.get(
            "numerical/rejected_steps", 0
        ):
            raise ValueError("rejected/nonfinite learning step")
    updates: list[Update] = []
    for number in range(2, leg.updates + 1):
        previous, row = raw[number - 1], raw[number]
        start = origin + previous["coordinates/training_seconds"]
        end = origin + row["coordinates/training_seconds"]
        decisions = (
            row["coordinates/learner_transitions"]
            - previous["coordinates/learner_transitions"]
        )
        if end <= start or decisions <= 0:
            raise ValueError("invalid update clocks")
        games = row["coordinates/games"] - previous["coordinates/games"]
        environment_decisions = (
            row["coordinates/environment_decisions"]
            - previous["coordinates/environment_decisions"]
        )
        if games <= 0 or environment_decisions <= 0:
            raise ValueError("invalid complete-game exposure clocks")
        # Cumulative exposure must be differenced before comparing game lengths;
        # environment decisions count the game, learner rows count training work.
        values = row | {
            "games": games,
            "decisions_per_game": environment_decisions / games,
        }
        updates.append(Update(number, start, end, decisions, values))
    gpu_samples = [line.split(",") for line in samples]
    resources = {
        "gpu_busy_percent_whole_run": st.mean(float(row[1]) for row in gpu_samples),
        "gpu_sampled_peak_mib": max(float(row[2]) for row in gpu_samples),
        "peak_reserved_cuda_bytes": max(
            row.get("memory/peak_reserved_cuda_bytes", 0) for row in raw.values()
        ),
        "peak_allocated_cuda_bytes": max(
            row.get("memory/peak_allocated_cuda_bytes", 0) for row in raw.values()
        ),
    }
    if receipt.stages and receipt.stages[0].sampled_peak_rss_bytes is not None:
        resources["peak_rss_bytes"] = receipt.stages[0].sampled_peak_rss_bytes
    last = raw[leg.updates]
    if "coordinates/process_cpu_seconds" in last:
        resources["busy_cpu_cores_whole_run"] = (
            last["coordinates/process_cpu_seconds"]
            / last["coordinates/training_seconds"]
        )
    exposure = {
        key: value for key, value in last.items() if key.startswith("coordinates/")
    }
    return Run(
        run_id,
        receipt.seed,
        origin,
        tuple(updates),
        source,
        regime.world,
        host,
        control,
        resources,
        exposure,
    )


def precision_check(loaded: dict[str, list[Run]]) -> dict[str, object]:
    """Gross late-update health against the float32 repeat, never equivalence."""
    prefix = "c1-fast-w160l8-k1-"
    names = {
        "float32": prefix + "float32-u12",
        "repeat": prefix + "float32-repeat-u12",
        "tf32": prefix + "tf32-u12",
        "bf16": prefix + "bf16-u12",
    }
    missing = [label for label, name in names.items() if name not in loaded]
    if missing:
        return {"status": "incomplete", "missing": missing}
    runs = {label: loaded[name][0] for label, name in names.items()}
    if len({(r.source, r.world, r.host, r.seed) for r in runs.values()}) != 1:
        raise ValueError("precision check mixes source/world/host/seed identities")
    metrics = (
        "policy_loss",
        "value_loss",
        "entropy",
        "collection_kl",
        "reference_kl",
        "numerical/clip_fraction",
        "rows",
        "games",
        "decisions_per_game",
    )
    means = {}
    for label, run in runs.items():
        rows = [u for u in run.updates if 5 <= u.number <= 12]
        if [u.number for u in rows] != list(range(5, 13)):
            raise ValueError("precision check requires every update 5-12")
        means[label] = {
            metric: st.mean(u.values[metric] for u in rows) for metric in metrics
        }
    differences = {
        label: {metric: values[metric] - means["float32"][metric] for metric in metrics}
        for label, values in means.items()
        if label != "float32"
    }
    return {
        "status": "read",
        "ordinals": list(range(5, 13)),
        "means": means,
        "differences_from_float32": differences,
        "absolute_difference_over_repeat": {
            label: {
                metric: abs(values[metric] / differences["repeat"][metric])
                if differences["repeat"][metric] != 0
                else None
                for metric in metrics
            }
            for label, values in differences.items()
            if label != "repeat"
        },
        "interpretation": (
            "gross numeric health only; equivalence unestablished; ratios are "
            "unstable near zero repeat differences and null at zero"
        ),
    }


def _packing_comparisons(
    loaded: dict[str, list[Run]], readings: dict[str, Reading]
) -> dict[str, object]:
    """Choose scheduled attempts and matching solo endpoints without speed-picking."""
    comparisons: dict[str, object] = {}
    for name, leg in C1_LEGS.items():
        if leg.packed == 1 or leg.overlap_retry:
            continue
        candidates = [name] + [
            n
            for n, other in C1_LEGS.items()
            if other.overlap_retry
            and (other.gpu, other.size, other.packed, other.precision)
            == (leg.gpu, leg.size, leg.packed, leg.precision)
        ]
        # First admissible scheduled cell, not the fastest measured cell.
        chosen = next(
            (n for n in candidates if n in readings and readings[n].status == "read"),
            None,
        )
        if chosen is None:
            comparisons[name] = {"status": "unmeasured"}
            continue
        solo_names = [
            n
            for n, other in C1_LEGS.items()
            if other.packed == 1
            and not other.repeat
            and not other.admission
            and (other.gpu, other.size, other.precision)
            == (leg.gpu, leg.size, leg.precision)
        ]
        for solo_name in sorted(solo_names, key=lambda n: C1_LEGS[n].updates):
            if solo_name not in loaded:
                continue
            try:
                comparison = compare(
                    loaded[solo_name][0], readings[chosen], loaded[chosen]
                )
            except ValueError:
                continue
            comparisons[name] = {
                "cell": chosen,
                "solo": solo_name,
                **comparison.model_dump(),
            }
            break
        else:
            comparisons[name] = {"status": "missing matched solo", "cell": chosen}
    return comparisons


def analyze(root: Path, source: str) -> dict[str, object]:
    if root.name != "step-3-c1":
        raise ValueError("C1 reads only its separately retained step-3-c1 directory")
    rentals = TypeAdapter(dict[str, Rental]).validate_json(
        (root / "rentals.json").read_bytes()
    )
    cells = TypeAdapter(dict[str, Disposition]).validate_json(
        (root / "cells.json").read_bytes()
    )
    if cells.keys() != C1_LEGS.keys():
        raise ValueError(
            f"reconcile every C1 cell: missing={sorted(C1_LEGS.keys() - cells.keys())}, unknown={sorted(cells.keys() - C1_LEGS.keys())}"
        )
    used = [run for cell in cells.values() for run in (*cell.runs, *cell.failed_runs)]
    if len(used) != len(set(used)):
        raise ValueError("a run cannot supply more than one declared cell")
    loaded: dict[str, list[Run]] = {}
    readings: dict[str, Reading] = {}
    for name, cell in cells.items():
        leg = C1_LEGS[name]
        if cell.status != "completed" or leg.admission:
            continue
        if cell.host not in rentals or rentals[cell.host].gpu != leg.gpu:
            raise ValueError(
                "completed cells require the declared GPU and exact rental host"
            )
        loaded[name] = [load_run(root, r, leg, cell.host, source) for r in cell.runs]
        readings[name] = read_cell(leg, loaded[name])
    costs = {}
    for name, reading in readings.items():
        if reading.status != "read":
            continue
        assert reading.decisions_per_hour is not None
        rental = rentals[cells[name].host]
        costs[name] = {
            "steady_usd_per_million": rental.usd_per_hour
            * 1e6
            / reading.decisions_per_hour
        }
    return {
        "source": source,
        "effective_records": "hashed",
        "policy_input_version": 6,
        "cells": {n: c.model_dump() for n, c in cells.items()},
        "readings": {n: r.model_dump() for n, r in readings.items()},
        "comparisons": _packing_comparisons(loaded, readings),
        "precision_check": precision_check(loaded),
        "completed_run_clocks": {
            run.id: run.exposure for runs in loaded.values() for run in runs
        },
        "costs": costs,
        "rentals": {host: r.model_dump() for host, r in rentals.items()},
        "billed_usd": sum(r.billed_usd for r in rentals.values()),
        "outstanding_usd": sum(r.outstanding_usd for r in rentals.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.source), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
