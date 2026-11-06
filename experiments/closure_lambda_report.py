"""Regenerate a closure-lambda report from stored Etude runs and evaluations.

Two study sets share this builder: `allies-lessons` (Task fc6ffb, protocol
experiments/protocols/closure-lambda-v1.md) and `elemental` (Task 07c490, protocol
closure-lambda-elemental-v1.md). Each reads its two studies' training records
(`run.json`, per-update metric blocks) and evaluation tables from the bucket, then
writes tables, figures and `results.json` under its `experiments/data/<set>/` folder.
The prose in `report.md` is written from those tables, so a rebuild never rewrites it.

Jobs are labelled by job name and seed count and averaged across seeds. Scores are
interpolated onto a common clock per run, then averaged, so runs whose checkpoints fall
at different learner decisions, seconds or dollars compare at equal positions.

    uv run --extra dev --extra artifacts --extra reports python -m \
        experiments.closure_lambda_report --set elemental --costs costs.json
"""

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import cast

from etude.eval.query import statsdb
from etude.experiment import observe
from etude.run.block_file import parse, select
from etude.run.storage import Storage, storage_from_uri
from etude.train.block import decode, rows
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
SOURCE = "r2://manabot/jobs"
DISTANCES = ("terminal", "1-4", "5+", "censored")
# The rate every rental in both protocols paid; receipts in costs.json hold the bills.
L4_USD_PER_HOUR = 0.59
BOOTSTRAP_RESAMPLES = 2000
INTERVAL = 0.95
# A healthy assembly takes about a second at these row counts.
ASSEMBLY_STALL_SECONDS = 5.0
# Two-sided 95% Student t quantiles by degrees of freedom; studies run 2 to 3 seeds.
T_95 = {1: 12.706, 2: 4.303}


@dataclass(frozen=True)
class Study:
    experiment: str
    baseline: str
    title: str


@dataclass(frozen=True)
class StudySet:
    """A protocol's two studies, the legs each scored pairing plays, and its folder."""

    name: str
    out: Path
    legs: int
    studies: tuple[Study, ...]


SETS = {
    "allies-lessons": StudySet(
        "allies-lessons",
        DATA / "closure-lambda-v1",
        8,
        (
            Study(
                "batch_closure",
                "batch-closure-exact-quota-v1",
                "Study 1: batch closure, Allies/Lessons only",
            ),
            Study(
                "lambda_sweep",
                "lambda-sweep-p0.5-v0.8-v1",
                "Study 2: lambda, Allies/Lessons only",
            ),
        ),
    ),
    "elemental": StudySet(
        "elemental",
        DATA / "closure-lambda-elemental-v1",
        4,
        (
            Study(
                "elemental_closure",
                "elemental-closure-exact-quota-v1",
                "Study 1: batch closure, ten-deck Elemental, one learning seat",
            ),
            Study(
                "elemental_lambda",
                "elemental-lambda-p0.5-v0.8-v1",
                "Study 2: lambda, ten-deck Elemental",
            ),
        ),
    ),
}


@dataclass
class TrainingRecord:
    """One training run: its stage totals and per-update metric rows."""

    run: str
    job: str
    seed: int
    stage: dict[str, object]
    updates: list[dict[str, float]]
    # The run's game-error ledger state; empty when the regime records none.
    game_errors: dict[str, object] = field(default_factory=dict)

    def row(self, update: int) -> dict[str, float]:
        """The metric row at an update, 1-indexed by the metric rows."""
        return next(row for row in self.updates if row["update"] == update)

    def at_update(self, update: int, key: str) -> float:
        return self.row(update)[key]


@dataclass
class Scored:
    """A run's held-out score at each evaluated update: per-deal means, in deal order."""

    run: str
    updates: list[int] = field(default_factory=list)
    deal_scores: dict[int, dict[int, float]] = field(default_factory=dict)

    def score(self, update: int) -> float:
        return float(np.mean(list(self.deal_scores[update].values())))


# --------------------------------------------------------------------------- reading


def _latest_run_json(
    storage: Storage, run: str, keys: Sequence[str]
) -> dict[str, object]:
    found = sorted(k for k in keys if k.endswith("training/run.json"))
    if not found:
        raise ValueError(f"{run} stored no training/run.json")
    return cast(dict[str, object], json.loads(storage.get(found[-1])))


def _metric_rows(
    storage: Storage, run: str, keys: Sequence[str]
) -> list[dict[str, float]]:
    """Numeric per-update rows in order; each is keyed `update` from its coordinates."""
    files = [
        each
        for each in (parse(k.removeprefix(f"{run}/")) for k in keys)
        if each is not None
    ]
    found: list[dict[str, float]] = []
    for each in select(files):
        for row in rows(decode(storage.get(f"{run}/{each.key}"))):
            values = cast(dict[str, object], row.get("values", row))
            merged = {**row, **values}
            if merged.get("coordinates/updates") is None or merged.get("rows") is None:
                continue
            numbers = {
                k: float(v)
                for k, v in merged.items()
                if isinstance(v, int | float) and not k.startswith("cumulative_games")
            }
            numbers["update"] = numbers["coordinates/updates"]
            found.append(numbers)
    return found


def training_records(storage: Storage, experiment: str) -> list[TrainingRecord]:
    records: list[TrainingRecord] = []
    for run in observe(storage, experiment).runs:
        keys = list(storage.keys(run.name))
        if not any(k.endswith("training/run.json") for k in keys):
            continue  # stopped before training; the report's failure table names it
        document = _latest_run_json(storage, run.name, keys)
        regime = cast(dict[str, object], document["regime"])
        stages = cast(list[dict[str, object]], document["stages"])
        records.append(
            TrainingRecord(
                run=run.name,
                job=cast(str, regime["id"]),
                seed=cast(int, document["seed"]),
                stage=stages[0],
                updates=_metric_rows(storage, run.name, keys),
                game_errors=cast(
                    dict[str, object], document.get("game_errors") or {}
                ),
            )
        )
    return sorted(records, key=lambda r: (r.job, r.seed))


def scored_runs(storage: Storage, experiment: str, legs: int) -> dict[str, Scored]:
    """Candidate scores from the evaluation database, averaged over each deal's legs."""
    database = statsdb(experiment, storage)
    found = database.execute(
        """
        select regexp_extract(s.matchup, '^(.*) u[0-9]+ vs heuristic$', 1) as run,
               cast(regexp_extract(s.matchup, ' u([0-9]+) vs heuristic$', 1) as int) as update,
               cast(json_extract(m.conditions, '$.deal') as bigint) as deal,
               avg(p.score) as score,
               count(*) as plays
        from plays p
        join matches m on p.run = m.run and p.match = m.match
        join schedule s on s.match = p.match
        join seats t on t.run = p.run and t.match = p.match and t.seat = p.seat
        join players y on y.player = t.player
        where y.kind = 'model'
        group by 1, 2, 3
        order by 1, 2, 3
        """
    ).fetchall()
    scored: dict[str, Scored] = {}
    for run, update, deal, score, plays in found:
        if plays != legs:
            raise ValueError(
                f"{run} u{update} deal {deal} has {plays} legs, not {legs}"
            )
        entry = scored.setdefault(run, Scored(run=run))
        if update not in entry.deal_scores:
            entry.updates.append(update)
            entry.deal_scores[update] = {}
        entry.deal_scores[update][deal] = float(score)
    return scored


# --------------------------------------------------------------------------- analysis


def label(job: str, seeds: int) -> str:
    return f"{job} · {seeds} seed{'s' if seeds != 1 else ''}"


def by_job(records: Iterable[TrainingRecord]) -> dict[str, list[TrainingRecord]]:
    jobs: dict[str, list[TrainingRecord]] = {}
    for each in records:
        jobs.setdefault(each.job, []).append(each)
    return jobs


def _clock(record: TrainingRecord, update: int, kind: str) -> float:
    """Where a run stood at an update on the decisions, seconds or dollars clock."""
    if kind == "decisions":
        return record.at_update(update, "coordinates/learner_transitions")
    seconds = record.at_update(update, "coordinates/active_training_seconds")
    if kind == "seconds_without_assembly":
        # Assembly stalls are intermittent and host-side, in both arms (see report.md).
        return seconds - sum(
            row.get("time/collection/batch_assembly", 0.0)
            for row in record.updates
            if row["update"] <= update
        )
    return seconds if kind == "seconds" else seconds / 3600 * L4_USD_PER_HOUR


def curve(
    record: TrainingRecord, scored: Scored, kind: str, grid: np.ndarray
) -> np.ndarray:
    """Score interpolated onto `grid`; positions outside the scored span are NaN."""
    updates = sorted(scored.updates)
    x = np.array([_clock(record, u, kind) for u in updates])
    y = np.array([scored.score(u) for u in updates])
    return np.interp(grid, x, y, left=np.nan, right=np.nan)


def matched_grid(
    records: Sequence[TrainingRecord], scored: dict[str, Scored], kind: str
) -> np.ndarray:
    """The span every run reached, from the last first checkpoint to the first last."""
    firsts: list[float] = []
    lasts: list[float] = []
    for record in records:
        updates = sorted(scored[record.run].updates)
        firsts.append(_clock(record, updates[0], kind))
        lasts.append(_clock(record, updates[-1], kind))
    return np.linspace(max(firsts), min(lasts), 5)


def seed_interval(differences: Sequence[float]) -> tuple[float, float, float]:
    """Mean with a t interval over paired seed differences."""
    values = np.asarray(differences, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    half = float(T_95[len(values) - 1] * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def deal_interval(
    arm: Sequence[Scored],
    baseline: Sequence[Scored],
    arm_update: Sequence[int],
    base_update: Sequence[int],
) -> tuple[float, float]:
    """Interval for the arm difference from resampling the shared deals, seeds fixed."""
    deals = sorted(arm[0].deal_scores[arm_update[0]])
    a = np.array(
        [[s.deal_scores[u][d] for d in deals] for s, u in zip(arm, arm_update)]
    ).mean(axis=0)
    b = np.array(
        [[s.deal_scores[u][d] for d in deals] for s, u in zip(baseline, base_update)]
    ).mean(axis=0)
    generator = np.random.default_rng(6510)
    picks = generator.integers(0, len(deals), (BOOTSTRAP_RESAMPLES, len(deals)))
    samples = (a - b)[picks].mean(axis=1)
    return (
        float(np.quantile(samples, (1 - INTERVAL) / 2)),
        float(np.quantile(samples, (1 + INTERVAL) / 2)),
    )


def residual_by_distance(record: TrainingRecord, update: int) -> dict[str, float]:
    """Critic residual per distance bin, aggregated by rows across action types."""
    row = record.row(update)
    totals: dict[str, list[float]] = {each: [0.0, 0.0] for each in DISTANCES}
    for key, value in row.items():
        parts = key.split("/")
        if len(parts) == 4 and parts[0] == "selection_groups" and parts[3] == "rows":
            residual = row.get("/".join((*parts[:3], "residual_abs_mean")))
            if residual is not None and parts[2] in totals:
                totals[parts[2]][0] += value * residual
                totals[parts[2]][1] += value
    return {
        each: (total / count if count else float("nan"))
        for each, (total, count) in totals.items()
    }


def _timing(record: TrainingRecord) -> dict[str, float]:
    stage = record.stage
    collection = cast(dict[str, dict[str, float]], stage["collection_timings"])
    learning = cast(dict[str, dict[str, float]], stage["learning_timings"])
    updates = float(cast(int, stage["updates"]))
    decisions = float(cast(int, stage["environment_decisions"]))
    idle = float(cast(int, stage["idle_stream_steps"]))
    rows_total = float(cast(int, stage["learner_transitions"]))
    collection_seconds = float(cast(float, stage["collection_seconds"]))
    learning_seconds = float(cast(float, stage["learning_seconds"]))
    assembly_seconds = collection["Batch assembly"]["seconds"]
    idle_share = idle / (idle + decisions)
    return {
        "updates": updates,
        "rows": rows_total,
        "rows_per_update": rows_total / updates,
        "games": float(cast(int, stage["games"])),
        "optimizer_steps": float(cast(int, stage["optimizer_updates"])),
        "environment_decisions": decisions,
        "collection_seconds": collection_seconds,
        "learning_seconds": learning_seconds,
        "assembly_seconds": assembly_seconds,
        "targets_seconds": learning["Targets"]["seconds"],
        "diagnostics_seconds": learning["Selection and health diagnostics"]["seconds"],
        "idle_stream_steps": idle,
        # Games that failed (capacity overflows under recovery) and the unfinished
        # samples they discarded.
        "crashed_games": float(cast(int, record.game_errors.get("crashed", 0))),
        "discarded_samples": float(
            cast(int, record.game_errors.get("discarded_samples", 0))
        ),
        "idle_share": idle_share,
        # Wall time lost to idle slots if an idle slot cost as much as a stepped one:
        # an upper bound, since batched inference is sublinear in live streams.
        "idle_collection_seconds_bound": (collection_seconds - assembly_seconds)
        * idle_share,
        "assembly_stalled_updates": float(
            sum(
                row.get("time/collection/batch_assembly", 0.0) > ASSEMBLY_STALL_SECONDS
                for row in record.updates
            )
        ),
        "median_update_seconds": float(
            np.median(
                [
                    row["time/collection"] + row["time/learning"]
                    for row in record.updates
                ]
            )
        ),
        "seconds_per_million_rows": (collection_seconds + learning_seconds)
        / rows_total
        * 1e6,
    }


def _mean_table(records: Sequence[TrainingRecord]) -> dict[str, float]:
    timings = [_timing(r) for r in records]
    return {k: float(np.mean([t[k] for t in timings])) for k in timings[0]}


def analyze(
    study: Study, records: list[TrainingRecord], scored: dict[str, Scored]
) -> dict[str, object]:
    jobs = by_job(records)
    baseline = jobs[study.baseline]
    result: dict[str, object] = {
        "experiment": study.experiment,
        "title": study.title,
        "runs": [
            {
                "run": r.run,
                "job": r.job,
                "seed": r.seed,
                "scored_updates": sorted(scored[r.run].updates)
                if r.run in scored
                else [],
                **_timing(r),
            }
            for r in records
        ],
        "jobs": {},
    }
    usable = [r for r in records if r.run in scored and scored[r.run].updates]
    views: dict[str, dict[str, object]] = {}
    for kind in ("decisions", "seconds", "seconds_without_assembly", "dollars"):
        grid = matched_grid(usable, scored, kind)
        # Each job's curves by seed; the baseline's seeds pair with every other job's.
        by_seed = {
            job: {
                r.seed: curve(r, scored[r.run], kind, grid)
                for r in members
                if r.run in scored
            }
            for job, members in jobs.items()
        }
        base_by_seed = by_seed[study.baseline]
        view: dict[str, object] = {"grid": grid.tolist(), "jobs": {}}
        for job, per_seed in by_seed.items():
            shared = sorted(set(per_seed) & set(base_by_seed))
            end = [float(per_seed[s][-1] - base_by_seed[s][-1]) for s in shared]
            cast(dict[str, object], view["jobs"])[job] = {
                "label": label(job, len(jobs[job])),
                "score": np.nanmean(list(per_seed.values()), axis=0).tolist(),
                "difference_at_end": seed_interval(end)
                if job != study.baseline
                else None,
            }
        views[kind] = view
    result["views"] = views
    # Deal intervals compare each arm's last common scored update with the baseline's.
    common = sorted(set.intersection(*(set(scored[r.run].updates) for r in usable)))
    last = common[-1]
    base_scored = [scored[r.run] for r in baseline if r.run in scored]
    for job, members in jobs.items():
        present = [r for r in members if r.run in scored]
        entry: dict[str, object] = {
            "label": label(job, len(members)),
            "seeds": len(members),
            "means": _mean_table(members),
            "score_by_update": {
                str(u): float(
                    np.mean(
                        [
                            scored[r.run].score(u)
                            for r in present
                            if u in scored[r.run].deal_scores
                        ]
                    )
                )
                for u in common
            },
            "seed_scores_at_last_common": {
                str(r.seed): scored[r.run].score(last) for r in present
            },
            "residual_by_distance": {
                str(u): {
                    each: float(
                        np.nanmean([residual_by_distance(r, u)[each] for r in members])
                    )
                    for each in DISTANCES
                }
                for u in common
            },
        }
        if job != study.baseline:
            entry["deal_interval_at_last_common"] = deal_interval(
                [scored[r.run] for r in present],
                base_scored,
                [last] * len(present),
                [last] * len(base_scored),
            )
        cast(dict[str, object], result["jobs"])[job] = entry
    result["last_common_update"] = last
    return result


# --------------------------------------------------------------------------- outputs


def _figure(
    study: Study, records: list[TrainingRecord], result: dict[str, object], out: Path
) -> None:
    views = cast(dict[str, dict[str, object]], result["views"])
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for axis, (kind, title) in zip(
        axes,
        (
            ("decisions", "Learner decisions"),
            ("seconds", "Active training seconds"),
            ("dollars", "Training dollars"),
        ),
    ):
        view = views[kind]
        for values in cast(dict[str, dict[str, object]], view["jobs"]).values():
            axis.plot(view["grid"], values["score"], marker="o", label=values["label"])
        axis.set_xlabel(title)
    axes[0].set_ylabel("Score against the heuristic")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.suptitle(study.title)
    figure.tight_layout(rect=(0, 0.14, 1, 1))
    figure.savefig(out / f"{study.experiment}-strength.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for record in records:
        update = [r["update"] for r in record.updates]
        axes[0].plot(
            update,
            [r["rows"] for r in record.updates],
            lw=0.8,
            label=f"{record.job} {record.seed}",
        )
        axes[1].plot(
            update,
            [r.get("time/collection/batch_assembly", np.nan) for r in record.updates],
            lw=0.8,
        )
    axes[0].set_ylabel("Rows per update")
    axes[1].set_ylabel("Batch assembly seconds")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("Update")
    axes[0].legend(fontsize=5)
    figure.tight_layout()
    figure.savefig(out / f"{study.experiment}-rows-assembly.png", dpi=140)
    plt.close(figure)


def _assembly_series(records: list[TrainingRecord]) -> list[dict[str, object]]:
    """Each run's assembly seconds against its rows, per update: the cause's test data."""
    return [
        {
            "run": r.run,
            "job": r.job,
            "seed": r.seed,
            "host_cpus": None,
            "update": [x["update"] for x in r.updates],
            "rows": [x["rows"] for x in r.updates],
            "assembly": [x.get("time/collection/batch_assembly") for x in r.updates],
            "targets": [x.get("time/learning/targets") for x in r.updates],
        }
        for r in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=sorted(SETS), default="allies-lessons")
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--out", type=Path, help="default: the set's data folder")
    parser.add_argument(
        "--costs", type=Path, help="billed receipts to carry into results"
    )
    arguments = parser.parse_args()
    chosen = SETS[arguments.set]
    out = arguments.out or chosen.out
    out.mkdir(parents=True, exist_ok=True)
    storage = storage_from_uri(arguments.source)
    results: dict[str, object] = {}
    for study in chosen.studies:
        records = training_records(storage, study.experiment)
        if not records:
            # A study stopped before training (its protocol says why) has no arms.
            results[study.experiment] = {"title": study.title, "runs": []}
            continue
        scored = scored_runs(storage, study.experiment, chosen.legs)
        result = analyze(study, records, scored)
        result["assembly"] = _assembly_series(records)
        results[study.experiment] = result
        _figure(study, records, result, out)
    if arguments.costs is not None:
        results["costs"] = json.loads(arguments.costs.read_text())
    (out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
