"""Portable ETU-105 evidence export and notebook analysis, with no execution.

Original receipts retain authority. The gzip bundle removes only per-update
selection_groups; every scalar, coordinate, configuration and scored arena row
remains. Hashes bind full originals, including failed calibration attempts.
"""

from __future__ import annotations

import argparse
import gzip
from html import escape
import json
from pathlib import Path
from statistics import mean
from typing import Literal, NotRequired, TypedDict

from matplotlib.figure import Figure
import numpy as np
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, model_validator

from experiments.runners.training_protocol import EvaluationProtocol
from manabot.arena.models import file_sha256
from manabot.training.analysis import cost_comparison
from manabot.training.experiment_execution import ExperimentRun
from manabot.training.experiment_report import ReportEvidence
from manabot.training.models import TrainingRun
from manabot.training.monitor_evaluation import (
    ArenaRow,
    MonitorResult,
    TrainingCoordinates,
)
from manabot.training.report_study import (
    ScientificStudy,
    StudyCell,
    StudyMeasurement,
    StudyRun,
)


class FileReceipt(BaseModel):
    path: str
    sha256: str
    bytes: int


class FinalCell(BaseModel):
    case: str
    seed: int
    family: str
    coordinates: TrainingCoordinates
    rows: list[ArenaRow]
    seconds: float
    status: str
    admission: dict[str, JsonValue]


class Bundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    source: str
    runs: list[TrainingRun]
    monitors: list[MonitorResult]
    execution: ExperimentRun
    final: list[FinalCell]
    protocol: EvaluationProtocol
    retained: dict[str, dict[str, JsonValue]]
    originals: list[FileReceipt]
    omitted: str = (
        "Per-update selection_groups only; full originals remain hash-bound on mini."
    )

    @model_validator(mode="after")
    def complete_cohort(self) -> Bundle:
        expected = {
            (case, seed)
            for case in ("actor", "actor-critic")
            for seed in self.protocol.training_seeds
        }
        if (
            len(self.runs) != 6
            or {(r.regime.id, r.seed) for r in self.runs} != expected
        ):
            raise ValueError("missing or duplicated training seed/arm")
        final_expected = {
            (case, seed, family)
            for case, seed in expected
            for family in ("final-greedy", "random-diagnostic")
        }
        if (
            len(self.final) != 12
            or {(c.case, c.seed, c.family) for c in self.final} != final_expected
        ):
            raise ValueError("missing or duplicated final seed/arm/family")
        for cell in self.final:
            deals = (
                self.protocol.endpoint_anchor_deals
                if cell.family == "final-greedy"
                else self.protocol.random_diagnostic_deals
            )
            if cell.status != "completed" or {
                (r.deal_seed, r.leg) for r in cell.rows
            } != {(d, leg) for d in deals for leg in range(4)}:
                raise ValueError("final cohort differs from frozen deals")
            blocks(cell.rows)
            run = next(
                r for r in self.runs if (r.regime.id, r.seed) == (cell.case, cell.seed)
            )
            if (
                cell.coordinates.updates != 1240
                or cell.admission["artifact"] != run.stages[-1].artifacts["raw"]
            ):
                raise ValueError("final endpoint changed")
        for run in self.runs:
            monitors = [m for m in self.monitors if m.run_id == run.id]
            if len(monitors) != 4 or not {0, 620, 1240} <= {
                m.coordinates.updates for m in monitors
            }:
                raise ValueError("monitoring milestone missing")
            for monitor in monitors:
                if (
                    monitor.protocol.deal_seeds != self.protocol.anchor_deals
                    or monitor.status != "completed"
                ):
                    raise ValueError("monitoring cohort changed")
                blocks(monitor.rows)
        return self


class Estimate(TypedDict):
    mean: float
    lower: float
    upper: float
    per_seed: list[float]


class Effect(TypedDict):
    actor_minus_actor_critic: Estimate
    arms: NotRequired[dict[str, Estimate]]
    comparison: NotRequired[dict[str, JsonValue]]


def read_object(path: Path) -> dict[str, JsonValue]:
    return TypeAdapter(dict[str, JsonValue]).validate_json(path.read_bytes())


def export(source: Path, final: Path, prior: Path, output: Path) -> None:
    originals: list[FileReceipt] = []

    def receipt(path: Path) -> None:
        originals.append(
            FileReceipt(
                path=str(path), sha256=file_sha256(path), bytes=path.stat().st_size
            )
        )

    execution_path = source / "science/experiment.json"
    receipt(execution_path)
    execution = ExperimentRun.model_validate_json(execution_path.read_bytes())
    if execution.status != "completed" or len(execution.attempts) != 6:
        raise ValueError("all six frozen runs required")
    runs: list[TrainingRun] = []
    for attempt in execution.attempts:
        path = source / "science/training" / Path(attempt.path).name / "run.json"
        receipt(path)
        run = TrainingRun.model_validate_json(path.read_bytes())
        if run.id != attempt.run_id or run.status != "completed":
            raise ValueError("TrainingRun differs from frozen attempt")
        for stage in run.stages:
            stage.diagnostics = [
                {k: v for k, v in d.items() if k != "selection_groups"}
                for d in stage.diagnostics
            ]
        runs.append(run)
    monitors: list[MonitorResult] = []
    for path in sorted((source / "science/monitoring").rglob("monitor.json")):
        receipt(path)
        monitors.append(MonitorResult.model_validate_json(path.read_bytes()))
    if len(monitors) != 24 or any(m.status != "completed" for m in monitors):
        raise ValueError(
            "requires all 24 original monitoring milestones, including initialization"
        )
    cells: list[FinalCell] = []
    for job_path in sorted(final.glob("job-*.json")):
        receipt(job_path)
        job = read_object(job_path)
        directory = final / Path(str(job["output"])).name
        result_path, admission_path = (
            directory / "result.json",
            directory / "admission.json",
        )
        receipt(result_path)
        receipt(admission_path)
        result, admission = read_object(result_path), read_object(admission_path)
        run = next(
            r
            for r in runs
            if any(
                s.artifacts.get("raw", {}).get("sha256")
                == admission["artifact"]["sha256"]
                for s in r.stages
            )
        )
        cells.append(
            FinalCell(
                case=run.regime.id,
                seed=run.seed,
                family="final-greedy"
                if job["opponent"] == "scripted_greedy"
                else "random-diagnostic",
                coordinates=TrainingCoordinates.model_validate(
                    admission["coordinates"]
                ),
                rows=TypeAdapter(list[ArenaRow]).validate_python(result["rows"]),
                seconds=float(result["seconds"]),
                status=str(result["status"]),
                admission=admission,
            )
        )
    if len(cells) != 12 or any(
        c.status != "completed"
        or len(c.rows) != 100
        or not all(r.valid for r in c.rows)
        for c in cells
    ):
        raise ValueError(
            "all frozen final/random cells required; partial results are not a complete screen"
        )
    retained: dict[str, dict[str, JsonValue]] = {}
    paths = [
        *(
            source / name
            for name in (
                "preflight.json",
                "calibration-intent.json",
                "frozen-plan.json",
                "recovery-verified.json",
                "supervisor-result.json",
                "evaluation-protocol.json",
            )
        ),
        source / "calibration/experiment.json",
        *sorted((source / "calibration/training").glob("*/run.json")),
        *sorted((source / "calibration/monitoring").rglob("monitor.json")),
        final / "supervisor.json",
        *sorted(prior.rglob("*.json")),
    ]
    for path in paths:
        receipt(path)
        obj = read_object(path)
        if path.name == "run.json":
            # Earlier calibration attempts retain recipes/identities/costs and failures.
            for stage in obj.get("stages", []):
                stage.pop("diagnostics", None)
        retained[str(path)] = obj
    bundle = Bundle(
        source="jack@100.96.227.95:/Users/jack/src/etude/.runs/etu105-mini-filter-scope-20261006-3-calibration-continuation",
        runs=runs,
        monitors=monitors,
        execution=execution,
        final=cells,
        protocol=EvaluationProtocol.model_validate_json(
            (source / "evaluation-protocol.json").read_bytes()
        ),
        retained=retained,
        originals=originals,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        bundle.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    output.write_bytes(gzip.compress(payload, mtime=0))
    print(
        f"Exported {len(runs)} runs, {len(monitors)} monitoring cohorts and {len(cells)} held-out/diagnostic cohorts: {output}"
    )


def load_bundle(path: Path) -> Bundle:
    return Bundle.model_validate_json(gzip.decompress(path.read_bytes()))


def evidence(bundle: Bundle, path: Path) -> ReportEvidence:
    return ReportEvidence(
        path.parent,
        tuple(bundle.runs),
        tuple(bundle.monitors),
        (bundle.execution,),
        (),
        (path,),
    )


def final_study(bundle: Bundle) -> ScientificStudy:
    """Project complete final cells into the shared scientific report reader."""
    return ScientificStudy(
        study="filter-scope-mini",
        status="completed",
        seeds=list(bundle.protocol.training_seeds),
        seconds=sum(c.seconds for c in bundle.final),
        runs=[StudyRun(regime=case) for case in ("actor-critic", "actor")],
        measurements=[
            StudyMeasurement(
                regime=c.case,
                seed=c.seed,
                phase=c.family,
                cutoff=1240,
                opponent=c.family,
                complete=c.status == "completed",
                score=float(blocks(c.rows).mean()),
                training_seconds=c.coordinates.training_seconds,
                learner_transitions=c.coordinates.learner_transitions,
                decisions=c.coordinates.environment_decisions,
            )
            for c in bundle.final
        ],
        comparisons=[
            StudyCell(
                a=c.case,
                b=c.family,
                training_seed=c.seed,
                cutoff=1240,
                phase=c.family,
                scheduled_games=100,
                rows=c.rows,
                replay={
                    "passed": c.status == "completed" and all(r.valid for r in c.rows)
                },
            )
            for c in bundle.final
        ],
    )


def interval(values: np.ndarray) -> Estimate:
    """Resample paired training seeds and shared four-leg deals, after subtraction."""
    if values.shape != (3, 25):
        raise ValueError("requires three paired seeds and 25 complete shared deals")
    rng = np.random.default_rng(105)
    estimates = np.array(
        [
            values[np.ix_(rng.integers(3, size=3), rng.integers(25, size=25))].mean()
            for _ in range(10000)
        ]
    )
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(estimates, 0.025)),
        "upper": float(np.quantile(estimates, 0.975)),
        "per_seed": values.mean(axis=1).tolist(),
    }


def blocks(rows: list[ArenaRow]) -> np.ndarray:
    deals = sorted({r.deal_seed for r in rows})
    if (
        len(rows) != 100
        or len(deals) != 25
        or any({r.leg for r in rows if r.deal_seed == d} != {0, 1, 2, 3} for d in deals)
        or not all(r.valid for r in rows)
    ):
        raise ValueError("incomplete or invalid four-leg evidence")
    return np.array(
        [mean(float(r.score_a) for r in rows if r.deal_seed == d) for d in deals]
    )


def checkpoint_rate(rows: list[ArenaRow]) -> str:
    """Checkpoint-conditional deal uncertainty, separate from training seeds."""
    values = blocks(rows)
    rng = np.random.default_rng(105)
    draws = values[rng.integers(25, size=(10000, 25))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return f"{values.mean():.0%} [{low:.0%}, {high:.0%}]"


def summarize(bundle: Bundle) -> dict[str, JsonValue]:
    effects: dict[str, Effect] = {}
    for family in ("final-greedy", "random-diagnostic"):
        arrays = {
            case: np.stack(
                [
                    blocks(
                        next(
                            c.rows
                            for c in bundle.final
                            if (c.case, c.seed, c.family) == (case, seed, family)
                        )
                    )
                    for seed in bundle.protocol.training_seeds
                ]
            )
            for case in ("actor-critic", "actor")
        }
        effects[family] = {
            "arms": {case: interval(a) for case, a in arrays.items()},
            "actor_minus_actor_critic": interval(
                arrays["actor"] - arrays["actor-critic"]
            ),
        }
    monitoring: list[dict[str, JsonValue]] = []
    for m in bundle.monitors:
        run = next(r for r in bundle.runs if r.id == m.run_id)
        monitoring.append(
            {
                "regime": run.regime.id,
                "seed": run.seed,
                "opponent": "greedy",
                "phase": "development",
                "complete": True,
                "score": m.score.mean,
                "training_seconds": m.coordinates.training_seconds,
                "checkpoint": m.artifact,
                "updates": m.coordinates.updates,
            }
        )
    common = cost_comparison(monitoring, "greedy")
    cutoff = cost_comparison(monitoring, "greedy", end_seconds=3600)
    # Paired differences use the actual earlier artifacts selected at the cutoff.
    for name, comparison in (("common-cost", common), ("one-hour", cutoff)):
        chosen: dict[tuple[str, int], MonitorResult] = {}
        for row in comparison["rows"]:
            chosen[(row["regime"], row["seed"])] = next(
                m
                for m in bundle.monitors
                if m.training_seed == row["seed"]
                and m.artifact["sha256"] == row["checkpoint"]["sha256"]
            )
        effects[name] = {
            "comparison": comparison,
            "actor_minus_actor_critic": interval(
                np.stack(
                    [
                        blocks(chosen[("actor", s)].rows)
                        - blocks(chosen[("actor-critic", s)].rows)
                        for s in bundle.protocol.training_seeds
                    ]
                )
            ),
        }
    exposures: list[dict[str, JsonValue]] = []
    diagnostics_summary: list[dict[str, JsonValue]] = []
    for run in bundle.runs:
        diagnostics = [d for s in run.stages for d in s.diagnostics]
        diagnostics_summary.append(
            {
                "case": run.regime.id,
                "seed": run.seed,
                **{
                    key: mean(
                        float(d[key])
                        for d in diagnostics[-100:]
                        if d.get(key) is not None
                    )
                    for key in (
                        "value_loss",
                        "entropy",
                        "collection_kl",
                        "reference_kl",
                        "bootstrapped_tail_fraction",
                    )
                },
                "gradient_cap_fraction": mean(
                    float(d["gradient_norm"]) > 0.267
                    for d in diagnostics
                    if d.get("gradient_norm") is not None
                ),
            }
        )
        exposures.append(
            {
                "case": run.regime.id,
                "seed": run.seed,
                "seconds": run.seconds,
                **{
                    k: sum(float(d.get(k, 0) or 0) for d in diagnostics)
                    for k in (
                        "actor_exposures",
                        "critic_exposures",
                        "optimizer_exposures",
                        "retained",
                        "rows",
                    )
                },
                "empty_actor_updates": sum(d["retained"] == 0 for d in diagnostics),
                "decisions": sum(s.environment_decisions for s in run.stages),
                "collection_seconds": sum(s.collection_seconds for s in run.stages),
                "learning_seconds": sum(s.learning_seconds for s in run.stages),
                "export_seconds": sum(s.export_seconds for s in run.stages),
            }
        )
    means = {
        case: mean(r.seconds for r in bundle.runs if r.regime.id == case)
        for case in ("actor", "actor-critic")
    }
    return {
        "effects": effects,
        "exposures": exposures,
        "last_100_update_diagnostics": diagnostics_summary,
        "mean_training_seconds": means,
        "actor_cost_ratio": means["actor"] / means["actor-critic"],
        "monitoring_games": sum(len(m.rows) for m in bundle.monitors),
        "final_games": sum(len(c.rows) for c in bundle.final),
        "monitoring_seconds": sum(m.evaluation_seconds for m in bundle.monitors),
        "final_evaluation_seconds": sum(c.seconds for c in bundle.final),
        "uncertainty": "95% paired training-seed/shared-deal percentile bootstrap, 10000 replicates, seed 105. Four seat/deck legs remain together. Three seeds give descriptive uncertainty only.",
    }


def table(headers: list[str], rows: list[list[str]]) -> str:
    return (
        '<div class="table"><table><thead><tr>'
        + "".join(f"<th>{escape(x)}</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(x)}</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def summary_html(bundle: Bundle) -> str:
    summary = summarize(bundle)
    rows = []
    for run in sorted(bundle.runs, key=lambda r: (r.seed, r.regime.id)):
        ms = [m for m in bundle.monitors if m.run_id == run.id]
        marks = []
        for update in (0, 620, 1240):
            m = next(m for m in ms if m.coordinates.updates == update)
            marks.append(
                f"{m.score.mean:.0%} [{m.score.lower:.0%}, {m.score.upper:.0%}]"
            )
        final = next(
            c
            for c in bundle.final
            if (c.case, c.seed, c.family) == (run.regime.id, run.seed, "final-greedy")
        )
        random = next(
            c
            for c in bundle.final
            if (c.case, c.seed, c.family)
            == (run.regime.id, run.seed, "random-diagnostic")
        )
        rows.append(
            [
                str(run.seed),
                run.regime.id,
                *marks,
                checkpoint_rate(final.rows),
                checkpoint_rate(random.rows),
                f"{run.seconds / 60:.1f}",
            ]
        )
    output = (
        "<section><h2>Every seed, without selection</h2>"
        + table(
            [
                "Seed",
                "Filter scope",
                "Init monitor",
                "620 monitor",
                "1240 monitor",
                "Final held-out greedy",
                "Final random",
                "Train min",
            ],
            rows,
        )
        + "<p>Each score uses 100 balanced games. Brackets are checkpoint-conditional deal intervals. Held-out/random results use separate frozen deal families; they are not additional training replicates.</p>"
    )
    effect_rows = []
    for family, entry in summary["effects"].items():
        d = entry["actor_minus_actor_critic"]
        effect_rows.append(
            [
                family,
                ", ".join(f"{v * 100:+.1f}" for v in d["per_seed"]),
                f"{d['mean'] * 100:+.2f} [{d['lower'] * 100:+.2f}, {d['upper'] * 100:+.2f}]",
            ]
        )
    output += "<h2>Actor-only minus actor-and-critic</h2>" + table(
        [
            "Comparison",
            "Paired seed differences (points)",
            "Mean and 95% interval (points)",
        ],
        effect_rows,
    )
    periodic = []
    for monitor in bundle.monitors:
        if monitor.coordinates.updates in (0, 620, 1240):
            continue
        run = next(r for r in bundle.runs if r.id == monitor.run_id)
        periodic.append(
            [
                str(run.seed),
                run.regime.id,
                str(monitor.coordinates.updates),
                f"{monitor.coordinates.training_seconds:.1f}",
                checkpoint_rate(monitor.rows),
            ]
        )
    output += (
        "<details><summary>Additional hourly milestones</summary>"
        + table(
            ["Seed", "Scope", "Updates", "Training seconds", "Greedy score"], periodic
        )
        + "</details>"
    )
    output += f"<p>Actor-only consumed {summary['actor_cost_ratio']:.2f}× mean training wall time at the same 317,440 collected transitions per run. Common-cost and one-hour contrasts use earlier observed monitoring checkpoints, never interpolate final scores. They remain exploratory monitoring.</p>"
    cost_rows = [
        [
            str(r["seed"]),
            str(r["case"]),
            f"{r['actor_exposures']:,.0f}",
            f"{r['critic_exposures']:,.0f}",
            f"{r['collection_seconds']:.1f}",
            f"{r['learning_seconds']:.1f}",
            f"{r['export_seconds']:.1f}",
        ]
        for r in summary["exposures"]
    ]
    output += "<h2>Actual sample use and phase costs</h2>" + table(
        [
            "Seed",
            "Scope",
            "Actor exposures",
            "Critic exposures",
            "Collection s",
            "Learning s",
            "Export s",
        ],
        cost_rows,
    )
    diagnostic_rows = [
        [
            str(r["seed"]),
            str(r["case"]),
            f"{r['value_loss']:.4g}",
            f"{r['entropy']:.3f}",
            f"{r['collection_kl']:.4f}",
            f"{r['reference_kl']:.3f}",
            f"{r['gradient_cap_fraction']:.1%}",
        ]
        for r in summary["last_100_update_diagnostics"]
    ]
    output += (
        "<details><summary>Value, entropy, KL and gradient diagnostics</summary>"
        + table(
            [
                "Seed",
                "Scope",
                "Value loss",
                "Entropy",
                "Collection KL",
                "Reference KL",
                "Norm cap active",
            ],
            diagnostic_rows,
        )
    )
    output += "<p>Loss/entropy/KL average available values in the last 100 update records; norm-cap fraction uses all available updates. These describe optimized samples, which differ by scope. Lower value loss does not establish better value calibration. Full time series remain in the bundle and editable notebook.</p></details>"
    earlier = [
        [
            Path(path).parts[-4] if len(Path(path).parts) >= 4 else path,
            str(record.get("seed", "unavailable")),
            str(record.get("status", "unavailable")),
            str(record.get("error") or "none"),
            f"{float(record.get('seconds', 0)):.2f}",
        ]
        for path, record in bundle.retained.items()
        if Path(path).name == "run.json"
    ]
    output += "<details><summary>All earlier calibration attempts and allocation limits</summary>"
    output += table(
        ["Retained attempt", "Seed", "Status", "Error", "Recorded run seconds"], earlier
    )
    output += "<p>A preflight-only failed start has no TrainingRun or terminal-cause receipt. Two process-table PermissionErrors and the terminated calibration supervisor remain retained. Its first child completed 40 updates and four initialization games while its parent stayed stale/running. The successful continuation repeated calibration, not a scientific seed. The original monitoring estimate omitted six periodic checkpoints; final/random cost reserves omitted the six-checkpoint multiplier. Closeout stayed within the original absolute 24-hour deadline; no new allocation or training.</p></details>"
    return output + "</section>"


def monitoring_figures(bundle: Bundle) -> list[Figure]:
    """Keep periodic checkpoints visible even when their update indexes differ."""
    figures = []
    for axis, label in (
        ("training_seconds", "Training wall seconds"),
        ("learner_transitions", "Collected learner transitions"),
        ("optimizer_exposures", "Optimizer sample exposures (scope-dependent)"),
    ):
        fig = Figure(figsize=(10, 4), layout="constrained")
        ax = fig.subplots()
        for run in sorted(bundle.runs, key=lambda r: (r.regime.id, r.seed)):
            points = sorted(
                [m for m in bundle.monitors if m.run_id == run.id],
                key=lambda m: getattr(m.coordinates, axis),
            )
            ax.step(
                [getattr(m.coordinates, axis) for m in points],
                [m.score.mean for m in points],
                where="post",
                marker="o",
                label=f"{run.regime.id} / {run.seed}",
            )
        ax.set(xlabel=label, ylabel="Score vs greedy (monitoring)", ylim=(0, 1))
        ax.legend(ncol=2, fontsize=8)
        figures.append(fig)
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.final, args.prior, args.output)


if __name__ == "__main__":
    main()
