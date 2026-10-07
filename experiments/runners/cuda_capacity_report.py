"""Offline ETU-103 report from immutable hardware, TrainingRun and arena receipts.

Hardware cells retain short-run selection limits. Scientific contrasts require
all three paired seeds and all declared deal/seat/deck blocks; incomplete attempts
remain visible without a comparative estimate. The notebook is the editable view.
"""

import argparse
from html import escape
import json
from pathlib import Path
from statistics import median
from typing import Literal

from matplotlib import rc_context
from matplotlib.figure import Figure
import nbformat
import numpy as np
from pydantic import BaseModel

from experiments.runners.cuda_capacity import CapacityPlan
from experiments.runners.cuda_performance import ModelResult, Sweep
from manabot.arena.models import file_sha256
from manabot.remote.deploy import Receipt
from manabot.training.analysis import cost_comparison
from manabot.training.capacity_analysis import Measurement, threshold_crossing
from manabot.training.checkpoint_queue import Attempt
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRun
from manabot.training.monitor_evaluation import MonitorResult


class HardwareRow(BaseModel):
    card: str
    cell: str
    kind: str
    status: str
    batch: int
    streams: int
    capacity: str
    seconds: float | None
    inference_observations_per_second: float | None = None
    optimizer_observations_per_second: float | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    transitions: int | None = None
    exposures: int | None = None
    training_seconds: float | None = None
    collection_seconds: float | None = None
    learning_seconds: float | None = None
    export_seconds: float | None = None
    device: str | None = None
    memory_bytes: int | None = None
    attention_slots: int | None = None
    parameter_count: int | None = None
    path: str


def _model_row(row: HardwareRow, result_path: Path) -> None:
    model = ModelResult.model_validate_json(result_path.read_text())
    row.device, row.memory_bytes, row.attention_slots = (
        model.device,
        model.total_memory_bytes,
        model.attention_slots,
    )
    if model.architecture is not None:
        row.parameter_count = model.architecture.parameters.total
    for name, phase in (
        ("inference", model.inference),
        ("optimizer", model.optimizer),
    ):
        if phase is not None:
            rate = median(w.observations / w.seconds for w in phase.windows)
            if name == "inference":
                row.inference_observations_per_second = rate
            else:
                row.optimizer_observations_per_second = rate
            row.peak_allocated_bytes = max(
                row.peak_allocated_bytes or 0, phase.peak_allocated_bytes
            )
            row.peak_reserved_bytes = max(
                row.peak_reserved_bytes or 0, phase.peak_reserved_bytes
            )


def hardware(root: Path) -> list[HardwareRow]:
    rows: list[HardwareRow] = []
    for card in ("l4", "a40"):
        rental = root / f"calibration-{card}-sweep"
        returned = (
            rental / "bulk-return/evidence"
            if (rental / "bulk-return/receipt.json").exists()
            else rental / "evidence"
        )
        directory = returned / "performance"
        path = directory / "sweep.json"
        if not path.exists():
            continue
        sweep = Sweep.model_validate_json(path.read_text())
        attempts = {a.cell.name: a for a in sweep.attempts}
        for cell in sweep.expected_cells:
            attempt = attempts.get(cell.name)
            row = HardwareRow(
                card=card,
                cell=cell.name,
                kind=cell.kind,
                status=attempt.status if attempt else "unvisited",
                batch=cell.batch,
                streams=cell.streams,
                capacity=cell.capacity,
                seconds=attempt.seconds if attempt else 0,
                path=str(directory / cell.name),
            )
            result_path = directory / cell.name / "result.json"
            if cell.kind == "model" and result_path.exists():
                _model_row(row, result_path)
            if cell.kind == "loop":
                run_path = directory / cell.name / "run/run.json"
                if run_path.exists():
                    run = TrainingRun.model_validate_json(run_path.read_text())
                    row.transitions = sum(s.learner_transitions for s in run.stages)
                    row.exposures = sum(s.optimizer_exposures for s in run.stages)
                    row.training_seconds = run.seconds
                    row.collection_seconds = sum(
                        s.collection_seconds for s in run.stages
                    )
                    row.learning_seconds = sum(s.learning_seconds for s in run.stages)
                    row.export_seconds = sum(s.export_seconds for s in run.stages)
                if result_path.exists():
                    # Loop result is a small native JSON boundary; counters are
                    # independently authoritative in the typed TrainingRun above.
                    value = json.loads(result_path.read_text())
                    row.peak_allocated_bytes = int(value["peak_allocated_bytes"])
                    row.peak_reserved_bytes = int(value["peak_reserved_bytes"])
            rows.append(row)
    for card, supplemental in (
        ("a40", root / "calibration-a40-sweep/fit512/evidence"),
        ("l4", root / "calibration-live-workflow/evidence/fit512"),
    ):
        result_path = supplemental / "w384-d8-model-b512-s0/result.json"
        exit_path = supplemental / "exit.txt"
        if not exit_path.exists():
            continue
        status = "completed" if exit_path.read_text().strip() == "0" else "failed"
        log = supplemental / "worker.log"
        if (
            status == "failed"
            and log.exists()
            and "OutOfMemoryError" in log.read_text()
        ):
            status = "oom"
        row = HardwareRow(
            card=card,
            cell="w384-d8-model-b512-s0",
            kind="model",
            status=status,
            batch=512,
            streams=0,
            capacity="w384-d8",
            seconds=None,
            path=str(supplemental),
        )
        if result_path.exists():
            _model_row(row, result_path)
        rows.append(row)
    return rows


def hardware_comparison(rows: list[HardwareRow]) -> dict[str, object]:
    """Best observed short cells, plus matched cells; neither is strength."""
    best = []
    for card in ("l4", "a40"):
        for capacity in ("w64-d1", "w64-d2", "w128-d2", "w384-d8"):
            eligible = [
                r
                for r in rows
                if r.card == card
                and r.capacity == capacity
                and r.status in {"completed", "oom"}
            ]
            for phase in ("inference", "optimizer", "loop"):

                def rate(row: HardwareRow) -> float:
                    if phase == "inference":
                        return row.inference_observations_per_second or 0
                    if phase == "optimizer":
                        return row.optimizer_observations_per_second or 0
                    return (
                        row.transitions / row.training_seconds
                        if row.kind == "loop"
                        and row.status == "completed"
                        and row.transitions is not None
                        and row.training_seconds
                        else 0
                    )

                if eligible:
                    winner = max(eligible, key=rate)
                    if rate(winner) > 0:
                        best.append(
                            dict(
                                card=card,
                                capacity=capacity,
                                phase=phase,
                                cell=winner.cell,
                                rate=rate(winner),
                                exposures=winner.exposures,
                                seconds=winner.seconds,
                            )
                        )
    matched = []
    l4 = {r.cell: r for r in rows if r.card == "l4"}
    for right in rows:
        if right.card != "a40" or right.cell not in l4:
            continue
        left = l4[right.cell]
        matched.append(
            dict(
                cell=right.cell,
                l4=left.model_dump(mode="json"),
                a40=right.model_dump(mode="json"),
            )
        )
    return dict(
        best_observed=best,
        matched=matched,
        limits="Selection is among these short measured cells, not a global optimum. Loop rates include executor setup/export; collection/update clocks and actual exposures remain separate. Cards may have different host CPU/RAM and sampled workload trajectories.",
    )


def interval(
    values: np.ndarray, mode: Literal["seed", "deal", "joint"]
) -> dict[str, float]:
    rng = np.random.default_rng(1035106)
    estimates = []
    for _ in range(10000):
        seeds = (
            rng.integers(values.shape[0], size=values.shape[0])
            if mode != "deal"
            else np.arange(values.shape[0])
        )
        deals = (
            rng.integers(values.shape[1], size=values.shape[1])
            if mode != "seed"
            else np.arange(values.shape[1])
        )
        estimates.append(float(values[np.ix_(seeds, deals)].mean()))
    bounds = np.quantile(estimates, [0.025, 0.975])
    return dict(
        mean=float(values.mean()), lower=float(bounds[0]), upper=float(bounds[1])
    )


def scientific(out: Path) -> dict[str, object]:
    if not (out / "plan.json").exists():
        return dict(status="unlaunched")

    plan = CapacityPlan.model_validate_json((out / "plan.json").read_text())
    run_paths = list(out.glob("rental-*/evidence/run/run.json"))
    runs = {
        r.id: r
        for r in (TrainingRun.model_validate_json(p.read_text()) for p in run_paths)
    }
    results = [
        MonitorResult.model_validate_json(p.read_text())
        for p in out.glob("evaluation/attempt-*/evaluation/monitor.json")
    ]
    measurements: list[Measurement] = []
    attempts = [
        Attempt.model_validate_json(p.read_text())
        for p in out.glob("evaluation/attempt-*/attempt.json")
    ]
    complete = (
        len(attempts) == 24
        and all(a.status == "completed" for a in attempts)
        and len(results) == 24
        and len(runs) == 6
        and all(r.status == "completed" for r in results)
        and all(r.status == "completed" for r in runs.values())
    )
    for result in results:
        run = runs.get(result.run_id)
        if run is None:
            continue
        c = result.coordinates
        phase = (
            "endpoint" if result.purpose == "frozen-study-evaluation" else "development"
        )
        measurements.append(
            Measurement(
                regime=run.regime.id,
                seed=run.seed,
                phase=phase,
                cutoff=0 if c.updates == 0 else (1 if c.stage_id == "policy-0" else 2),
                opponent=result.protocol.opponent,
                checkpoint=result.artifact,
                training_seconds=c.training_seconds,
                decisions=c.environment_decisions,
                optimizer_exposures=c.optimizer_exposures,
                score=result.score.mean if result.score else None,
                complete=result.status == "completed",
            )
        )
    development = [
        m
        for m in measurements
        if m.opponent == "scripted_greedy" and m.phase == "development"
    ]
    crossings = [
        threshold_crossing(
            [p for p in development if (p.regime, p.seed) == key], 600, 0.5
        ).model_dump(mode="json")
        for key in sorted({(m.regime, m.seed) for m in development})
    ]
    curves = {}
    if complete:
        for axis in ("training_seconds", "decisions", "optimizer_exposures"):
            mapped = [
                m.model_dump(mode="json") | {"training_seconds": getattr(m, axis)}
                for m in development
            ]
            curves[axis] = cost_comparison(
                mapped,
                "scripted_greedy",
                end_seconds=600 if axis == "training_seconds" else None,
            )
    endpoint = {}
    if complete:
        scores = {}
        for result in results:
            if result.purpose != "frozen-study-evaluation":
                continue
            run = runs[result.run_id]
            scores[(run.regime.id, run.seed)] = [
                np.mean([r.score_a for r in result.rows if r.deal_seed == deal])
                for deal in plan.protocol.endpoint_anchor_deals
            ]
        difference = np.array(
            [
                np.array(scores[("cuda-w384-d8", seed)])
                - np.array(scores[("cuda-w64-d2", seed)])
                for seed in plan.protocol.training_seeds
            ]
        )
        endpoint = {
            "difference_large_minus_small": {
                mode: interval(difference, mode) for mode in ("seed", "deal", "joint")
            },
            "paired_seed_differences": difference.mean(1).tolist(),
        }
    return dict(
        status="complete" if complete else "incomplete",
        measurements=[m.model_dump(mode="json") for m in measurements],
        crossings=crossings,
        common_support=curves,
        endpoint=endpoint,
        decision="No automatic promotion. Three-seed evidence and one greedy opponent cannot alone establish general challenger strength.",
        limits="One-pass CPU inference fixes the rule, not equal FLOPs or latency. Early area uses only the observed shared development range; final deals are separate. Threshold hits are interval-censored scheduled observations; misses are right-censored. Filtered exposures are post-treatment quantities. Seed, deal and joint percentile intervals retain whole four-leg blocks.",
    )


def build_report(root: Path, scientific_out: Path | None = None) -> Path:
    root = root.resolve()
    out = scientific_out.resolve() if scientific_out else root / "comparison"
    rows = hardware(root)
    receipts = [
        Receipt.model_validate_json(p.read_text())
        for p in root.glob("calibration-*/deployment.json")
    ]
    evidence = {
        str(p): file_sha256(p) for p in root.glob("calibration-*/deployment.json")
    }
    for row in rows:
        directory = Path(row.path)
        for name in ("result.json", "run/run.json", "worker.log", "exit.txt"):
            path = directory / name
            if path.is_file():
                evidence[str(path)] = file_sha256(path)
    for path in root.glob("calibration-*/**/bundle.json"):
        evidence[str(path)] = file_sha256(path)
    model_protocols = []
    for row in rows:
        result = Path(row.path) / "result.json"
        if row.kind == "model" and result.exists():
            model = ModelResult.model_validate_json(result.read_text())
            model_protocols.append(
                {
                    "card": row.card,
                    "capacity": row.capacity,
                    **model.model_dump(
                        exclude={"inference", "optimizer", "architecture"}
                    ),
                }
            )
    science = scientific(out)
    report = dict(
        model_protocols=model_protocols,
        hardware=[r.model_dump(mode="json") for r in rows],
        hardware_comparison=hardware_comparison(rows),
        calibration_receipts=[r.model_dump(mode="json") for r in receipts],
        science=science,
        evidence=evidence,
        launch_boundary="Scientific comparison not started: ETU-123 disconnect-safe lifecycle requires integration; final live-workflow proof timed out during guardian bootstrap before training. L4 batch512 and collection profiler remain unmeasured."
        if science["status"] == "unlaunched"
        else "See the retained campaign receipt for execution status.",
        shared_budget={
            "ceiling_dollars": 15,
            "etu123_reserved_dollars": 3,
            "etu103_ceiling_dollars": 12,
        },
    )
    atomic_json(root / "capacity-report.json", report)
    return render_report(root / "capacity-report.json")


def render_report(source: Path) -> Path:
    """Render a portable compact extract without requiring original model bytes."""
    root = source.resolve().parent
    report = json.loads(source.read_text())
    rows = [HardwareRow.model_validate(row) for row in report["hardware"]]
    receipts = [Receipt.model_validate(row) for row in report["calibration_receipts"]]
    table = "<table><tr><th>Card / capacity</th><th>Kind / batch / streams</th><th>Status</th><th>Inference obs/s</th><th>Adam obs/s</th><th>Loop transitions/s</th><th>Peak allocated GiB</th></tr>"
    for r in rows:
        rate = (
            r.transitions / r.training_seconds
            if r.transitions is not None and r.training_seconds
            else None
        )

        def number(v: float | None) -> str:
            return "—" if v is None else f"{v:,.1f}"

        table += f"<tr><td>{escape(r.card + ' / ' + r.capacity)}</td><td>{r.kind} / {r.batch} / {r.streams}</td><td>{escape(r.status)}</td><td>{number(r.inference_observations_per_second)}</td><td>{number(r.optimizer_observations_per_second)}</td><td>{number(rate)}</td><td>{number(r.peak_allocated_bytes / 1024**3 if r.peak_allocated_bytes is not None else None)}</td></tr>"
    table += "</table>"
    fig = Figure(figsize=(9, 4))
    ax = fig.subplots()
    for card in ("l4", "a40"):
        for capacity in ("w64-d2", "w384-d8"):
            points = [
                r
                for r in rows
                if r.card == card
                and r.capacity == capacity
                and r.kind == "model"
                and r.inference_observations_per_second is not None
            ]
            if points:
                points.sort(key=lambda point: point.batch)
                ax.plot(
                    [p.batch for p in points],
                    [p.inference_observations_per_second for p in points],
                    marker="o",
                    label=f"{card} {capacity}",
                )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Fixed real batch rows",
        ylabel="Inference observations/s",
    )
    ax.legend()
    fig.tight_layout()
    with rc_context({"svg.hashsalt": "etu103-cuda-capacity"}):
        fig.savefig(root / "hardware-throughput.svg", metadata={"Date": None})
    dollars = sum(r.estimated_dollars or 0 for r in receipts)
    summary = "<table><tr><th>Card / capacity / phase</th><th>Best observed cell</th><th>Rows or transitions/s</th></tr>"
    for item in report["hardware_comparison"]["best_observed"]:
        summary += f"<tr><td>{escape(item['card'] + ' / ' + item['capacity'] + ' / ' + item['phase'])}</td><td>{escape(item['cell'])}</td><td>{item['rate']:,.1f}</td></tr>"
    summary += "</table>"
    ledger = "<table><tr><th>Attempt</th><th>Outcome</th><th>Deleted</th><th>Estimated dollars</th></tr>"
    for receipt in receipts:
        ledger += f"<tr><td>{escape(receipt.plan_sha256[:12])}</td><td>{escape(receipt.error or ('complete' if receipt.complete else 'incomplete'))}</td><td>{escape(receipt.phase)}</td><td>{receipt.estimated_dollars or 0:.5f}</td></tr>"
    ledger += "</table>"
    body = f"""<!doctype html><meta charset="utf-8"><title>ETU-103 capacity</title><style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:6px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}pre{{white-space:pre-wrap}}img{{width:100%}}</style><h1>CUDA capacity: hardware and learning</h1><p>Calibration rental estimate: ${dollars:.4f}. Provider billing may differ; intent-to-confirmed-deletion time includes setup, idle and storage allowance. Every failure and unvisited cell remains below.</p><p>Model-only Adam is a diagnostic loss on fixed real rows. Peaks cover completed phases; OOM logs retain failed-allocation details. Complete-loop cells use ordinary Ataraxos self-play. Three-update cells and three one-second windows measure short-run behavior, not strength or sustained throughput. Collection includes engine and inference; its share alone does not prove GPU starvation.</p><img src="hardware-throughput.svg" alt="Fixed real batch inference throughput"><h2>Best observed configurations</h2><p>Separate maxima over the declared short grid, not sustained operating points. Compare matched cells in the JSON extract before attributing differences to hardware.</p>{summary}<details><summary>All hardware cells</summary>{table}</details><h2>All rental attempts</h2>{ledger}<p>{escape(report.get("launch_boundary", ""))}</p><p>$3 remains reserved for ETU-123 within the same $15 ceiling.</p><h2>Learning cohort</h2><p>{escape(str(report["science"]["status"]))}</p><details><summary>Resolved measurements, uncertainty and limits</summary><pre>{escape(json.dumps(report["science"], indent=2))}</pre></details>"""
    target = root / "report.html"
    target.write_text(body)
    return target


def write_notebook(root: Path) -> Path:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# CUDA capacity\nEdit the evidence path and analysis cells; this notebook generates the read-only HTML report. Hardware speed, fit and playing strength are separate outcomes."
            ),
            nbformat.v4.new_code_cell(
                f"from pathlib import Path\nimport sys\nrepo = Path({str(Path.cwd())!r})\nsys.path.insert(0, str(repo))\nfrom experiments.runners.cuda_capacity_report import render_report\nroot = Path({str(root.resolve())!r})\nreport = render_report(root / 'capacity-report.json')\nreport"
            ),
            nbformat.v4.new_code_cell(
                "import json\nimport pandas as pd\nevidence = json.loads((root / 'capacity-report.json').read_text())\nframe = pd.DataFrame(evidence['hardware'])\nframe"
            ),
        ]
    )
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    path = root / "report.ipynb"
    nbformat.write(notebook, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--science", type=Path)
    parser.add_argument("--notebook", action="store_true")
    args = parser.parse_args()
    build_report(args.root, args.science)
    if args.notebook:
        write_notebook(args.root)


if __name__ == "__main__":
    main()
