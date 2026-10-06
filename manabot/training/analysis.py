"""Offline analysis of retained training and arena evidence."""

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.runners.training_protocol import EvaluationProtocol
from manabot.arena.models import file_sha256
from manabot.training import capacity_analysis
from manabot.training.models import CollectSelection, TrainingRun
from manabot.training.selection_analysis import (
    analyze_selection,
    write_selection_report,
)
from manabot.training.selection_data import SelectionDataset

from .execution import atomic_json


def valid_game(row):
    """Only authoritative terminal outcomes with exact replay can be scored."""
    return (
        row["failure"] is None
        and row["terminated"]
        and not row["truncated"]
        and row["replay_passed"]
    )


def cost_comparison(
    rows: list[dict[str, Any]],
    anchor: str = "random-smoke-anchor",
    *,
    end_seconds: float | None = None,
) -> dict[str, Any]:
    # Existing study JSON contains heterogeneous checkpoint/metric values.

    """Compare only observed checkpoints at a shared cost; never backfill."""
    groups = {}
    for row in rows:
        if (
            row["opponent"] == anchor
            and row.get("phase", "development") == "development"
            and row["complete"]
            and row["score"] is not None
        ):
            groups.setdefault(
                (row["regime"], row["seed"], row.get("variant", "raw")), []
            ).append(row)
    if not groups:
        return {"status": "unavailable", "reason": "no complete anchor measurements"}
    groups = {
        key: sorted(points, key=lambda r: r["training_seconds"])
        for key, points in groups.items()
    }
    start = max(points[0]["training_seconds"] for points in groups.values())
    end = min(points[-1]["training_seconds"] for points in groups.values())
    if end_seconds is not None:
        end = min(end, end_seconds)
    if end < start:
        return {
            "status": "unavailable",
            "reason": "observed cost ranges do not overlap",
        }
    results = []
    for (regime, seed, variant), points in sorted(groups.items()):
        # Each checkpoint supplies the score until the next observed checkpoint.
        available = [p for p in points if p["training_seconds"] <= end]
        selected = available[-1]
        area = 0.0
        for point, following in zip(available, available[1:]):
            left = max(start, point["training_seconds"])
            right = following["training_seconds"]
            area += max(0, right - left) * point["score"]
        area += (end - max(start, selected["training_seconds"])) * selected["score"]
        results.append(
            dict(
                regime=regime,
                seed=seed,
                variant=variant,
                score=selected["score"],
                checkpoint=selected["checkpoint"],
                checkpoint_seconds=selected["training_seconds"],
                mean_score=area / (end - start) if end > start else None,
            )
        )
    return dict(status="available", start_seconds=start, end_seconds=end, rows=results)


def cost_cutoffs(rows, cutoffs, anchor="random-smoke-anchor"):
    """Frozen cutoffs use only earlier checkpoints within observed support."""
    comparison = cost_comparison(rows, anchor)
    results = []
    for cutoff in cutoffs:
        if (
            comparison["status"] != "available"
            or not comparison["start_seconds"] <= cutoff <= comparison["end_seconds"]
        ):
            results.append(
                dict(
                    cutoff=cutoff,
                    status="unavailable",
                    reason="outside shared observed cost support",
                )
            )
            continue
        selected = []
        for item in comparison["rows"]:
            eligible = [
                r
                for r in rows
                if r["opponent"] == anchor
                and r.get("phase", "development") == "development"
                and r["complete"]
                and r["score"] is not None
                and r["regime"] == item["regime"]
                and r["seed"] == item["seed"]
                and r.get("variant", "raw") == item["variant"]
                and r["training_seconds"] <= cutoff
            ]
            selected.append(max(eligible, key=lambda r: r["training_seconds"]))
        results.append(dict(cutoff=cutoff, status="available", rows=selected))
    return results


def verify_saved_inputs(out, study):
    """Reject altered frozen protocols, run exports and measured checkpoints."""
    from manabot.arena.models import canonical_sha256

    protocol_path = out / "protocol.json"
    if "protocol_sha256" in study:
        protocol = json.loads(protocol_path.read_text())
        if canonical_sha256(protocol) != study["protocol_sha256"]:
            raise ValueError("protocol digest mismatch")
        recipes = json.loads((out / "recipes.json").read_text())
        if [canonical_sha256(r) for r in recipes] != protocol["regime_digests"]:
            raise ValueError("resolved recipe digest mismatch")
    for item in study["runs"]:
        path = Path(item["path"])
        if (
            "sha256" in item
            and hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError("run export digest mismatch")
    for cell in study.get("comparisons", []):
        trace = cell.get("trace")
        if (
            trace
            and hashlib.sha256(Path(trace["path"]).read_bytes()).hexdigest()
            != trace["sha256"]
        ):
            raise ValueError("arena trace digest mismatch")
    for row in study["measurements"]:
        artifact = row["checkpoint"]
        if (
            hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest()
            != artifact["sha256"]
        ):
            raise ValueError("checkpoint digest mismatch")


def paired_uncertainty(comparisons):
    """Bootstrap training seeds and common four-leg deals, never individual games."""
    import numpy as np

    groups = {}
    for cell in comparisons:
        groups.setdefault(
            (
                cell["a"],
                cell["b"],
                cell["cutoff"],
                cell.get("phase", "development"),
                cell.get("variant", "raw"),
            ),
            [],
        ).append(cell)
    results = []
    for (a, b, cutoff, phase, variant), cells in sorted(groups.items()):
        result = dict(
            a=a, b=b, cutoff=cutoff, phase=phase, variant=variant, status="unavailable"
        )
        results.append(result)
        seeds = {c.get("training_seed") for c in cells}
        if None in seeds or len(seeds) < 3:
            result["reason"] = "requires at least three distinct training seeds"
            continue
        if any(
            not c["replay"]["passed"]
            or len(c["rows"]) != c["scheduled_games"]
            or not all(valid_game(r) for r in c["rows"])
            for c in cells
        ):
            result["reason"] = "incomplete cohort"
            continue
        blocks = {}
        for cell in cells:
            deals = {}
            for row in cell["rows"]:
                deals.setdefault(row["deal_seed"], []).append(row["score_a"])
            key = (cell["training_seed"], cell.get("b_training_seed"))
            if key in blocks:
                raise ValueError("duplicate training seed cell")
            blocks[key] = deals
        deals = sorted(next(iter(blocks.values())))
        if any(
            set(block) != set(deals) or any(len(v) != 4 for v in block.values())
            for block in blocks.values()
        ):
            result["reason"] = "unmatched four-leg deal blocks"
            continue
        a_seeds = sorted(seeds)
        b_seeds = {key[1] for key in blocks}
        crossed = None not in b_seeds and len(blocks) == len(a_seeds) * len(b_seeds)
        if crossed:
            b_seeds = sorted(b_seeds)
            scores = np.array(
                [
                    [[np.mean(blocks[(sa, sb)][d]) for d in deals] for sb in b_seeds]
                    for sa in a_seeds
                ]
            )
        elif len(blocks) == len(a_seeds) and (
            b_seeds == {None} or all(sa == sb for sa, sb in blocks)
        ):
            scores = np.array(
                [[np.mean(blocks[key][d]) for d in deals] for key in sorted(blocks)]
            )
        else:
            result["reason"] = "seed schedule is neither complete crossed nor paired"
            continue
        rng = np.random.default_rng(0)
        estimates = []
        for _ in range(2000):
            indexes = [rng.integers(n, size=n) for n in scores.shape]
            estimates.append(float(scores[np.ix_(*indexes)].mean()))
        result.update(
            status="available",
            score_a=float(scores.mean()),
            interval_95=np.quantile(estimates, [0.025, 0.975]).tolist(),
            training_seeds=len(seeds),
            deal_blocks=len(deals),
            method=("crossed" if crossed else "paired")
            + " training-seed/common-deal bootstrap; descriptive at small seed count",
        )
    return results


def report(out: Path | str) -> None:
    from nbclient import NotebookClient
    import nbformat

    out = Path(out).resolve()
    study = json.loads((out / "study.json").read_text())
    verify_saved_inputs(out, study)
    rows = study["measurements"]
    comparison = (
        cost_comparison(
            rows,
            "scripted-greedy-fixed-anchor"
            if study["study"]
            in {"value-token-screen", "pooling-filter", "history-input"}
            else "random-smoke-anchor",
        )
        if study["status"] == "completed"
        else {"status": "unavailable", "reason": "study cohort incomplete"}
    )
    atomic_json(out / "cost-comparison.json", comparison)
    protocol_path = out / "protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    anchors = sorted({r["opponent"] for r in rows if r["opponent"].endswith("-anchor")})
    atomic_json(
        out / "anchor-cost-comparisons.json",
        {
            anchor: cost_comparison(rows, anchor)
            if study["status"] == "completed"
            else comparison
            for anchor in anchors
        },
    )
    atomic_json(
        out / "cost-cutoffs.json",
        {
            anchor: cost_cutoffs(rows, protocol.get("cost_cutoffs_seconds", ()), anchor)
            if study["status"] == "completed"
            else []
            for anchor in anchors
        },
    )
    uncertainty = paired_uncertainty(study["comparisons"])
    atomic_json(out / "uncertainty.json", uncertainty)
    lines = [
        f"# {study['study']} — {study['profile']}",
        "",
        f"Status: {study['status']}. {study['limits']}",
        "",
        "Playing score is measured against the named opponent. Cost curves use the declared anchor (scripted for screening, random otherwise); paired-recipe matches are listed separately. Smoke points prove execution only. Scientific profiles report every seed separately; three seeds provide only exploratory uncertainty, not a confirmatory method claim.",
        "",
        "| Recipe | Seed | Variant | Phase | Cutoff | Opponent | Training seconds | Decisions | Complete games | Score |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        score = "unavailable" if row["score"] is None else f"{row['score']:.3f}"
        lines.append(
            f"| {row['regime']} | {row['seed']} | {row.get('variant', 'raw')} | {row.get('phase', 'development')} | {row['cutoff']} | {row['opponent']} | {row['training_seconds']:.2f} | {row['decisions']} | {row['games']} | {score} |"
        )
    if study["study"] in {
        "value-models",
        "value-token-screen",
        "pooling-filter",
        "history-input",
    }:
        lines += [
            "",
            "Value-model disposition: exploratory evidence only; no automatic promotion. The token "
            "changes the shared policy representation; one CPU thread is not equal inference cost.",
            "",
            "Resolved configurations: [recipes](recipes.json), [protocol](protocol.json), "
            "[allocation](resolved-plan.json). Every run and failed attempt remains in study.json.",
            "",
            "| Recipe | Stage | Status | Learner transitions | Native steps | Optimizer exposures | Collection s | Learning s | Export s |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for entry in study["runs"]:
            if not Path(entry["path"]).exists():
                lines.append(f"Run export unavailable: {entry['path']}")
                continue
            run = TrainingRun.model_validate_json(Path(entry["path"]).read_text())
            for stage in run.stages:
                lines.append(
                    f"| {run.regime.id} | {stage.id} | {stage.status} | "
                    f"{stage.learner_transitions} | {stage.environment_decisions} | "
                    f"{stage.optimizer_exposures} | {stage.collection_seconds:.3f} | "
                    f"{stage.learning_seconds:.3f} | {stage.export_seconds:.3f} |"
                )
        lines += [
            "",
            "Per-player inference latency is retained in each arena game row. "
            "Parameter counts are in the arena registrations; host RSS/CPU observations "
            "are in run exports. These do not correct concurrent host contention.",
        ]
    if study["study"] == "compound-decisions":
        lines += [
            "",
            "Compound accounting (training; forced factors and native automatic steps remain distinct):",
            "",
            "| Recipe | Seed | Checkpoint | Groups | Native microchoices | Decoder factors | Forced factors | Auto-resolved | Policy seconds | Max group latency seconds |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        seen = set()
        for row in rows:
            key = (row["regime"], row["seed"], row["checkpoint"]["sha256"])
            if key in seen:
                continue
            seen.add(key)
            statistics = [
                item for stage in row.get("compound_accounting", []) for item in stage
            ]
            sums = {
                name: sum(item[name] for item in statistics)
                for name in (
                    "decisions",
                    "microchoices",
                    "factors",
                    "forced_factors",
                    "auto_resolved",
                    "decision_seconds",
                )
            }
            latency = max(
                (item["max_decision_seconds"] for item in statistics), default=0
            )
            lines.append(
                f"| {row['regime']} | {row['seed']} | {row['cutoff']} | {sums['decisions']} | {sums['microchoices']} | {sums['factors']} | {sums['forced_factors']} | {sums['auto_resolved']} | {sums['decision_seconds']:.3f} | {latency:.6f} |"
            )
        lines += [
            "",
            "Grouping and estimator effects remain unproven by smoke. Blocker/payment prompts remain separate engine decisions. The sequential arm uses conditional decoder factors, not the historical flat policy.",
        ]
    lines += [
        "",
        "Equal-cost comparison uses the last available checkpoint, without interpolation.",
    ]
    if comparison["status"] == "available":
        lines += [
            "",
            f"Shared observed window: {comparison['start_seconds']:.2f}–{comparison['end_seconds']:.2f} seconds.",
            "",
            "| Recipe | Seed | Variant | Checkpoint seconds | Score at common horizon | Mean score over shared window |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
        for row in comparison["rows"]:
            mean = (
                "unavailable"
                if row["mean_score"] is None
                else f"{row['mean_score']:.3f}"
            )
            lines.append(
                f"| {row['regime']} | {row['seed']} | {row['variant']} | {row['checkpoint_seconds']:.2f} | {row['score']:.3f} | {mean} |"
            )
    else:
        lines += ["", f"Unavailable: {comparison['reason']}."]
    if not rows:
        lines += ["", "No completed measurements."]
    lines += [
        "",
        "Paired seed/deal uncertainty (checkpoint index comparisons are not equal-cost claims):",
    ]
    for item in uncertainty:
        lines += [
            "",
            f"{item['a']} / {item['b']} {item['variant']} {item['phase']} cutoff {item['cutoff']}: "
            + (
                f"A score {item['score_a']:.3f}, descriptive 95% interval {item['interval_95']}; {item['method']}."
                if item["status"] == "available"
                else item["reason"] + "."
            ),
        ]
    if (
        study.get("study") == "omitted-controls"
        or protocol.get("study") == "omitted-controls"
    ):
        lines += [
            "",
            "Technique disposition: unresolved. Raw/EMA are correlated outputs of each training seed; they never add training replicates.",
        ]
    selection_rows = []
    for entry in study["runs"]:
        if not Path(entry["path"]).exists():
            continue
        run = json.loads(Path(entry["path"]).read_text())
        for stage in run["stages"]:
            for iteration, diagnostic in enumerate(stage["diagnostics"]):
                for group in diagnostic.get("selection_groups", []):
                    selection_rows.append(
                        dict(
                            run=run["id"],
                            seed=run["seed"],
                            stage=stage["id"],
                            iteration=iteration,
                            **group,
                        )
                    )
    atomic_json(out / "selection-diagnostics.json", selection_rows)
    lines += [
        "",
        "[Raw/retained selection diagnostics](selection-diagnostics.json): signed advantage quartiles and lambda-target residuals by action type and observed terminal distance. Censored tails remain unknown. Residual association does not establish critic error causality.",
    ]
    if study.get("error"):
        lines += ["", study["error"]]
    lines += [
        "",
        "Evaluation seconds: "
        + str(sum(c["evaluation_seconds"] for c in study["comparisons"])),
        "",
        "Not established by this report: confirmatory strength, S1–S5 current-world rebinding, human play, compound-action strength, belief-guided search, update distillation, belief sampling and exploiters. See the experiment protocols before funding those runs.",
        "",
        "Traces and per-game failures:",
    ]
    for cell in study["comparisons"]:
        trace = cell.get("trace")
        trace_link = (
            f"[trace]({trace['path']})"
            if trace
            else f"Partial artifacts: {cell.get('path', 'unavailable')}"
        )
        lines += [
            "",
            f"{cell['a']} / {cell['b']} cutoff {cell['cutoff']}: {trace_link}; exact replay {cell['replay']['passed']}.",
            "",
            "| Deal | Leg | A seat | Seat decks | A score | Failure |",
            "| ---: | ---: | ---: | --- | ---: | --- |",
        ]
        valid_scores = [r["score_a"] for r in cell["rows"] if valid_game(r)]
        scheduled = cell.get("scheduled_games", len(cell["rows"]))
        if scheduled and len(valid_scores) < scheduled:
            lower = sum(valid_scores) / scheduled
            upper = (sum(valid_scores) + scheduled - len(valid_scores)) / scheduled
            lines += [
                "",
                f"Unresolved-game bounds for A: [{lower:.3f}, {upper:.3f}]; incomplete cohort, no strength claim.",
                "",
            ]
        for row in cell["rows"]:
            score = row["score_a"] if valid_game(row) else "unavailable"
            lines.append(
                f"| {row['deal_seed']} | {row['leg']} | {row['player_a_seat']} | {row.get('seat_decks', [])} | {score} | {row['failure']} |"
            )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    atomic_json(out / "metrics.json", rows)
    if study["study"] == "model-capacity":
        recipes = json.loads((out / "recipes.json").read_text())
        with (out / "report.md").open("a") as stream:
            stream.write(
                f"\nTotal retained study seconds (including evaluation): {study.get('seconds', 'unavailable')}.\n"
            )
        capacity_analysis.write_capacity_report(
            out,
            [capacity_analysis.Measurement.model_validate(r) for r in rows],
            EvaluationProtocol.model_validate(protocol),
            tuple(r["id"] for r in recipes),
            study["status"] == "completed",
        )

    notebook = nbformat.read(
        Path(__file__).resolve().parents[2]
        / "experiments/study/training-regimes.ipynb",
        as_version=4,
    )
    notebook.cells.insert(
        0, nbformat.v4.new_code_cell(f"study_path = {str(out / 'study.json')!r}")
    )
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(out)}},
    ).execute()
    nbformat.write(notebook, out / "analysis.ipynb")


def report_selection_run(run_path: Path, stage_id: str, out: Path) -> None:
    """Regenerate a complete selection report from content-checked run artifacts.

    No execution state is created or modified; the JSON is an export of the
    existing VerifyStore run. Incomplete cohorts and changed evidence fail closed.
    """
    run = TrainingRun.model_validate_json(run_path.read_text())
    stage = next(s for s in run.regime.stages if s.id == stage_id)
    record = next(s for s in run.stages if s.id == stage_id)
    if not isinstance(stage, CollectSelection) or record.status != "completed":
        raise ValueError("selection report requires a completed collection stage")
    for artifact in record.artifacts.values():
        if file_sha256(artifact["path"]) != artifact["sha256"]:
            raise ValueError("selection artifact digest mismatch")
    dataset = SelectionDataset.model_validate_json(
        Path(record.artifacts["dataset"]["path"]).read_text()
    )
    if (
        dataset.run_id != run.id
        or dataset.stage_id != stage_id
        or dataset.policy_sha256 != record.inputs["policy"]["sha256"]
        or dataset.policy_run_id != (stage.source_run or run.id)
        or dataset.policy_stage_id != stage.policy
        or dataset.weights != stage.weights
    ):
        raise ValueError("selection dataset provenance differs from run")
    report = analyze_selection(dataset, stage)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "analysis.json", report.model_dump(mode="json"))
    write_selection_report(report, out / "report.md")
