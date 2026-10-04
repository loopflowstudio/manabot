"""Offline analysis of retained training and arena evidence."""

import hashlib
import json
from pathlib import Path

from .execution import atomic_json


def cost_comparison(rows):
    """Compare only observed checkpoints at a shared cost; never backfill."""
    groups = {}
    for row in rows:
        if (
            row["opponent"] == "random-smoke-anchor"
            and row["complete"]
            and row["score"] is not None
        ):
            groups.setdefault((row["regime"], row["seed"]), []).append(row)
    if not groups:
        return {"status": "unavailable", "reason": "no complete anchor measurements"}
    groups = {
        key: sorted(points, key=lambda r: r["training_seconds"])
        for key, points in groups.items()
    }
    start = max(points[0]["training_seconds"] for points in groups.values())
    end = min(points[-1]["training_seconds"] for points in groups.values())
    if end < start:
        return {
            "status": "unavailable",
            "reason": "observed cost ranges do not overlap",
        }
    results = []
    for (regime, seed), points in sorted(groups.items()):
        selected = max(
            (p for p in points if p["training_seconds"] <= end),
            key=lambda p: p["training_seconds"],
        )
        boundaries = (
            [start]
            + [
                p["training_seconds"]
                for p in points
                if start < p["training_seconds"] < end
            ]
            + [end]
        )
        area = sum(
            (right - left)
            * max(
                (p for p in points if p["training_seconds"] <= left),
                key=lambda p: p["training_seconds"],
            )["score"]
            for left, right in zip(boundaries, boundaries[1:])
        )
        results.append(
            dict(
                regime=regime,
                seed=seed,
                score=selected["score"],
                checkpoint=selected["checkpoint"],
                checkpoint_seconds=selected["training_seconds"],
                mean_score=area / (end - start) if end > start else None,
            )
        )
    return dict(status="available", start_seconds=start, end_seconds=end, rows=results)


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
    for row in study["measurements"]:
        artifact = row["checkpoint"]
        if (
            hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest()
            != artifact["sha256"]
        ):
            raise ValueError("checkpoint digest mismatch")


def report(out):
    from nbclient import NotebookClient
    import nbformat

    out = Path(out).resolve()
    study = json.loads((out / "study.json").read_text())
    verify_saved_inputs(out, study)
    rows = study["measurements"]
    comparison = (
        cost_comparison(rows)
        if study["status"] == "completed"
        else {"status": "unavailable", "reason": "study cohort incomplete"}
    )
    atomic_json(out / "cost-comparison.json", comparison)
    lines = [
        f"# {study['study']} — {study['profile']}",
        "",
        f"Status: {study['status']}. {study['limits']}",
        "",
        "Playing score is measured against the named opponent. Cost curves use the fixed random anchor; paired-recipe matches are listed separately. The smoke has one training seed and one deal block: cross-seed uncertainty is unavailable. These points prove execution, not learning-speed superiority.",
        "",
        "| Recipe | Cutoff | Opponent | Training seconds | Decisions | Complete games | Score |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        score = "unavailable" if row["score"] is None else f"{row['score']:.3f}"
        lines.append(
            f"| {row['regime']} | {row['cutoff']} | {row['opponent']} | {row['training_seconds']:.2f} | {row['decisions']} | {row['games']} | {score} |"
        )
    lines += [
        "",
        "Equal-cost comparison uses the last available checkpoint, without interpolation.",
    ]
    if comparison["status"] == "available":
        lines += [
            "",
            f"Shared observed window: {comparison['start_seconds']:.2f}–{comparison['end_seconds']:.2f} seconds.",
            "",
            "| Recipe | Seed | Checkpoint seconds | Score at common horizon | Mean score over shared window |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in comparison["rows"]:
            mean = (
                "unavailable"
                if row["mean_score"] is None
                else f"{row['mean_score']:.3f}"
            )
            lines.append(
                f"| {row['regime']} | {row['seed']} | {row['checkpoint_seconds']:.2f} | {row['score']:.3f} | {mean} |"
            )
    else:
        lines += ["", f"Unavailable: {comparison['reason']}."]
    if not rows:
        lines += ["", "No completed measurements."]
    if study.get("error"):
        lines += ["", study["error"]]
    lines += [
        "",
        "Evaluation seconds: "
        + str(sum(c["evaluation_seconds"] for c in study["comparisons"])),
        "",
        "Not run: independent-seed inference, confirmatory scoring, S1–S5 current-world rebinding, human play, raw/EMA arena comparison, compound actions, belief-guided search, update distillation, belief sampling and exploiters. See the experiment protocols before funding those runs.",
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
        valid_scores = [
            r["score_a"]
            for r in cell["rows"]
            if r["failure"] is None and r["terminated"] and r["replay_passed"]
        ]
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
            score = (
                row["score_a"]
                if row["failure"] is None and row["terminated"]
                else "unavailable"
            )
            lines.append(
                f"| {row['deal_seed']} | {row['leg']} | {row['player_a_seat']} | {row.get('seat_decks', [])} | {score} | {row['failure']} |"
            )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    atomic_json(out / "metrics.json", rows)
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
