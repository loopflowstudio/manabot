"""Render report tables from real short-screen evidence, clearly labeled as such."""

import csv
from datetime import UTC, datetime
import io
import json
from pathlib import Path

from learning_resources import CellResources, digest
from learning_selection import Point
from render_learning_report import checkpoint_table, endpoint_table, role_csv

from manabot.training.monitor_evaluation import MonitorResult

ROOT = Path(".runs/grid-007")
OUT = ROOT / "learning-017/report-rendering-check"


def main() -> None:
    analysis_path = ROOT / "early-learning-013-added-analysis/results.json"
    analysis = json.loads(analysis_path.read_text())
    rows = {(r["cell"], r["anchor"]): r for r in analysis["results"]}
    resource_path = ROOT / "learning-017/resource-readiness.json"
    resource_receipt = json.loads(resource_path.read_text())
    resources = [CellResources.model_validate(r) for r in resource_receipt["resources"]]
    sources = {str(p): digest(p) for p in (analysis_path, resource_path)}
    points: list[Point] = []
    for path in sorted(
        (ROOT / "early-learning-013-added/panels").glob("*/transitions-*/monitor.json")
    ):
        result = MonitorResult.model_validate_json(path.read_text())
        if result.status != "completed" or result.score is None:
            raise ValueError("Earlier screen has an incomplete monitor")
        cell = path.parent.parent.name.removesuffix("-" + result.protocol.opponent)
        size, streams = cell.rsplit("-s", 1)
        row = rows[cell, result.protocol.opponent]
        initial = result.coordinates.learner_transitions == 0
        means = row["initial" if initial else "endpoint"]
        if result.score.mean != means["mean"]:
            raise ValueError("Retained monitor and short-screen analysis disagree")
        roles: dict[str, list[float]] = {}
        for game in result.rows:
            extra = game.model_extra or {}
            seat, decks = extra["player_a_seat"], extra["seat_decks"]
            role = f"{decks[seat]} vs {decks[1 - seat]}"
            if game.score_a is None:
                raise ValueError("Missing game score")
            roles.setdefault(role, []).append(game.score_a)
        points.append(
            Point(
                name=cell,
                cell=cell,
                size=size,
                streams=int(streams),
                seed=result.training_seed,
                anchor=result.protocol.opponent,
                transitions=result.coordinates.learner_transitions,
                mean=result.score.mean,
                deal_blocks=row[
                    "initial_deal_block_scores"
                    if initial
                    else "endpoint_deal_block_scores"
                ],
                roles={
                    role: sum(values) / len(values) for role, values in roles.items()
                },
                coordinates=result.coordinates,
                checkpoint_sha256=result.artifact["sha256"],
                initialization_improvement=0 if initial else row["improvement"],
                paired_deal_improvement_95=[0, 0]
                if initial
                else row["paired_exact_block_bootstrap_95"],
            )
        )
        sources[str(path)] = digest(path)
    if len(points) != 44 or len(resources) != 11 or {p.seed for p in points} != {13571}:
        raise ValueError("Expected the complete eleven-cell added short screen")
    checkpoint_rows = checkpoint_table(points)
    endpoint_rows = endpoint_table(points, resources)
    role_rows = role_csv(points)
    if len(checkpoint_rows.splitlines()) != 46 or len(endpoint_rows.splitlines()) != 13:
        raise AssertionError("Report table omitted an actual observation")
    if len(list(csv.DictReader(io.StringIO(role_rows)))) != 176:
        raise AssertionError("Report omitted a deck-role observation")
    for cell in resources:
        end = next(
            p
            for p in points
            if p.cell == cell.cell
            and p.anchor == "scripted_greedy"
            and p.transitions == 131072
        )
        if (
            end.coordinates.optimizer_updates != cell.adam_steps
            or end.coordinates.active_training_seconds != cell.active_training_seconds
            or end.coordinates.training_seconds > cell.inclusive_training_seconds
        ):
            raise ValueError(
                "Real earlier checkpoint clocks/counters disagree with resource records"
            )
    OUT.mkdir(exist_ok=True)
    report = OUT / "earlier-screen-report-tables.md"
    report.write_text(
        "# Earlier screen013 — report formatting check\n\n"
        "Actual eleven-cell short-screen data, seed13571, endpoint131072 transitions. "
        "These are not protocol017 results.\n\n"
        + endpoint_rows
        + "\n\n"
        + checkpoint_rows
        + "\n"
    )
    roles_path = OUT / "earlier-screen-deck-roles.csv"
    roles_path.write_text(role_rows)
    receipt = {
        "status": "passed",
        "observed_at": datetime.now(UTC).isoformat(),
        "scope": "Actual earlier screen013 report tables; no long-study outcome or full renderer verification",
        "checkpoint_anchor_rows": 44,
        "endpoint_resource_rows": 11,
        "deck_role_rows": 176,
        "checkpoint_vs_fit_clocks_checked": 11,
        "source_sha256": sources,
        "renderer_sha256": digest(
            Path(__file__).with_name("render_learning_report.py")
        ),
        "readiness_script_sha256": digest(Path(__file__)),
        "outputs": {str(p): digest(p) for p in (report, roles_path)},
    }
    (OUT / "table-readiness.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "status",
                    "scope",
                    "checkpoint_anchor_rows",
                    "endpoint_resource_rows",
                    "deck_role_rows",
                    "checkpoint_vs_fit_clocks_checked",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
