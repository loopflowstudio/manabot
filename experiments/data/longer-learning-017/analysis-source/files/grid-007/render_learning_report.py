"""Render factual protocol017 tables and figures after complete offline audit.

The output is an evidence package for interpretation and the MTG-122 handoff.
It preserves both seeds and the frozen selector; a failed competence screen is
never converted into a qualified recommendation. No training/evaluation runs here.
"""

import csv
from decimal import Decimal
import io
import json
from pathlib import Path
from typing import Literal

from learning_figures import LABELS, ORDER, Tradeoff, curves, endpoint_tradeoffs
from learning_resources import Budget, CellResources, GroupResources, digest
from learning_selection import COUNTS, SEEDS, Point, Recipe
from pydantic import BaseModel
from retain_analysis_source import verify_current as verify_source_archive

ROOT = Path(".runs/grid-007/learning-017")
OUT = ROOT / "report"


class Verification(BaseModel):
    status: Literal["verified"]
    completed_commands: int
    completed_panels: int
    completed_replayed_games: int
    protocol_sha256: str
    root_manifest_sha256: str
    evaluation_wall_seconds: float
    analyzer_sha256: str
    selection_sha256: str
    tradeoff_analysis_sha256: str


class Analysis(BaseModel):
    verification: Verification
    points: list[Point]
    recipes: list[Recipe]
    preferred_by_frozen_rule: Recipe | None
    score_time_tradeoffs: list[Tradeoff]


class Resources(BaseModel):
    status: Literal["verified"]
    groups: list[GroupResources]
    study_conservative_charge_usd: Decimal
    campaign_budget: Budget
    input_sha256: dict[str, str]


class ResolvedRecipe(BaseModel):
    cell: str
    size: str
    hidden_dim: int
    attention_layers: int
    streams: int
    total_transitions: int
    total_iterations: int
    raw_checkpoint_iterations: list[int]
    schedule_reference_iterations: int
    regime_path: str
    regime_sha256: str


class RecipeIndex(BaseModel):
    source: str
    allocation_sha256: str
    schedule_formula: str
    records: list[ResolvedRecipe]


def percentage(value: float) -> str:
    return f"{100 * value:.2f}%"


def delta(value: float) -> str:
    return f"{100 * value:+.2f}"


def point_order(point: Point) -> tuple[int, int, int, int, str]:
    return (
        ORDER.index(point.size),
        point.streams,
        point.seed,
        point.transitions,
        point.anchor,
    )


def checkpoint_table(points: list[Point]) -> str:
    """Preserve each training seed, raw score and initialization-relative interval."""
    rows = [
        "| Model | Streams | Seed | Transitions | Anchor | Score | Gain (pp) | Paired-deal 95% gain interval (pp) | Checkpoint minutes |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for point in sorted(points, key=point_order):
        low, high = point.paired_deal_improvement_95
        rows.append(
            f"| {LABELS[point.size]} | {point.streams} | {point.seed} | {point.transitions:,} | "
            f"{point.anchor} | {percentage(point.mean)} | {delta(point.initialization_improvement)} | "
            f"[{delta(low)}, {delta(high)}] | {point.coordinates.training_seconds / 60:.3f} |"
        )
    return "\n".join(rows)


def endpoint_table(points: list[Point], resources: list[CellResources]) -> str:
    endpoint = max(p.transitions for p in points)
    indexed = {
        (p.cell, p.seed, p.anchor): p for p in points if p.transitions == endpoint
    }
    rows = [
        "| Model | Streams | Seed | Greedy | Random | Greedy gain (pp) | Inclusive fit min | Active min | Adam steps | Selected exposures | Empty groups skipped | Peak CUDA GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for resource in sorted(
        resources, key=lambda r: (ORDER.index(r.size), r.streams, r.seed)
    ):
        g, r = (
            indexed[resource.cell, resource.seed, "scripted_greedy"],
            indexed[resource.cell, resource.seed, "random"],
        )
        rows.append(
            f"| {LABELS[resource.size]} | {resource.streams} | {resource.seed} | "
            f"{percentage(g.mean)} | {percentage(r.mean)} | {delta(g.initialization_improvement)} | "
            f"{resource.inclusive_training_seconds / 60:.3f} | {resource.active_training_seconds / 60:.3f} | "
            f"{resource.adam_steps:,} | {resource.optimizer_exposures:,} | {resource.skipped_empty_groups:,} | "
            f"{resource.peak_cuda_bytes / 2**30:.3f} |"
        )
    return "\n".join(rows)


def selection_table(recipes: list[Recipe]) -> str:
    rows = [
        "| Recipe | Sustained qualification | First qualifying transitions | Worst seed minutes to qualification | Parameters |",
        "|---|---|---:|---:|---:|",
    ]
    for recipe in recipes:
        count = (
            f"{recipe.first_qualifying_transitions:,}"
            if recipe.first_qualifying_transitions is not None
            else "—"
        )
        seconds = recipe.worst_seed_seconds_to_first_qualified
        minutes = f"{seconds / 60:.3f}" if seconds is not None else "—"
        rows.append(
            f"| {recipe.cell} | {'Passed' if recipe.qualified else 'Did not pass'} | {count} | "
            f"{minutes} | {recipe.parameters:,} |"
        )
    return "\n".join(rows)


def role_csv(points: list[Point]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        (
            "model",
            "streams",
            "training_seed",
            "learner_transitions",
            "anchor",
            "deck_role",
            "score",
        )
    )
    for point in sorted(points, key=point_order):
        if len(point.roles) != 4:
            raise ValueError("A scored panel is missing deck-role cells")
        for role, score in sorted(point.roles.items()):
            writer.writerow(
                (
                    LABELS[point.size],
                    point.streams,
                    point.seed,
                    point.transitions,
                    point.anchor,
                    role,
                    score,
                )
            )
    return stream.getvalue()


def validate(analysis: Analysis, resources: Resources, index: RecipeIndex) -> None:
    proof = analysis.verification
    for name, expected in (
        ("analyze_learning.py", proof.analyzer_sha256),
        ("learning_selection.py", proof.selection_sha256),
        ("learning_figures.py", proof.tradeoff_analysis_sha256),
    ):
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError("Current analysis source differs from the result receipt")
    if (
        proof.completed_commands,
        proof.completed_panels,
        proof.completed_replayed_games,
    ) != (44, 176, 18304):
        raise ValueError("Full declared evaluation has not passed")
    if len(analysis.points) != 176 or len(analysis.recipes) != 11:
        raise ValueError("Analysis cohort is incomplete")
    if {p.seed for p in analysis.points} != set(SEEDS):
        raise ValueError("Unexpected training seeds")
    if {p.transitions for p in analysis.points} != set(COUNTS):
        raise ValueError("Unexpected retained checkpoint coordinates")
    if proof.root_manifest_sha256 != digest(ROOT / "evaluation/manifest.json"):
        raise ValueError("Evaluation manifest changed after analysis")
    if proof.protocol_sha256 != digest(ROOT / "evaluation/protocol.md"):
        raise ValueError("Evaluated protocol differs from analysis")
    for path, expected in resources.input_sha256.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Resource input changed: {path}")
    if index.allocation_sha256 != digest(ROOT / "allocation.json"):
        raise ValueError("Resolved recipes refer to another allocation")
    for recipe in index.records:
        if (
            digest(ROOT / "resolved-recipes" / recipe.regime_path)
            != recipe.regime_sha256
        ):
            raise ValueError("Resolved recipe bytes changed")
    records = {(c.cell, c.seed): c for group in resources.groups for c in group.cells}
    if (
        len(records) != sum(len(g.cells) for g in resources.groups)
        or len(records) != 22
    ):
        raise ValueError("Resource cohort is incomplete or duplicated")
    for point in analysis.points:
        if point.transitions != COUNTS[-1]:
            continue
        resource = records[point.cell, point.seed]
        coords = point.coordinates
        if (
            coords.optimizer_updates != resource.adam_steps
            or coords.optimizer_exposures != resource.optimizer_exposures
            or coords.games != resource.games
            or coords.updates != resource.iterations
            or coords.active_training_seconds != resource.active_training_seconds
            or coords.training_seconds > resource.inclusive_training_seconds
        ):
            raise ValueError(
                "Checkpoint clocks/counters disagree with the retained fit"
            )


def main() -> None:
    analysis_path, resources_path = (
        ROOT / "analysis/results.json",
        ROOT / "resources.json",
    )
    analysis = Analysis.model_validate_json(analysis_path.read_text())
    resources = Resources.model_validate_json(resources_path.read_text())
    index = RecipeIndex.model_validate_json(
        (ROOT / "resolved-recipes/index.json").read_text()
    )
    validate(analysis, resources, index)
    source_archive = verify_source_archive()
    cells = [cell for group in resources.groups for cell in group.cells]
    # Generate everything from verified inputs only. An existing output needs
    # inspection before replacement; it never admits another experiment.
    OUT.mkdir(exist_ok=False)
    source_target = OUT / "analysis-source"
    for relative in (
        Path("README.md"),
        Path("manifest.json"),
        *[Path(record.stored_path) for record in source_archive.files],
    ):
        target = source_target / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_archive.directory / relative).read_bytes())
    (OUT / "checkpoint-scores.md").write_text(checkpoint_table(analysis.points) + "\n")
    (OUT / "deck-role-scores.csv").write_text(role_csv(analysis.points))
    for name, path in (
        ("analysis.json", analysis_path),
        ("resources.json", resources_path),
        ("training-verification.json", ROOT / "verified-results.json"),
        ("evaluation-verification.json", ROOT / "analysis/verification.json"),
        ("evaluation-runtime.json", ROOT / "evaluation-runtime.json"),
        ("rental-settlement.json", ROOT / "rental-settlement.json"),
        ("guardian-verification.json", ROOT / "guardian-verification.json"),
        ("allocation.json", ROOT / "allocation.json"),
        ("protocol.md", ROOT / "evaluation/protocol.md"),
        (
            "concurrency-amendment.md",
            Path("scratch/grid-017-concurrency-amendment.md"),
        ),
    ):
        (OUT / name).write_bytes(path.read_bytes())
    (OUT / "recipes").mkdir()
    for path in (ROOT / "resolved-recipes").glob("*.json"):
        (OUT / "recipes" / path.name).write_bytes(path.read_bytes())
    for clock in ("transitions", "inclusive", "active"):
        curves(
            analysis.points,
            clock=clock,
            title=f"Longer learning017 — {clock}",
            target=OUT / f"learning-{clock}.svg",
        )
    endpoint_tradeoffs(
        analysis.score_time_tradeoffs,
        title="Longer learning017 — endpoint score and time",
        target=OUT / "endpoint-tradeoffs.svg",
    )
    preferred = analysis.preferred_by_frozen_rule
    if preferred is None:
        conclusion = (
            "**No recipe passes the frozen practical competence screen.** "
            "The full analysis retains each seed's endpoint shortfall. A fallback recommendation "
            "must state those shortfalls explicitly; none is promoted by changing the threshold."
        )
    else:
        recipe = next(r for r in index.records if r.cell == preferred.cell)
        conclusion = (
            f"The frozen rule selects **{LABELS[recipe.size]}, {recipe.streams} streams**. "
            f"Recommendation evidence concerns its full {recipe.total_transitions:,}-transition recipe "
            f"({recipe.total_iterations:,} iterations), even though sustained qualification first "
            f"appeared at {preferred.first_qualifying_transitions:,} transitions. "
            f"The exact resolved recipe is [retained here](recipes/{recipe.regime_path})."
        )
    rental_rows = []
    for group in resources.groups:
        rental = group.rentals[0]
        rental_rows.append(
            f"| {group.seed} | {group.intervals.start_to_first_cell_seconds / 60:.2f} | "
            f"{sum(c.inclusive_training_seconds for c in group.cells) / 60:.2f} | "
            f"{float(rental.request_to_deletion_seconds) / 3600:.3f} | ${rental.usd_per_hour} | "
            f"${rental.quoted_rate_cost_floor_usd:.3f} |"
        )
    budget = resources.campaign_budget
    text = f"""# Longer size and batching comparison: evidence for MTG-122

{conclusion}

This package contains the completed 22-fit comparison, both training seeds,
four exact raw checkpoints and all 18,304 replay-verified evaluation games.
The final interpretation and actual figures still require operator review.

## Measured endpoint

{endpoint_table(analysis.points, cells)}

Scores use 104 games per anchor/checkpoint: 13 paired deal blocks with eight
seat/deck/mirror legs. Wins count as 1, draws as 0.5, losses as 0. Each seed stays
separate. [All checkpoint scores and initialization-relative intervals](checkpoint-scores.md),
[all four deck-role cells](deck-role-scores.csv), and [complete paired comparisons](analysis.json)
remain available. Deck-role results accompany the aggregate scores.

![Learning against experience](learning-transitions.svg)
![Learning against inclusive checkpoint time](learning-inclusive.svg)
![Endpoint score/time tradeoff](endpoint-tradeoffs.svg)

## Frozen competence rule

{selection_table(analysis.recipes)}

Qualification requires both seeds at ≥50% against greedy, ≥75% against random,
and ≥20 percentage points greedy improvement from their own initialization,
at the selected checkpoint and all later retained points. Choose the smallest
worst-seed time; within 10%, prefer fewer parameters, then lower CUDA memory.
This is an operational rule, not a population significance test or skill ceiling.
An early threshold crossing does not authorize shortening the tested schedule.

## Recipe clocks and transfer

Source: `{index.source}`. All fits use corrected w6, cross-balanced current-self,
64 transitions/stream, Ataraxos MOVE/Adam, FP32 with TF32 off, one CPU producer/thread,
and RTX 5090. [Complete recipes](recipes/index.json) preserve all learner parameters.

{index.schedule_formula}

The full horizon is 524,288 transitions. Four/64/256 streams use 2,048/128/32
iterations respectively, with raw checkpoints at 0/131,072/262,144/524,288 transitions.
The learning rate remains 1e-4 over this tested horizon. EMA stays 0.999 per iteration.
Batching changes optimizer grouping, update opportunities, filtering and staleness;
these results compare whole recipes and do not isolate a single cause.

MTG-122's current saved direction selects 64-stream GPU training, Mini evaluation,
all 36,000 assessment games and its own shared research catalog. A 256-stream result
would be an alternative to that selected stream count, not an automatic amendment.
Keep common learnable tensor shapes across its pools, exact native-content admission,
fresh initializations and its fixed controls. This grid's product checkpoint identities
and transport-ID buffers are not interchangeable with the research catalog.
Recalibrate all three pools with its final source and diagnostics setting: these
measurements retain inline diagnostics and do not quote the split pipeline's cost.
This package changes no execution authorization or existing cohort.

## Time and cost

| Training seed | Worker start to first cell, min | Sum inclusive fit min | Request-to-deletion hours | Quoted hourly rate | Conservative occupied-cost floor |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rental_rows)}

The study's conservative settled charge is ${resources.study_conservative_charge_usd}.
Campaign accounting is ${budget.conservative_rental_total_usd} settled,
${budget.committed_caps_usd} in open reservations and ${budget.ancillary_reserve_usd}
ancillary reserve within the authorized ${budget.authorized_total_usd}.
SDK charges and quote-based occupied-time floors are not final provider invoices.
The independent MTG-121 startup budget remains separate.

Inclusive checkpoint clocks stop at the retained stage coordinate; inclusive fit
clocks also include final run bookkeeping. Active time is collection plus learning.
Worker phase intervals include dispatch/export/check overhead. Shared startup and
retention are not assigned to individual recipes. The local CPU evaluation used
{analysis.verification.evaluation_wall_seconds / 3600:.3f} hours of coordinator wall time
and no rented GPU. [Resource records](resources.json) preserve the separate counters.

## Limits and provenance

Two training seeds give a limited view of training variability. The paired-deal
bootstrap intervals condition on each seed; they neither estimate population
training variance nor correct the many grid comparisons. Scores at intermediate
wall times were not measured. Different earlier normalized schedules mean the
131,072 point here is not another replicate of the short screen.

The two-rental account rule was exceeded at admission when another task's pod
appeared while local jobs were queued. The controller omitted queued reservations
from its count, and the two admitted rentals made three account pods. Existing
trajectories were preserved and further admissions held. No scientific recipe,
source, checkpoint, endpoint or budget was extended. Earlier CUDA initialization
failures and the historical finite-weight anomaly remain unresolved; this cohort's
health checks do not establish their causes.

All facts derive from the [frozen protocol](protocol.md), [allocation](allocation.json),
[audited analysis](analysis.json) and [resources](resources.json). The manifest binds
this rendered package. The [analysis source archive](analysis-source/README.md)
retains the exact programs and documents the additional runtime/data prerequisites.
The [training verification](training-verification.json) retains raw-checkpoint,
recipe, numerical-audit and per-phase timing evidence. [Evaluation verification](evaluation-verification.json),
[runtime identity](evaluation-runtime.json) and [rental settlement](rental-settlement.json)
remain separate receipts, rather than being inferred from the score table.
The [guardian payload check](guardian-verification.json) binds both provider
allocations to the inspected independent in-pod deadline program; it is not
proof of a live guardian process or final provider deletion.
The [prospective concurrency amendment](concurrency-amendment.md) records the
operator's correction of an unsupported account-wide rental limit so the separately
budgeted startup test could run alongside this study. Frozen scientific inputs and
analysis were unchanged; earlier admission deviations remain part of the record.
MTG-151's approximately 20k-update follow-up remains paused;
it neither extends this cohort nor supplies evidence of a higher skill ceiling.
"""
    (OUT / "evidence.md").write_text(text)
    receipt = {
        "status": "rendered; visual and scientific interpretation review pending",
        "input_sha256": {
            str(p): digest(p)
            for p in (
                analysis_path,
                resources_path,
                ROOT / "resolved-recipes/index.json",
            )
        },
        "renderer_sha256": digest(Path(__file__)),
        "files": [
            {
                "path": str(p.relative_to(OUT)),
                "bytes": p.stat().st_size,
                "sha256": digest(p),
            }
            for p in sorted(OUT.rglob("*"))
            if p.is_file()
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rendered": str(OUT),
                "files": len(receipt["files"]),
                "preferred_by_frozen_rule": preferred.cell if preferred else None,
            }
        )
    )


if __name__ == "__main__":
    main()
