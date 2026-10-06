"""Admission and two-hour feasibility stop for the prospective value-token screen.

This screen uses fixed update counts, not deadline-truncated successful training.
The diagnostic reads durable run receipts; failed or partial cohorts stay failed.
"""

from pathlib import Path
import shutil

from manabot.training.execution import atomic_json
from manabot.training.models import TrainingRegime, TrainingRun, TrainSelfPlay


def validate_screen_recipes(recipes: list[TrainingRegime]) -> None:
    """Allow only aggregation to vary; retain paired initialization and workload."""
    if [r.agent.value_aggregation for r in recipes] != [
        "historical_mean",
        "masked_mean",
        "value_token",
    ]:
        raise ValueError("screen must order historical, masked, token arms")
    controls: list[dict[str, object]] = []
    for recipe in recipes:
        if (
            recipe.agent.hidden_dim != 64
            or recipe.agent.attention_layers != 1
            or recipe.agent.value_kind != "scalar"
            or not recipe.agent.attention_on
            or recipe.agent.compound_decisions
            or recipe.wall_seconds != 2400
        ):
            raise ValueError("screen requires ordinary width-64 depth-1 scalar arms")
        if len(recipe.stages) != 2 or any(
            not isinstance(s, TrainSelfPlay)
            or s.updates != 400
            or s.transitions != 64
            or s.streams != 4
            or s.learning.gradient != "ataraxos_move"
            or s.execution.device != "cpu"
            or s.execution.threads != 1
            or s.execution.wall_seconds != 1190
            for s in recipe.stages
        ):
            raise ValueError("screen requires two equal 400-update CPU stages")
        if (
            recipe.stages[0].initial is not None
            or recipe.stages[1].initial != recipe.stages[0].id
        ):
            raise ValueError(
                "screen midpoint and endpoint must share continued training"
            )
        control = recipe.model_dump(mode="json")
        control.pop("id")
        control["agent"].pop("value_aggregation")
        controls.append(control)
    if any(control != controls[0] for control in controls[1:]):
        raise ValueError("screen may vary only value aggregation and recipe ID")


def screen_diagnostic(
    out: Path, elapsed: float, *, training: bool, planned_updates: int = 7200
) -> None:
    """Stop on insufficient progress or disk, without inspecting strength scores.

    Conservative remaining-training projection uses the slowest observed run's
    seconds per completed update, with 25% headroom. Startup and partial-update
    cost are included. This is a feasibility heuristic, not timing calibration.
    """
    runs = [
        TrainingRun.model_validate_json(p.read_text())
        for p in sorted(out.glob("*/run.json"))
    ]
    updates = [sum(len(s.diagnostics) for s in r.stages) for r in runs]
    completed = sum(updates)
    rates = [r.seconds / n for r, n in zip(runs, updates, strict=True) if n]
    projected = (
        elapsed
        + 1.25 * max(rates, default=float("inf")) * max(0, planned_updates - completed)
        if training
        else elapsed
    )
    free = shutil.disk_usage(out).free
    reasons = []
    if any(r.status in {"failed", "interrupted"} for r in runs):
        reasons.append("failed training attempt")
    if training and (not rates or projected > 21600):
        reasons.append("conservative training projection exceeds six hours")
    if free < 4 * 1024**3:
        reasons.append("less than 4 GiB evidence reserve")
    atomic_json(
        out / "diagnostic-2h.json",
        {
            "elapsed_seconds": elapsed,
            "training": training,
            "completed_updates": completed,
            "planned_updates": planned_updates,
            "projected_training_seconds": projected if rates else None,
            "free_disk_bytes": free,
            "decision": "stop" if reasons else "continue",
            "reasons": reasons,
            "strength_inspected": False,
        },
    )
    if reasons:
        raise RuntimeError("two-hour diagnostic stop: " + "; ".join(reasons))
