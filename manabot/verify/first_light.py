"""Read and report historical first-light evidence; training uses regimes."""

import json
import math
from pathlib import Path
from typing import Any

from .store import VerifyStore

DEFAULT_REPORT_DIR = Path("reports")

DELTA_METRICS = (
    "win_rate",
    "win_ci_lower",
    "landed_when_able",
    "cast_when_able",
    "passed_when_able",
    "pass_land_pass_rate",
    "pass_land_land_rate",
    "mean_pass_prob_when_pass_land",
    "mean_land_prob_when_pass_land",
)


def summarize_run(store: VerifyStore, run_id: int) -> dict[str, Any]:
    run = store.get_run(run_id)
    configs = store.get_run_configs(run_id)
    evaluations = store.get_evaluations(run_id)

    baseline = next((row for row in evaluations if row["kind"] == "baseline"), None)
    final = next((row for row in reversed(evaluations) if row["kind"] == "final"), None)
    if final is None and evaluations:
        final = evaluations[-1]

    deltas = {}
    if baseline and final:
        for key in DELTA_METRICS:
            deltas[key] = float(final[key]) - float(baseline[key])

    recommendation, reasons, heuristic = recommend_next_step(baseline, final)
    checkpoints = [row for row in evaluations if row["kind"] == "checkpoint"]

    return {
        "run": run,
        "configs": configs,
        "baseline": baseline,
        "final": final,
        "checkpoints": checkpoints,
        "deltas": deltas,
        "recommendation": recommendation,
        "reasons": reasons,
        "heuristic": heuristic,
    }


def recommend_next_step(
    baseline: dict[str, Any] | None,
    final: dict[str, Any] | None,
) -> tuple[str, list[str], dict[str, bool]]:
    if final is None:
        return (
            "Stay in first-light: no final evaluation was recorded",
            ["The run has no final evaluation row, so there is nothing to judge."],
            {
                "win_signal_ok": False,
                "land_chain_ok": False,
                "spell_chain_ok": False,
            },
        )

    baseline_win = float(baseline["win_rate"]) if baseline else 0.0
    baseline_landed = float(baseline["landed_when_able"]) if baseline else 0.0
    win_delta = float(final["win_rate"]) - baseline_win

    win_signal_ok = win_delta >= 0.05 and float(final["win_ci_lower"]) >= max(
        0.0,
        baseline_win - 0.02,
    )
    land_chain_ok = (
        float(final["landed_when_able"]) >= max(0.35, baseline_landed - 0.05)
        and float(final["pass_land_land_rate"]) > float(final["pass_land_pass_rate"])
        and float(final["mean_land_prob_when_pass_land"])
        > float(final["mean_pass_prob_when_pass_land"])
    )
    spell_chain_ok = float(final["cast_when_able"]) >= 0.70

    reasons = [
        (
            f"Win rate delta vs baseline: {win_delta:+.1%} "
            f"(baseline {baseline_win:.1%}, final {float(final['win_rate']):.1%})."
        ),
        (
            "Pass-vs-land choice state: "
            f"land {float(final['pass_land_land_rate']):.1%}, "
            f"pass {float(final['pass_land_pass_rate']):.1%}."
        ),
        (
            "Land probability in pass+land states: "
            f"{float(final['mean_land_prob_when_pass_land']):.1%} "
            f"vs pass {float(final['mean_pass_prob_when_pass_land']):.1%}."
        ),
        f"Cast when able: {float(final['cast_when_able']):.1%}.",
    ]

    if win_signal_ok and land_chain_ok and spell_chain_ok:
        recommendation = "Move on to auxiliary-head experiments"
    elif not win_signal_ok:
        recommendation = "Stay in first-light: random-opponent win signal is too weak"
    elif not land_chain_ok:
        recommendation = (
            "Stay in first-light: pass-collapse still dominates land decisions"
        )
    else:
        recommendation = "Stay in first-light: creature-play signal remains weak"

    return (
        recommendation,
        reasons,
        {
            "win_signal_ok": win_signal_ok,
            "land_chain_ok": land_chain_ok,
            "spell_chain_ok": spell_chain_ok,
        },
    )


def _format_percent(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.1%}"


def _format_delta(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.1%}"


def render_report(store: VerifyStore, run_id: int) -> tuple[str, dict[str, Any]]:
    summary = summarize_run(store, run_id)
    run = summary["run"]
    configs = summary["configs"]
    baseline = summary["baseline"]
    final = summary["final"]
    deltas = summary["deltas"]

    metric_rows = [
        ("Win rate", "win_rate"),
        ("Win CI lower", "win_ci_lower"),
        ("Landed when able", "landed_when_able"),
        ("Cast when able", "cast_when_able"),
        ("Passed when able", "passed_when_able"),
        ("Pass+land: pass rate", "pass_land_pass_rate"),
        ("Pass+land: land rate", "pass_land_land_rate"),
        ("Mean pass prob in pass+land", "mean_pass_prob_when_pass_land"),
        ("Mean land prob in pass+land", "mean_land_prob_when_pass_land"),
    ]

    lines = [
        f"# First Light Report — run {run_id}",
        "",
        "## Recommendation",
        "",
        f"**{summary['recommendation']}**",
        "",
        *[f"- {reason}" for reason in summary["reasons"]],
        "",
        "## Key metrics and deltas vs untrained baseline",
        "",
        "| Metric | Baseline | Final | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in metric_rows:
        baseline_value = float(baseline[key]) if baseline else None
        final_value = float(final[key]) if final else None
        delta_value = deltas.get(key)
        lines.append(
            f"| {label} | {_format_percent(baseline_value)} | "
            f"{_format_percent(final_value)} | {_format_delta(delta_value)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation of causal-chain metrics",
            "",
            f"- Final landed_when_able: {_format_percent(float(final['landed_when_able'])) if final else 'n/a'}",
            (
                "- Final pass+land split: "
                f"land {_format_percent(float(final['pass_land_land_rate'])) if final else 'n/a'}, "
                f"pass {_format_percent(float(final['pass_land_pass_rate'])) if final else 'n/a'}"
            ),
            (
                f"- Final cast_when_able: {float(final['cast_when_able']):.1%}"
                if final
                else "- Final cast_when_able: n/a"
            ),
            (
                f"- Final explained variance: {float(final['explained_variance']):.3f}"
                if final and final["explained_variance"] is not None
                else "- Final explained variance: n/a"
            ),
            "",
            "## Experiment config",
            "",
            f"- Label: {run['label'] or 'n/a'}",
            f"- Mode: {run['mode']}",
            f"- Seed: {run['seed']}",
            f"- Opponent: {run['opponent_policy']}",
            f"- Recipe: {run['recipe_name']}",
            f"- Total timesteps: {run['total_timesteps']}",
            f"- Eval interval: {run['eval_interval']}",
            f"- Eval games: {run['eval_num_games']}",
            f"- Git branch: {run['git_branch'] or 'n/a'}",
            f"- Git commit: {run['git_commit'] or 'n/a'}",
            f"- Tree dirty: {bool(run['tree_dirty'])}",
            "",
            "## Optional detailed notes / tables",
            "",
            "```json",
            json.dumps(configs, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )

    return "\n".join(lines), summary


def write_report(
    store: VerifyStore,
    *,
    run_id: int,
    output_path: Path | None,
    report_kind: str = "summary",
) -> tuple[str, Path | None, dict[str, Any]]:
    markdown, summary = render_report(store, run_id)
    written_path = output_path
    if written_path is not None:
        written_path.parent.mkdir(parents=True, exist_ok=True)
        written_path.write_text(markdown)

    store.record_report(
        run_id=run_id,
        report_kind=report_kind,
        markdown_path=str(written_path) if written_path else None,
        recommendation=summary["recommendation"],
        summary=summary,
    )
    return markdown, written_path, summary


def default_report_path(run_id: int) -> Path:
    return DEFAULT_REPORT_DIR / f"first-light-run-{run_id}.md"
