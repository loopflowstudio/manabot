"""First-light experiment harness and report generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
from typing import Any

from manabot.env import Match, ObservationSpace, Reward, VectorEnv
from manabot.infra import Experiment
from manabot.model import Agent, Trainer

from .store import VerifyStore
from .util import STANDARD_DECK, build_hypers, capture_evaluation

FIRST_LIGHT_RECIPE_NAME = "first_light_shaped_v1"
DEFAULT_REPORT_DIR = Path("reports")

MODE_PRESETS = {
    "dev": {
        "num_envs": 4,
        "num_steps": 128,
        "total_timesteps": 262_144,
        "eval_interval": 65_536,
        "eval_num_games": 50,
    },
    "decision": {
        "num_envs": 8,
        "num_steps": 128,
        "total_timesteps": 1_048_576,
        "eval_interval": 262_144,
        "eval_num_games": 200,
    },
}

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


@dataclass(frozen=True)
class FirstLightRunConfig:
    """Resolved first-light run configuration."""

    db: Path
    seed: int
    mode: str
    label: str | None
    opponent_policy: str
    num_envs: int
    num_steps: int
    total_timesteps: int
    eval_interval: int | None
    eval_num_games: int
    baseline: bool
    report: bool
    report_path: Path | None
    notes: str | None = None


def resolve_run_config(
    *,
    db: Path | str,
    seed: int,
    mode: str,
    label: str | None,
    opponent_policy: str,
    num_envs: int | None,
    num_steps: int | None,
    total_timesteps: int | None,
    eval_interval: int | None,
    eval_num_games: int | None,
    baseline: bool,
    report: bool,
    report_path: Path | str | None,
    notes: str | None = None,
) -> FirstLightRunConfig:
    defaults = MODE_PRESETS[mode]
    return FirstLightRunConfig(
        db=Path(db),
        seed=seed,
        mode=mode,
        label=label,
        opponent_policy=opponent_policy,
        num_envs=num_envs if num_envs is not None else defaults["num_envs"],
        num_steps=num_steps if num_steps is not None else defaults["num_steps"],
        total_timesteps=(
            total_timesteps
            if total_timesteps is not None
            else defaults["total_timesteps"]
        ),
        eval_interval=(
            eval_interval if eval_interval is not None else defaults["eval_interval"]
        ),
        eval_num_games=(
            eval_num_games if eval_num_games is not None else defaults["eval_num_games"]
        ),
        baseline=baseline,
        report=report,
        report_path=Path(report_path) if report_path else None,
        notes=notes,
    )


def _run_name(config: FirstLightRunConfig, suffix: str) -> str:
    label = config.label or config.mode
    safe = label.replace("/", "-").replace(" ", "-")
    return f"first-light-{safe}-{suffix}"


def build_first_light_hypers(
    config: FirstLightRunConfig,
    *,
    total_timesteps: int,
    seed: int,
    exp_name_suffix: str,
) -> Any:
    return build_hypers(
        experiment={
            "seed": seed,
            "exp_name": _run_name(config, exp_name_suffix),
        },
        train={
            "num_envs": config.num_envs,
            "num_steps": config.num_steps,
            "total_timesteps": total_timesteps,
            "opponent_policy": config.opponent_policy,
            "eval_num_games": config.eval_num_games,
        },
        reward={
            "win_reward": 1.0,
            "lose_reward": -1.0,
            "land_play_reward": 0.03,
            "creature_play_reward": 0.06,
            "opponent_life_loss_reward": 0.01,
        },
        match={
            "hero_deck": STANDARD_DECK,
            "villain_deck": STANDARD_DECK,
        },
    )


def collect_git_metadata() -> dict[str, Any]:
    """Best-effort branch/commit/dirty metadata for run provenance."""

    def _run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip() or None

    status = _run("status", "--porcelain") or ""
    return {
        "git_commit": _run("rev-parse", "HEAD"),
        "git_branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "tree_dirty": bool(status),
    }


def _batch_size(config: FirstLightRunConfig) -> int:
    return config.num_envs * config.num_steps


def step_to_update_index(config: FirstLightRunConfig, step: int) -> int | None:
    batch_size = _batch_size(config)
    if batch_size <= 0:
        return None
    return step // batch_size


def checkpoint_steps(total_timesteps: int, eval_interval: int | None) -> list[int]:
    if not eval_interval or eval_interval <= 0:
        return []
    return list(range(eval_interval, total_timesteps, eval_interval))


def _train_chunk(
    agent: Agent | None,
    hypers,
) -> tuple[Agent, Trainer]:
    observation_space = ObservationSpace(hypers.observation)
    match = Match(hypers.match)
    reward = Reward(hypers.reward)
    experiment = Experiment(hypers.experiment, hypers)
    env = VectorEnv(
        hypers.train.num_envs,
        match,
        observation_space,
        reward,
        device=experiment.device,
        seed=hypers.experiment.seed,
        opponent_policy=hypers.train.opponent_policy,
    )
    learner = agent or Agent(observation_space, hypers.agent)
    trainer = Trainer(learner, experiment, env, hypers.train)
    trainer.train()
    return learner, trainer


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
    baseline_could_spell = float(baseline["could_spell"]) if baseline else 0.0

    win_delta = float(final["win_rate"]) - baseline_win
    could_spell_ratio = (
        float(final["could_spell"]) / baseline_could_spell
        if baseline_could_spell > 0
        else 1.0
    )

    win_signal_ok = win_delta >= 0.05 and float(final["win_ci_lower"]) >= max(
        0.0,
        baseline_win - 0.02,
    )
    land_chain_ok = (
        float(final["landed_when_able"]) >= max(0.35, baseline_landed - 0.05)
        and float(final["pass_land_land_rate"])
        > float(final["pass_land_pass_rate"])
        and float(final["mean_land_prob_when_pass_land"])
        > float(final["mean_pass_prob_when_pass_land"])
    )
    spell_chain_ok = (
        float(final["cast_when_able"]) >= 0.70 and could_spell_ratio >= 0.75
    )

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
        (
            "Spell opportunity retention: "
            f"{could_spell_ratio:.2f}x baseline "
            f"with cast_when_able {float(final['cast_when_able']):.1%}."
        ),
    ]

    if win_signal_ok and land_chain_ok and spell_chain_ok:
        recommendation = "Move on to auxiliary-head experiments"
    elif not win_signal_ok:
        recommendation = "Stay in first-light: random-opponent win signal is too weak"
    elif not land_chain_ok:
        recommendation = "Stay in first-light: pass-collapse still dominates land decisions"
    else:
        recommendation = "Stay in first-light: creature-play signal remains weak"

    return recommendation, reasons, {
        "win_signal_ok": win_signal_ok,
        "land_chain_ok": land_chain_ok,
        "spell_chain_ok": spell_chain_ok,
    }


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
                "- Final spell opportunity count: "
                f"{float(final['could_spell']):.0f}" if final else "- Final spell opportunity count: n/a"
            ),
            (
                "- Final explained variance: "
                f"{float(final['explained_variance']):.3f}"
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


def run_first_light(config: FirstLightRunConfig) -> dict[str, Any]:
    """Execute a first-light run and persist all results to SQLite."""

    store = VerifyStore(config.db)
    git = collect_git_metadata()
    initial_hypers = build_first_light_hypers(
        config,
        total_timesteps=max(config.total_timesteps, 0),
        seed=config.seed,
        exp_name_suffix="bootstrap",
    )
    run_id = store.create_run(
        label=config.label,
        status="running",
        mode=config.mode,
        seed=config.seed,
        git_commit=git["git_commit"],
        git_branch=git["git_branch"],
        tree_dirty=git["tree_dirty"],
        recipe_name=FIRST_LIGHT_RECIPE_NAME,
        opponent_policy=config.opponent_policy,
        num_envs=config.num_envs,
        num_steps=config.num_steps,
        total_timesteps=config.total_timesteps,
        eval_interval=config.eval_interval,
        eval_num_games=config.eval_num_games,
        baseline_auto=config.baseline,
        notes=config.notes,
        configs={
            "experiment": initial_hypers.experiment.model_dump(mode="json"),
            "train": initial_hypers.train.model_dump(mode="json"),
            "reward": initial_hypers.reward.model_dump(mode="json"),
            "agent": initial_hypers.agent.model_dump(mode="json"),
            "match": initial_hypers.match.model_dump(mode="json"),
            "observation": initial_hypers.observation.model_dump(mode="json"),
        },
    )

    try:
        observation_space = ObservationSpace(initial_hypers.observation)
        match = Match(initial_hypers.match)
        reward = Reward(initial_hypers.reward)

        if config.baseline:
            baseline_agent = Agent(observation_space, initial_hypers.agent)
            baseline = capture_evaluation(
                baseline_agent,
                observation_space,
                match,
                reward,
                num_games=config.eval_num_games,
                opponent_policy=config.opponent_policy,
                deterministic=False,
                seed=config.seed,
                capture_actions=True,
            )
            store.record_evaluation(
                run_id=run_id,
                kind="baseline",
                step=0,
                update_index=0,
                opponent_policy=config.opponent_policy,
                num_games=config.eval_num_games,
                deterministic=False,
                artifacts=baseline,
                retain_actions=True,
            )

        trained_agent: Agent | None = None
        last_explained_variance: float | None = None
        previous_step = 0

        for checkpoint_step in checkpoint_steps(
            config.total_timesteps,
            config.eval_interval,
        ):
            chunk_hypers = build_first_light_hypers(
                config,
                total_timesteps=checkpoint_step - previous_step,
                seed=config.seed + checkpoint_step,
                exp_name_suffix=f"ckpt-{checkpoint_step}",
            )
            trained_agent, trainer = _train_chunk(trained_agent, chunk_hypers)
            last_explained_variance = trainer.last_explained_variance
            checkpoint = capture_evaluation(
                trained_agent,
                observation_space,
                match,
                reward,
                num_games=config.eval_num_games,
                opponent_policy=config.opponent_policy,
                deterministic=False,
                seed=config.seed + 10_000 + checkpoint_step,
                capture_actions=False,
            )
            store.record_evaluation(
                run_id=run_id,
                kind="checkpoint",
                step=checkpoint_step,
                update_index=step_to_update_index(config, checkpoint_step),
                opponent_policy=config.opponent_policy,
                num_games=config.eval_num_games,
                deterministic=False,
                artifacts=checkpoint,
                explained_variance=last_explained_variance,
                retain_actions=False,
            )
            previous_step = checkpoint_step

        if config.total_timesteps > previous_step:
            final_hypers = build_first_light_hypers(
                config,
                total_timesteps=config.total_timesteps - previous_step,
                seed=config.seed + config.total_timesteps,
                exp_name_suffix="final",
            )
            trained_agent, trainer = _train_chunk(trained_agent, final_hypers)
            last_explained_variance = trainer.last_explained_variance

        if trained_agent is None:
            trained_agent = Agent(observation_space, initial_hypers.agent)

        final = capture_evaluation(
            trained_agent,
            observation_space,
            match,
            reward,
            num_games=config.eval_num_games,
            opponent_policy=config.opponent_policy,
            deterministic=False,
            seed=config.seed + 20_000 + config.total_timesteps,
            capture_actions=True,
        )
        store.record_evaluation(
            run_id=run_id,
            kind="final",
            step=config.total_timesteps,
            update_index=step_to_update_index(config, config.total_timesteps),
            opponent_policy=config.opponent_policy,
            num_games=config.eval_num_games,
            deterministic=False,
            artifacts=final,
            explained_variance=last_explained_variance,
            retain_actions=True,
        )
        store.update_run_status(run_id, "completed")

        report_path = None
        if config.report:
            report_path = config.report_path or default_report_path(run_id)
            write_report(
                store,
                run_id=run_id,
                output_path=report_path,
                report_kind="decision" if config.mode == "decision" else "summary",
            )

        summary = summarize_run(store, run_id)
        summary["run_id"] = run_id
        summary["report_path"] = str(report_path) if report_path else None
        return summary
    except Exception as exc:
        store.update_run_status(run_id, "failed", notes=str(exc))
        raise
    finally:
        store.close()
