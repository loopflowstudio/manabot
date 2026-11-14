"""CLI entrypoints for training and simulation."""

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import typer

from manabot.config.load import load_sim_config
from manabot.config.presets import DEFAULT_SIM_PRESET
from manabot.infra.config import MatchConfig
import managym

app = typer.Typer(
    help="Manabot training and simulation CLI", pretty_exceptions_show_locals=False
)


@app.command(
    "train",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train_command() -> None:
    """Training moved to Etude; run a recipe file with `etude train`."""
    typer.echo(
        "manabot train is removed. Use: uv run etude train FILE "
        "(for example, manabot/recipes/local.py). "
        "Resume an unfinished run with --resume.",
        err=True,
    )
    raise typer.Exit(2)


@app.command("sim")
def sim_command(
    preset: str = typer.Option(DEFAULT_SIM_PRESET, help="Simulation preset name"),
    set_values: Optional[list[str]] = typer.Option(
        None,
        "--set",
        help="Override config values with key.path=value (repeatable)",
    ),
    hero_deck_file: Path | None = typer.Option(
        None, help="UTF-8 main/sideboard decklist"
    ),
    villain_deck_file: Path | None = typer.Option(None, help="UTF-8 opponent decklist"),
) -> None:
    # Runtime imports stay at the command boundary so migration help works
    # without the training SDK.
    from manabot.sim.sim import run_simulation

    if (hero_deck_file is None) != (villain_deck_file is None):
        raise typer.BadParameter("Supply both --hero-deck-file and --villain-deck-file")
    try:
        sim_config, runtime_config = load_sim_config(preset, set_values)
        if hero_deck_file is not None and villain_deck_file is not None:
            hero, villain = managym.admit_decklists(
                [
                    ("hero", managym.Decklist.read(hero_deck_file)),
                    ("villain", managym.Decklist.read(villain_deck_file)),
                ]
            )
            config = sim_config.model_dump()
            config["match"] = MatchConfig.from_import(hero, villain).model_dump()
            sim_config = type(sim_config).model_validate(config)
            typer.echo(
                f"Admitted {managym.DEVELOPMENT_RULESET} decks "
                f"{hero.deck.identity} vs {villain.deck.identity}"
            )
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    run_simulation(sim_config, runtime_config)


@app.command("training-report")
def training_report_command(
    source: list[str] = typer.Option(
        ..., help="Local evidence directory or s3://bucket/prefix; repeatable"
    ),
    out: str = typer.Option("training-time.html", help="Self-contained output HTML"),
    run: list[str] | None = typer.Option(
        None, help="Exact logical run ID; repeatable. Otherwise select largest runs"
    ),
    top: int = typer.Option(
        5, min=1, help="Initially select this many runs by observed active time"
    ),
    include_benchmarks: bool = typer.Option(
        False,
        help="Include labeled benchmarks, smokes and fixtures in initial selection",
    ),
    training_clock: str = typer.Option(
        "training_seconds",
        help="Etude time clock measuring cumulative active training seconds",
    ),
) -> None:
    """Select and aggregate saved training runs; never launch compute."""
    from pathlib import Path

    from manabot.reports.time_cohort import write_cohort_report
    from manabot.reports.time_sources import load_time_sources

    try:
        cohort = load_time_sources(source, training_clock=training_clock)
        target = write_cohort_report(
            cohort,
            Path(out),
            selected=run,
            top=top,
            include_benchmarks=include_benchmarks,
        )
    except ImportError as error:
        raise typer.BadParameter(
            "Install report dependencies: uv sync --extra reports"
        ) from error
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        f"{target}: {len(cohort.runs)} logical runs; {len(cohort.warnings)} source/coverage warnings"
    )
    if not cohort.runs:
        raise typer.Exit(1)


@app.command("belief-demo")
def belief_demo_command() -> None:
    """Prove generated and supplied beliefs use one decision core."""

    from manabot.belief.demo import main as run_belief_demo

    run_belief_demo()


@app.command("belief-learn-demo")
def belief_learn_demo_command(
    episodes: int = typer.Option(
        160,
        min=40,
        max=256,
        help="Exact-p0 deals sampled through the frozen behavior population",
    ),
    held_out_episodes: int = typer.Option(
        32,
        min=8,
        max=128,
        help="Whole deals withheld before fresh-model training",
    ),
    steps: int = typer.Option(
        16,
        min=1,
        max=64,
        help="Bounded fresh-model population training steps",
    ),
    seed: int = typer.Option(197, help="Population sampling, split, and model seed"),
) -> None:
    """Compare population-trained exact-world beliefs with p0 on held-out deals."""

    from manabot.recipes.belief import main as run_belief_learn_demo

    run_belief_learn_demo(
        episodes=episodes,
        held_out_episodes=held_out_episodes,
        steps=steps,
        seed=seed,
    )


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        app()
    else:
        app(args=list(argv), standalone_mode=False)


if __name__ == "__main__":
    main()
