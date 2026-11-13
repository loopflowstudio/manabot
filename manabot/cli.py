"""CLI entrypoints for training and simulation."""

import ast
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
from typing import Any, Literal, Optional, get_type_hints

from etude.stopping import Stopped
from etude.train.run import local_launch
from pydantic import BaseModel
import typer

from manabot.config.load import load_sim_config
from manabot.config.presets import DEFAULT_SIM_PRESET
from manabot.infra.config import MatchConfig
from manabot.recipes import baseline, belief_sampler, elemental, local
from manabot.recipes.device import accelerator
from manabot.sim.sim import run_simulation
from manabot.training.regime import FROZEN_EXECUTION_REVISION
import managym

app = typer.Typer(
    help="Manabot training and simulation CLI", pretty_exceptions_show_locals=False
)

RECIPES = {
    "local": (local.train, local.LocalConfig),
    "baseline": (baseline.train, baseline.BaselineConfig),
    "elemental": (elemental.train, elemental.ElementalConfig),
    "belief": (belief_sampler.train, belief_sampler.BeliefConfig),
}


def recipe_names() -> list[str]:
    """List explicit training functions without importing experiment evaluations."""
    names = list(RECIPES)
    package = importlib.util.find_spec("experiments")
    if package is not None:
        for directory in package.submodule_search_locations or ():
            for path in sorted(Path(directory).glob("*.py")):
                tree = ast.parse(path.read_text())
                if (
                    any(
                        isinstance(node, ast.FunctionDef)
                        and node.name == "train"
                        and len(node.args.args) == 1
                        and node.args.args[0].arg == "config"
                        or isinstance(node, ast.ImportFrom)
                        and any(
                            alias.name == "train" and alias.asname is None
                            for alias in node.names
                        )
                        for node in tree.body
                    )
                    and path.stem not in names
                ):
                    names.append(path.stem)
    return names


def experiment_recipe(name: str) -> tuple[Callable[[Any], None], type[BaseModel]]:
    """Validate an experiment's typed function at the dynamic CLI boundary."""
    module_name, _, function = name.partition(":")
    if not module_name.isidentifier() or (function and not function.isidentifier()):
        raise typer.BadParameter("recipe must name a module and optional function")
    module = importlib.import_module(f"experiments.{module_name}")
    train = getattr(module, function or "train", None)
    if not callable(train):
        raise typer.BadParameter("recipe must be a function taking one typed config")
    parameters = inspect.signature(train).parameters
    if len(parameters) != 1:
        raise typer.BadParameter("recipe must be a function taking one typed config")
    parameter = next(iter(parameters))
    config = get_type_hints(train).get(parameter)
    if not isinstance(config, type) or not issubclass(config, BaseModel):
        raise typer.BadParameter("recipe config must be a Pydantic model")
    if not {"seed", "device"} <= config.model_fields.keys():
        raise typer.BadParameter("CLI recipe configs need seed and device fields")
    return train, config


@app.command("train")
def train_command(
    recipe: str = typer.Option("local", help="Training recipe"),
    list_recipes: bool = typer.Option(False, "--list", help="List runnable recipes"),
    seed: int = typer.Option(197, help="Training seed"),
    updates: int | None = typer.Option(None, min=1, help="Absolute update target"),
    steps: int | None = typer.Option(None, min=1, hidden=True),
    device: Literal["cpu", "mps", "cuda"] | None = typer.Option(None),
    out: Path | None = typer.Option(None, help="Run directory; rerun to resume"),
    seconds: float | None = typer.Option(
        None, min=1, help="Stop and save after this allowance"
    ),
    learning_seats: Literal["one", "both"] = typer.Option("both"),
    padded_shapes: bool = typer.Option(False, "--padded-shapes"),
    regime: str | None = typer.Option(None, hidden=True),
    regime_name: str | None = typer.Option(None, hidden=True),
    resume_from: str | None = typer.Option(None, hidden=True),
    source_repair_reason: str | None = typer.Option(None, hidden=True),
    checkpoint_updates: int | None = typer.Option(None, hidden=True),
    checkpoint_seconds: float | None = typer.Option(None, hidden=True),
    preset: str | None = typer.Option(None, hidden=True),
    set_values: list[str] | None = typer.Option(None, "--set", hidden=True),
) -> None:
    if any(
        value is not None
        for value in (
            regime,
            regime_name,
            resume_from,
            source_repair_reason,
            checkpoint_updates,
            checkpoint_seconds,
            preset,
            set_values,
        )
    ):
        raise typer.BadParameter(
            f"saved-regime execution is retired; run revision {FROZEN_EXECUTION_REVISION}. "
            "Current training uses --recipe; repeat --out to resume."
        )
    if list_recipes:
        typer.echo("\n".join(recipe_names()))
        return
    if recipe not in {"local", "baseline"} and (
        learning_seats != "both" or padded_shapes
    ):
        raise typer.BadParameter(
            "--learning-seats and --padded-shapes require local or baseline"
        )
    if steps is not None:
        if updates is not None:
            raise typer.BadParameter("choose --updates or --steps, not both")
        typer.echo(
            "Warning: --steps is deprecated; use --updates. "
            "Both flags now set only the stop target; for local and baseline, "
            "unlike the old --steps, they leave the recipe's schedule and "
            "checkpoint grid unchanged. "
            "--steps will be removed in Manabot 0.3.0.",
            err=True,
        )
        updates = steps
    chosen = device or accelerator() or "cpu"
    if recipe == "baseline" and chosen == "cpu":
        raise typer.BadParameter(
            "baseline requires CUDA or explicit --device mps; use local for CPU"
        )
    if recipe in RECIPES:
        recipe_train, config_type = RECIPES[recipe]
    else:
        try:
            recipe_train, config_type = experiment_recipe(recipe)
        except (ImportError, AttributeError) as error:
            raise typer.BadParameter(
                f"unknown recipe {recipe!r}; use --list"
            ) from error
    values: dict[str, object] = {"seed": seed, "device": chosen}
    if recipe in {"local", "baseline"}:
        values.update(learning_seats=learning_seats, padded_shapes=padded_shapes)
    config = config_type.model_validate(values)
    directory = out or Path(".runs") / (
        "local-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if recipe == "local"
        else f"{recipe.replace(':', '-')}-{seed}"
    )
    # A target-only continuation keeps the recipe's learning settings unchanged.
    environment = {}
    if updates is not None:
        environment["ETUDE_UNTIL_CLOCK"] = "updates"
        environment["ETUDE_UNTIL_VALUE"] = str(updates)
    allowance = (
        local.LOCAL_SECONDS if seconds is None and recipe == "local" else seconds
    )
    if allowance is not None:
        environment["ETUDE_DEADLINE"] = (
            datetime.now(timezone.utc) + timedelta(seconds=allowance)
        ).isoformat()
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        with local_launch(out=directory, experiment=recipe):
            recipe_train(config)
    except Stopped as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(75) from error
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    typer.echo(f"{recipe}: completed in {directory}")


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
