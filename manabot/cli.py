"""CLI entrypoints for training and simulation."""

import importlib
import inspect
from pathlib import Path
import pkgutil
from typing import Literal, Optional, Sequence

import typer

from manabot.config.load import load_sim_config
from manabot.config.presets import DEFAULT_SIM_PRESET
from manabot.recipes.baseline import BASELINE_STEPS, run_baseline
from manabot.recipes.local import LOCAL_SECONDS, LOCAL_STEPS, run_local
from manabot.sim.sim import run_simulation
from manabot.training.regime import TrainingRegime

app = typer.Typer(
    help="Manabot training and simulation CLI", pretty_exceptions_show_locals=False
)

# Named regimes are repository experiment code outside the installed package,
# found by module name: experiments/<experiment>.py, one experiment per file.
EXPERIMENTS = "experiments"


def regime_names() -> list[str]:
    """Every `--regime-name`, in file order.

    A recipe is a public zero-argument function returning a TrainingRegime.
    `<experiment>` names the module's canonical recipe, the function named like
    the module; `<experiment>:<variant>` names any other recipe in it.
    """
    names: list[str] = []
    package = importlib.import_module(EXPERIMENTS)
    for file in sorted(info.name for info in pkgutil.iter_modules(package.__path__)):
        module = importlib.import_module(f"{EXPERIMENTS}.{file}")
        for name, value in vars(module).items():
            if name.startswith("_") or not inspect.isfunction(value):
                continue
            signature = inspect.signature(value)
            if (
                value.__module__ == module.__name__
                and not signature.parameters
                and signature.return_annotation is TrainingRegime
            ):
                names.append(file if name == file else f"{file}:{name}")
    return names


def named_regime(name: str) -> TrainingRegime:
    """Build a fresh copy of the named recipe; reject an unknown name."""
    if name not in regime_names():
        raise typer.BadParameter(
            f"unknown regime {name!r}; available: {', '.join(regime_names())}"
        )
    file, _, function = name.partition(":")
    module = importlib.import_module(f"{EXPERIMENTS}.{file}")
    recipe = getattr(module, function or file)
    regime = recipe()
    assert isinstance(regime, TrainingRegime)
    return regime


def _run_sim(preset: str, set_values: list[str]) -> None:
    sim_config, runtime_config = load_sim_config(
        preset=preset, set_overrides=set_values
    )
    run_simulation(sim_config, runtime_config)


@app.command("train")
def train_command(
    preset: Optional[str] = typer.Option(
        None, help="Retired; select --recipe or --regime"
    ),
    recipe: str | None = typer.Option(
        None, help="Training recipe: local (default) or baseline"
    ),
    learning_seats: Literal["one", "both"] | None = typer.Option(
        None, help="Learning seats for local/baseline recipes (default: both)"
    ),
    steps: int | None = typer.Option(
        None, min=1, help="Recipe horizon in collection/update iterations"
    ),
    seconds: float | None = typer.Option(
        None, min=1, help="Recipe wall-time limit; does not alter learning schedules"
    ),
    regime: Optional[str] = typer.Option(None, help="TrainingRegime JSON file"),
    regime_name: str | None = typer.Option(
        None,
        help="Recipe in experiments/: <experiment> or <experiment>:<variant>",
    ),
    seed: Optional[int] = typer.Option(None, help="Regime or recipe training seed"),
    out: Optional[str] = typer.Option(None, help="New run directory"),
    resume_from: Optional[str] = typer.Option(
        None, help="Stopped TrainingRun ID in the same store"
    ),
    source_repair_reason: str | None = typer.Option(
        None, help="Explicit compatible source repair on complete-game recovery"
    ),
    checkpoint_updates: int | None = typer.Option(
        None, min=1, help="Raw monitoring export every N absolute self-play iterations"
    ),
    checkpoint_seconds: Optional[float] = typer.Option(
        None,
        help="Legacy elapsed-time monitoring interval for an explicitly frozen recipe",
    ),
    initial_admission: Optional[str] = typer.Option(None, hidden=True),
    set_values: Optional[list[str]] = typer.Option(
        None,
        "--set",
        help="Retired training overrides; use an explicit --regime JSON",
    ),
) -> None:
    if preset is not None or set_values:
        raise typer.BadParameter(
            "training --preset/--set are retired; use --recipe local, --recipe baseline, "
            "or an explicit --regime JSON. Historical reward and optimizer settings "
            "are not translated."
        )
    if recipe is None and regime is None and regime_name is None:
        recipe = "local"
    if recipe is not None:
        if recipe not in {"local", "baseline"}:
            raise typer.BadParameter("unknown recipe; available: local, baseline")
        if any(
            value is not None
            for value in (
                regime,
                regime_name,
                resume_from,
                source_repair_reason,
                checkpoint_updates,
                checkpoint_seconds,
                initial_admission,
            )
        ):
            raise typer.BadParameter(
                "--recipe cannot be combined with regime inputs or resume/cadence controls"
            )
        if recipe == "local":
            result = run_local(
                steps=LOCAL_STEPS if steps is None else steps,
                seed=197 if seed is None else seed,
                out=Path(out) if out else None,
                seconds=LOCAL_SECONDS if seconds is None else seconds,
                learning_seats=learning_seats or "both",
            )
            status = result.status
            typer.echo(f"local: {status}; run: {result.id}")
            if status == "completed":
                typer.echo(f"checkpoint: {result.stages[-1].artifacts['raw']['path']}")
            elif result.error:
                typer.echo(result.error, err=True)
        else:
            baseline_result = run_baseline(
                steps=BASELINE_STEPS if steps is None else steps,
                seed=197 if seed is None else seed,
                out=Path(out) if out else None,
                seconds=seconds,
                learning_seats=learning_seats or "both",
            )
            status = baseline_result.status
            typer.echo(
                f"baseline: {status}; report: {Path(baseline_result.notebook).with_name('report.md')}"
            )
        if status != "completed":
            raise typer.Exit(1)
        return
    if steps is not None or seconds is not None or learning_seats is not None:
        raise typer.BadParameter(
            "--steps, --seconds and --learning-seats require a recipe"
        )
    if regime is not None and regime_name is not None:
        raise typer.BadParameter("select --regime or --regime-name, not both")
    if out is None:
        raise typer.BadParameter("--regime requires --out")
    from manabot.training.execution import execute_regime
    from manabot.verify.store import VerifyStore

    if regime_name is not None:
        regime_spec = named_regime(regime_name)
    else:
        assert regime is not None
        regime_spec = TrainingRegime.model_validate_json(Path(regime).read_text())
    with VerifyStore(Path(out).parent / "training.sqlite") as store:
        execute_regime(
            regime_spec,
            197 if seed is None else seed,
            out,
            store,
            resume_from=resume_from,
            source_repair_reason=source_repair_reason,
            checkpoint_seconds=checkpoint_seconds,
            checkpoint_updates=checkpoint_updates,
            initial_admission=Path(initial_admission) if initial_admission else None,
        )


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
    content_pool: str = typer.Option(
        "tla-pool-v1", help="Qualified native content pool"
    ),
) -> None:
    if hero_deck_file is None and villain_deck_file is None:
        _run_sim(preset, set_values or [])
        return
    if hero_deck_file is None or villain_deck_file is None:
        raise typer.BadParameter("Supply both --hero-deck-file and --villain-deck-file")
    from manabot.decklist import import_files

    try:
        imported = import_files(hero_deck_file, villain_deck_file, content_pool)
        sim_config, runtime_config = load_sim_config(preset, set_values)
        config = sim_config.model_dump()
        config["match"] = imported.match_config().model_dump()
        sim_config = type(sim_config).model_validate(config)
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Admitted {imported.ruleset} setup {imported.setup_id}")
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
