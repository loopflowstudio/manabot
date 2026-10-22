"""CLI entrypoints for training and simulation."""

from pathlib import Path
from typing import Optional, Sequence

import typer

from manabot.config.load import load_sim_config, load_train_config
from manabot.config.presets import DEFAULT_SIM_PRESET, DEFAULT_TRAIN_PRESET
from manabot.model.train import run_training
from manabot.remote.cli import app as deploy_app
from manabot.sim.sim import run_simulation
from manabot.training.baseline import BASELINE_STEPS, run_baseline

app = typer.Typer(
    help="Manabot training and simulation CLI", pretty_exceptions_show_locals=False
)
app.add_typer(deploy_app, name="deploy")


def _run_train(preset: str, set_values: list[str]) -> None:
    hypers = load_train_config(preset=preset, set_overrides=set_values)
    run_training(hypers)


def _run_sim(preset: str, set_values: list[str]) -> None:
    sim_hypers, experiment_hypers = load_sim_config(
        preset=preset, set_overrides=set_values
    )
    run_simulation(sim_hypers, experiment_hypers)


@app.command("train")
def train_command(
    preset: Optional[str] = typer.Option(None, help="Training preset name"),
    recipe: str | None = typer.Option(None, help="Complete training recipe: baseline"),
    steps: int | None = typer.Option(
        None, min=1, help="Recipe horizon in collection/update iterations"
    ),
    seconds: float | None = typer.Option(
        None, min=1, help="Recipe wall-time limit; does not alter learning schedules"
    ),
    regime: Optional[str] = typer.Option(None, help="TrainingRegime JSON file"),
<<<<<<< HEAD
=======
    regime_name: str | None = typer.Option(
        None, help="Named regime from experiments/recipes.py"
    ),
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
    seed: Optional[int] = typer.Option(None, help="Regime or recipe training seed"),
    out: Optional[str] = typer.Option(None, help="New run directory"),
    resume_from: Optional[str] = typer.Option(
        None, help="Stopped TrainingRun ID in the same store"
    ),
    checkpoint_updates: int | None = typer.Option(
        None, min=1, help="Raw monitoring export every N absolute self-play iterations"
    ),
    checkpoint_seconds: Optional[float] = typer.Option(
        None,
        help="Legacy elapsed-time monitoring interval for an explicitly frozen recipe",
    ),
    initial_admission: Optional[str] = typer.Option(None, hidden=True),
    allocation: Optional[str] = typer.Option(
        None, help="Admitted allocation JSON, with absolute deadline"
    ),
    set_values: Optional[list[str]] = typer.Option(
        None,
        "--set",
        help="Override config values with key.path=value (repeatable)",
    ),
) -> None:
    if recipe is not None:
        if recipe != "baseline":
            raise typer.BadParameter("unknown recipe; available: baseline")
        if (
            any(
                value is not None
                for value in (
                    preset,
                    regime,
<<<<<<< HEAD
=======
                    regime_name,
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
                    resume_from,
                    checkpoint_updates,
                    checkpoint_seconds,
                    initial_admission,
                    allocation,
                )
            )
            or set_values
        ):
            raise typer.BadParameter(
                "--recipe cannot be combined with regime/preset overrides or resume/cadence controls"
            )
        result = run_baseline(
            steps=BASELINE_STEPS if steps is None else steps,
            seed=197 if seed is None else seed,
            out=Path(out) if out else None,
            seconds=seconds,
        )
        typer.echo(
            f"baseline: {result.status}; report: {Path(result.notebook).with_name('report.md')}"
        )
        if result.status != "completed":
            raise typer.Exit(1)
        return
    if steps is not None or seconds is not None:
        raise typer.BadParameter("--steps and --seconds require --recipe")
<<<<<<< HEAD
    if regime:
=======
    if regime and regime_name:
        raise typer.BadParameter("select --regime or --regime-name, not both")
    if regime or regime_name:
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
        if preset is not None or set_values:
            raise typer.BadParameter(
                "--regime cannot be combined with --preset or --set"
            )
        if out is None:
            raise typer.BadParameter("--regime requires --out")
        from manabot.remote.plan import Allocation
        from manabot.training.execution import execute_regime
        from manabot.training.models import TrainingRegime
        from manabot.verify.store import VerifyStore

<<<<<<< HEAD
        regime_spec = TrainingRegime.model_validate_json(Path(regime).read_text())
=======
        if regime_name:
            # Named regimes are repository experiment code, outside the package.
            from experiments.recipes import regime as named_regime

            regime_spec = named_regime(regime_name)
        elif regime:
            regime_spec = TrainingRegime.model_validate_json(Path(regime).read_text())
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
        with VerifyStore(Path(out).parent / "training.sqlite") as store:
            execute_regime(
                regime_spec,
                197 if seed is None else seed,
                out,
                store,
                resume_from=resume_from,
                allocation=Allocation.model_validate_json(Path(allocation).read_text())
                if allocation
                else None,
                checkpoint_seconds=checkpoint_seconds,
                checkpoint_updates=checkpoint_updates,
                initial_admission=Path(initial_admission)
                if initial_admission
                else None,
            )
    else:
        if (
            out is not None
            or seed is not None
            or resume_from is not None
            or checkpoint_seconds is not None
            or checkpoint_updates is not None
            or initial_admission is not None
            or allocation is not None
        ):
            raise typer.BadParameter(
                "--out, --seed, --resume-from and checkpoint cadence require --regime"
            )
        _run_train(preset or DEFAULT_TRAIN_PRESET, set_values or [])


@app.command("sim")
def sim_command(
    preset: str = typer.Option(DEFAULT_SIM_PRESET, help="Simulation preset name"),
    set_values: Optional[list[str]] = typer.Option(
        None,
        "--set",
        help="Override config values with key.path=value (repeatable)",
    ),
) -> None:
    _run_sim(preset, set_values or [])


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

    from manabot.training.time_cohort import write_cohort_report
    from manabot.training.time_sources import load_time_sources

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

    from manabot.belief.learning_demo import main as run_belief_learn_demo

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
