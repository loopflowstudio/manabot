"""CLI entrypoints for training and simulation."""

from typing import Optional, Sequence

import typer

from manabot.config.load import load_sim_config, load_train_config
from manabot.config.presets import DEFAULT_SIM_PRESET, DEFAULT_TRAIN_PRESET
from manabot.model.train import run_training
from manabot.remote.cli import app as deploy_app
from manabot.sim.sim import run_simulation

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
    regime: Optional[str] = typer.Option(None, help="TrainingRegime JSON file"),
    seed: Optional[int] = typer.Option(None, help="Regime execution seed"),
    out: Optional[str] = typer.Option(None, help="New run directory"),
    resume_from: Optional[str] = typer.Option(
        None, help="Stopped TrainingRun ID in the same store"
    ),
    checkpoint_seconds: Optional[float] = typer.Option(
        None,
        help="Monitoring raw checkpoint interval (3600 for hourly), at update/epoch boundaries",
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
    if regime:
        if preset is not None or set_values:
            raise typer.BadParameter(
                "--regime cannot be combined with --preset or --set"
            )
        if out is None:
            raise typer.BadParameter("--regime requires --out")
        from pathlib import Path

        from manabot.remote.plan import Allocation
        from manabot.training.execution import execute_regime
        from manabot.training.models import TrainingRegime
        from manabot.verify.store import VerifyStore

        recipe = TrainingRegime.model_validate_json(Path(regime).read_text())
        with VerifyStore(Path(out).parent / "training.sqlite") as store:
            execute_regime(
                recipe,
                197 if seed is None else seed,
                out,
                store,
                resume_from=resume_from,
                allocation=Allocation.model_validate_json(Path(allocation).read_text())
                if allocation
                else None,
                checkpoint_seconds=checkpoint_seconds,
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
            or initial_admission is not None
            or allocation is not None
        ):
            raise typer.BadParameter(
                "--out, --seed, --resume-from and --checkpoint-seconds require --regime"
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
