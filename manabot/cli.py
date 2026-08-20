"""CLI entrypoints for training and simulation."""

from typing import Optional, Sequence

import typer

from manabot.config.load import load_sim_config, load_train_config
from manabot.config.presets import DEFAULT_SIM_PRESET, DEFAULT_TRAIN_PRESET
from manabot.model.train import run_training
from manabot.sim.sim import run_simulation

app = typer.Typer(help="Manabot training and simulation CLI")


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
    preset: str = typer.Option(DEFAULT_TRAIN_PRESET, help="Training preset name"),
    set_values: Optional[list[str]] = typer.Option(
        None,
        "--set",
        help="Override config values with key.path=value (repeatable)",
    ),
) -> None:
    _run_train(preset, set_values or [])


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
    steps: int = typer.Option(
        80,
        min=1,
        max=256,
        help="Bounded fresh-model overfit steps",
    ),
    seed: int = typer.Option(197, help="Fresh model and materialization seed"),
) -> None:
    """Fit a fresh belief model to one authority-only hidden-world label."""

    from manabot.belief.learning_demo import main as run_belief_learn_demo

    run_belief_learn_demo(steps=steps, seed=seed)


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        app()
    else:
        app(args=list(argv), standalone_mode=False)


if __name__ == "__main__":
    main()
