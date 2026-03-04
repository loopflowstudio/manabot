"""CLI entrypoints for training and simulation."""

import argparse
from typing import Optional, Sequence

from manabot.config.load import load_sim_config, load_train_config
from manabot.model.train import run_training
from manabot.sim.sim import run_simulation

try:
    import typer
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    typer = None


def _run_train(preset: str, set_values: list[str]) -> None:
    hypers = load_train_config(preset=preset, set_overrides=set_values)
    run_training(hypers)


def _run_sim(preset: str, set_values: list[str]) -> None:
    sim_hypers, experiment_hypers = load_sim_config(
        preset=preset, set_overrides=set_values
    )
    run_simulation(sim_hypers, experiment_hypers)


if typer is not None:
    app = typer.Typer(help="Manabot training and simulation CLI")

    @app.command("train")
    def train_command(
        preset: str = typer.Option("local", help="Training preset name"),
        set_values: Optional[list[str]] = typer.Option(
            None,
            "--set",
            help="Override config values with key.path=value (repeatable)",
        ),
    ):
        _run_train(preset, set_values or [])

    @app.command("sim")
    def sim_command(
        preset: str = typer.Option("sim", help="Simulation preset name"),
        set_values: Optional[list[str]] = typer.Option(
            None,
            "--set",
            help="Override config values with key.path=value (repeatable)",
        ),
    ):
        _run_sim(preset, set_values or [])


def _fallback_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manabot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--preset", default="local")
    train_parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )

    sim_parser = subparsers.add_parser("sim", help="Run simulation")
    sim_parser.add_argument("--preset", default="sim")
    sim_parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "train":
        _run_train(args.preset, args.set_values)
    elif args.command == "sim":
        _run_sim(args.preset, args.set_values)


def main(argv: Sequence[str] | None = None) -> None:
    if typer is not None:
        if argv is None:
            app()
        else:
            app(args=list(argv), standalone_mode=False)
        return

    _fallback_main(argv)


if __name__ == "__main__":
    main()
