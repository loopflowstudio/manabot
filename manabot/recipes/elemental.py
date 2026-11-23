"""The current recipe: damped-policy self-play on the ten Elemental decks.

`train(config)` opens the Etude run and calls the self-play loop once.
Etude restores registered learner objects when another attempt carries it on.
"""

from pathlib import Path
import time
from typing import Literal

from etude.report import Report
from etude.train.tracking import track
from pydantic import Field, model_validator

from manabot.infra.run_settings import RunSettings
from manabot.recipes.builders import elemental_self_play, require_backend_device
from manabot.training.loops.self_play import CLOCKS, SelfPlay, self_play
from manabot.training.panels import self_play_report

# Longest the CUDA numerics gate may take before training.
NUMERICS_SECONDS = 1800


class ElementalConfig(RunSettings):
    backend: Literal["torch", "mlx"] = Field(
        default="torch", exclude_if=lambda value: value == "torch"
    )
    updates: int = Field(default=3200, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    # Run the CUDA numerics gate before the first update; refuse to train on failure.
    validate_numerics: bool = False
    streams: int = Field(default=256, ge=2)

    @model_validator(mode="after")
    def backend_runs_on_device(self) -> "ElementalConfig":
        require_backend_device(self.backend, self.device)
        if self.validate_numerics and self.backend != "torch":
            raise ValueError("the CUDA numerics gate requires backend 'torch'")
        return self


def gate(directory: Path) -> None:
    """Pass the CUDA numerics gate once per run directory."""
    from manabot.experiment.numerics import NumericsReceipt, check_numerics

    receipt = directory / "numerical-validation.json"
    if (
        receipt.exists()
        and NumericsReceipt.model_validate_json(receipt.read_bytes()).status
        == "completed"
    ):
        return
    status = check_numerics(directory, time.time() + NUMERICS_SECONDS).status
    if status != "completed":
        raise RuntimeError(f"CUDA numerics gate {status}")


def report() -> Report:
    return self_play_report()


def settings(config: ElementalConfig) -> SelfPlay:
    return elemental_self_play().model_copy(
        update={
            "device": config.device,
            "streams": config.streams,
            "run_settings": config.run_settings,
        }
    )


def train(config: ElementalConfig) -> None:
    from manabot.backends import resolve

    # Resolve before opening the run: a missing framework leaves no failed run.
    backend = resolve(config.backend)
    with track(clocks=CLOCKS, config=config) as run:
        if config.validate_numerics:
            gate(run.out / "numerics")
        self_play(
            settings(config),
            run,
            updates=config.updates,
            backend=backend,
        )
