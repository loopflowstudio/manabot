"""The current recipe: damped-policy self-play on the ten Elemental decks.

`train(config)` opens the Etude run and calls the self-play loop once.
Etude restores the learner from its committed save on the next attempt.
"""

from pathlib import Path
import time

from etude.report import Report
from etude.train.tracking import track
from pydantic import Field, model_validator

from manabot.recipes.builders import SelfPlayRecipe, elemental_self_play
from manabot.training.loops.self_play import CLOCKS, SelfPlay, self_play
from manabot.training.panels import self_play_report

# Longest the CUDA numerics gate may take before training.
NUMERICS_SECONDS = 1800


class ElementalConfig(SelfPlayRecipe):
    updates: int = Field(default=3200, ge=1)
    # Run the CUDA numerics gate before the first update; refuse to train on failure.
    validate_numerics: bool = False
    streams: int = Field(default=256, ge=2)

    @model_validator(mode="after")
    def numerics_gate_uses_torch(self) -> "ElementalConfig":
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
