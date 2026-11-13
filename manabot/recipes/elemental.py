"""The current recipe: damped-policy self-play on the ten Elemental decks.

`train(config)` names its run and calls the self-play loop once. Running
it again with the same config resumes from the run directory's `state.pt`.
"""

from pathlib import Path
import time
from typing import Literal

from etude.train.records import Step
from etude.train.run import track
from pydantic import BaseModel, Field

from manabot.experiment.numerics import NumericsReceipt, check_numerics
from manabot.recipes.builders import elemental_self_play
from manabot.training.loops.self_play import self_play

# Longest the CUDA numerics gate may take before training.
NUMERICS_SECONDS = 1800


class ElementalConfig(BaseModel):
    seed: int = 197
    updates: int = Field(default=3200, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    # Run the CUDA numerics gate before the first update; refuse to train on failure.
    validate_numerics: bool = False
    streams: int = Field(default=256, ge=2)


def gate(directory: Path) -> None:
    """Pass the CUDA numerics gate once per run directory."""
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


def train(config: ElementalConfig) -> None:
    settings = elemental_self_play().model_copy(
        update={"device": config.device, "streams": config.streams}
    )
    with track(
        clocks=[Step("updates", "optimizer updates")],
        seed=config.seed,
        config=config.model_dump(mode="json"),
    ) as run:
        updates = int(run.until("updates", default=config.updates))
        if config.validate_numerics:
            gate(run.out / "numerics")
        self_play(
            settings,
            run.out,
            seed=config.seed,
            updates=updates,
            run=run,
        )
