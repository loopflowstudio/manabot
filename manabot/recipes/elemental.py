"""The current recipe: damped-policy self-play on the ten Elemental decks.

`train(config)` opens the Etude run and calls the self-play loop once.
Etude restores registered learner objects when another attempt carries it on.
"""

from pathlib import Path
import time
from typing import Literal

from etude.train.tracking import track
from pydantic import Field

from manabot.experiment.numerics import NumericsReceipt, check_numerics
from manabot.infra.run_settings import RunSettings
from manabot.recipes.builders import elemental_self_play
from manabot.training.loops.self_play import CLOCKS, self_play

# Longest the CUDA numerics gate may take before training.
NUMERICS_SECONDS = 1800


class ElementalConfig(RunSettings):
    updates: int = Field(default=3200, ge=1)
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    # Run the CUDA numerics gate before the first update; refuse to train on failure.
    validate_numerics: bool = False
    streams: int = Field(default=256, ge=2)
    # Attack-lock study: simplify the model's object tokens (ModelConfig.input_ablation).
    input_ablation: Literal["none", "timestamp", "bare"] = "none"
    # Attack-lock study: weight of the KL to the reference policy (default 0.05).
    tau_scale: float | None = Field(default=None, ge=0)


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
    base = elemental_self_play()
    settings = base.model_copy(
        update={
            "model": base.model.model_copy(
                update={"input_ablation": config.input_ablation}
            ),
            "learning": base.learning
            if config.tau_scale is None
            else base.learning.model_copy(update={"tau_scale": config.tau_scale}),
            "device": config.device,
            "streams": config.streams,
            "run_settings": config.run_settings,
        }
    )
    with track(clocks=CLOCKS, config=config) as run:
        if config.validate_numerics:
            gate(run.out / "numerics")
        self_play(settings, run, updates=config.updates)
