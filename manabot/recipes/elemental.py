"""The current recipe: damped-policy self-play on the ten Elemental decks.

`elemental(config)` names its run and calls the self-play loop once. Running
it again with the same config resumes from the run directory's `state.pt`.
"""

from pathlib import Path
import time

from pydantic import BaseModel, Field

from manabot.experiment.numerics import NumericsReceipt, check_numerics
from manabot.recipes.builders import elemental_self_play
from manabot.recipes.proposed_etude import Step, track
from manabot.training.loops.self_play import SelfPlay, self_play

# Longest the CUDA numerics gate may take before training.
NUMERICS_SECONDS = 1800


class ElementalConfig(BaseModel):
    seed: int = 197
    updates: int = Field(default=3200, ge=1)
    self_play: SelfPlay = Field(default_factory=elemental_self_play)
    # Run the CUDA numerics gate before the first update; refuse to train on failure.
    validate_numerics: bool = False


def _gate(directory: Path) -> None:
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


def elemental(config: ElementalConfig) -> None:
    run = track(
        clocks=[Step("updates", "optimizer updates")],
        seed=config.seed,
        config=config.model_dump(mode="json"),
    )
    try:
        if config.validate_numerics:
            _gate(run.out / "numerics")
        self_play(
            config.self_play,
            run.out,
            seed=config.seed,
            updates=config.updates,
            run=run,
        )
    except BaseException:
        run.finish("failed")
        raise
    run.finish()
