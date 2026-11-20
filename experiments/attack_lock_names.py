"""Attack lock: do per-name embeddings on object tokens hide creature stats?

One arm of the Elemental recipe with `ModelConfig.input_ablation="names"`: the
permanent timestamp rank is scaled by 1/128 and card, permanent and stack
tokens take their definition embedding from the rules program alone, without
the per-name vector. Compare with `attack_lock`'s `timestamp` arm (same seed and
updates, names kept) and `bare` arm (definitions, facts, effects and shared
context all removed). Exploratory.

    uv run etude launch experiments/attack_lock_names.py --revision COMMIT
"""

from decimal import Decimal
from pathlib import Path

from etude.jobs import TrainingJob

from manabot.experiment.setup import machine_setup
from manabot.recipes.elemental import ElementalConfig, train

ROOT = Path(__file__).resolve().parents[1]
SEED = 41901


def training_jobs(
    revision: str, updates: int = 140, hours: float = 2.0, usd_per_run: float = 2.0
) -> list[TrainingJob]:
    return [
        TrainingJob(
            name="attack-lock-names",
            train=train,
            config=ElementalConfig(
                updates=updates,
                device="cuda",
                input_ablation="names",
                recover_capacity=True,
            ),
            smoke=ElementalConfig(
                updates=1, device="cpu", streams=2, input_ablation="names"
            ),
            seeds=(SEED,),
            repo=str(ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
        )
    ]
