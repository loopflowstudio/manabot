"""Attack lock, KL sweep: small local runs of the Elemental recipe by reference-KL weight.

Each job multiplies the recipe's `tau_scale` (0.05), from a weight that should
pin the policy at the action-type-uniform reference down to the default.
The default weight and its double are `experiments/attack_lock.py` arms.
Exploratory; a 64-stream CPU run finished no update in 13 minutes, so these rent.

    uv run etude launch experiments/attack_lock_kl.py --revision COMMIT
"""

from decimal import Decimal
from pathlib import Path

from etude.jobs import TrainingJob

from manabot.experiment.setup import machine_setup
from manabot.recipes.elemental import ElementalConfig, train

ROOT = Path(__file__).resolve().parents[1]
SEED = 41902
MULTIPLIERS = (4, 16, 100, 1000)


def training_jobs(
    revision: str, updates: int = 60, hours: float = 1.0, usd_per_run: float = 1.0
) -> list[TrainingJob]:
    return [
        TrainingJob(
            name=f"kl-x{multiplier}",
            train=train,
            config=ElementalConfig(
                updates=updates,
                device="cuda",
                tau_scale=0.05 * multiplier,
                checkpoint_every=10,
                recover_capacity=True,
            ),
            smoke=ElementalConfig(updates=1, device="cpu", streams=2),
            seeds=(SEED,),
            repo=str(ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
        )
        for multiplier in MULTIPLIERS
    ]
