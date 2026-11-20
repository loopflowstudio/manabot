"""Attack lock, longer: with the lock prevented, does the policy learn to hold back?

Two arms of the Elemental recipe at four times the default KL weight toward the
reference policy (`tau_scale` 0.2), which held the attack bias near 80% over 60
updates in `attack_lock_kl`. One keeps the default input; the other scales the
permanent timestamp rank by 1/128. Exploratory.

    uv run etude launch experiments/attack_lock_long.py --revision COMMIT
"""

from decimal import Decimal
from pathlib import Path

from etude.jobs import TrainingJob

from manabot.experiment.setup import machine_setup
from manabot.recipes.elemental import ElementalConfig, train

ROOT = Path(__file__).resolve().parents[1]
SEED = 41901
TAU_SCALE = 0.2
# Arm name -> input ablation.
ARMS: dict[str, str] = {"tau4x": "none", "tau4x-timestamp": "timestamp"}


def training_jobs(
    revision: str, updates: int = 400, hours: float = 4.0, usd_per_run: float = 2.5
) -> list[TrainingJob]:
    return [
        TrainingJob(
            name=f"attack-lock-long-{arm}",
            train=train,
            config=ElementalConfig(
                updates=updates,
                device="cuda",
                input_ablation=ablation,
                tau_scale=TAU_SCALE,
                recover_capacity=True,
            ),
            smoke=ElementalConfig(
                updates=1,
                device="cpu",
                streams=2,
                input_ablation=ablation,
                tau_scale=TAU_SCALE,
            ),
            seeds=(SEED,),
            repo=str(ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
        )
        for arm, ablation in ARMS.items()
    ]
