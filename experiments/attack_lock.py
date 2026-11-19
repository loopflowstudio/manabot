"""Attack lock: does a simpler observation stop self-play locking into always attacking?

Four arms of the current Elemental recipe at one commit and seed. Three differ
only in `ModelConfig.input_ablation`: `none` (control), `timestamp` (permanent
timestamp rank scaled by 1/128) and `bare` (also no definition embeddings,
facts, shared effects or shared context on object tokens). `tau2x` keeps the
control's input and doubles the KL weight toward the reference policy.
Exploratory.

    uv run etude launch experiments/attack_lock.py --revision COMMIT
"""

from decimal import Decimal
from pathlib import Path

from etude.jobs import TrainingJob

from manabot.experiment.setup import machine_setup
from manabot.recipes.elemental import ElementalConfig, train

ROOT = Path(__file__).resolve().parents[1]
SEED = 41901
# Arm name -> (input ablation, tau_scale override).
# `bare` stopped at update 5 on a relation-capacity overflow; `bare-recover`
# repeats it ending such games instead.
ARMS: dict[str, tuple[str, float | None]] = {
    "none": ("none", None),
    "timestamp": ("timestamp", None),
    "bare": ("bare", None),
    "tau2x": ("none", 0.1),
    "bare-recover": ("bare", None),
}


def training_jobs(
    revision: str, updates: int = 140, hours: float = 2.0, usd_per_run: float = 2.0
) -> list[TrainingJob]:
    return [
        TrainingJob(
            name=f"attack-lock-{arm}",
            train=train,
            config=ElementalConfig(
                updates=updates,
                device="cuda",
                input_ablation=ablation,
                tau_scale=tau,
                recover_capacity=arm.endswith("-recover"),
            ),
            smoke=ElementalConfig(
                updates=1,
                device="cpu",
                streams=2,
                input_ablation=ablation,
                tau_scale=tau,
            ),
            seeds=(SEED,),
            repo=str(ROOT),
            revision=revision,
            setup=machine_setup(" --extra artifacts"),
            max_hours=hours,
            max_usd=Decimal(str(usd_per_run)),
        )
        for arm, (ablation, tau) in ARMS.items()
    ]
