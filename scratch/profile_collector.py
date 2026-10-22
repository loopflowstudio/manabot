"""Bounded diagnostic slice using the unchanged positive-control implementation."""

import cProfile
from pathlib import Path
import pstats
import time

import numpy as np

from manabot.sim.net_opponent import NetOpponentTrainer, RolloutBatch
from manabot.training.models import Learning
from manabot.training.selection import UpdateDiagnostics
import manabot.verify.learning_control as control


class SliceComplete(Exception):
    pass


original = control.update_iteration
updates = 0


def update(
    trainer: NetOpponentTrainer,
    batch: RolloutBatch,
    learning: Learning,
    progress: float,
    rng: np.random.Generator,
) -> UpdateDiagnostics:
    global updates
    result = original(trainer, batch, learning, progress, rng)
    updates += 1
    print(f"completed update {updates}", flush=True)
    if updates == 2:
        raise SliceComplete
    return result


control.update_iteration = update
profile = cProfile.Profile()
try:
    profile.runcall(
        control.run_positive_control,
        Path(".runs/learnability/collector-profile"),
        deadline=time.perf_counter() + 90,
    )
except SliceComplete:
    pass
finally:
    profile.dump_stats("scratch/collector-profile.pstats")
    pstats.Stats(profile).strip_dirs().sort_stats("cumulative").print_stats(35)
