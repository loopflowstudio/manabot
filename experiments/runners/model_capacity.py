"""Construct the registered width/depth examples without allocating a study.

The caller supplies a complete baseline and owns workload, seeds and evaluation.
These are independent model variants, not compatible checkpoint weight transfers.
"""

from manabot.training.models import TrainingRegime
from manabot.training.recipes import with_capacity


def regimes(base: TrainingRegime) -> dict[str, TrainingRegime]:
    """Hold information, pooling, learning and budget fixed across three capacities."""
    return {
        "w64-d1": with_capacity(base, id="w64-d1", width=64, depth=1, heads=4),
        "w64-d2": with_capacity(base, id="w64-d2", width=64, depth=2, heads=4),
        "w128-d2": with_capacity(base, id="w128-d2", width=128, depth=2, heads=4),
    }
