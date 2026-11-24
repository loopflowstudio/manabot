"""Bounded damped-policy recipe for training continuation fixtures."""

from collections.abc import Mapping

import numpy as np

from manabot.recipes import baseline
from manabot.training.loops.self_play import SelfPlay


def bounded_self_play() -> SelfPlay:
    return baseline.settings(
        baseline.BaselineConfig(backend="mlx", device="mps")
    ).model_copy(
        update={
            "streams": 2,
            "transitions": 8,
            "learning_seats": "one",
            "batch_closure": "exact-quota",
            "bucketed_shapes": True,
            "checkpoint_every": 1,
        }
    )


def same_arrays(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> None:
    assert left.keys() == right.keys()
    for name, array in left.items():
        np.testing.assert_array_equal(array, right[name], strict=True, err_msg=name)
