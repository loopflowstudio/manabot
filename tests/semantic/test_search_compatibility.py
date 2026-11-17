"""Academy Search preserves the arena numerical update without changing its source."""

import numpy as np
import pytest

from manabot.sim.search.local_update_receipt import regularized_update as arena_update
from managym.search import regularized_update


@pytest.mark.parametrize(
    "base,values", [([0.25, 0.75], [0.8, -0.2]), ([1.0, 0.0], [-0.3, float("nan")])]
)
def test_academy_update_matches_arena(base: list[float], values: list[float]) -> None:
    prior = np.array(base)
    scores = np.array(values)
    reference = np.full(len(base), 1 / len(base))
    expected = arena_update(prior, scores, reference, alpha=0.02, beta=1.0)
    np.testing.assert_array_equal(
        regularized_update(prior, scores, reference, alpha=0.02, beta=1.0), expected
    )
