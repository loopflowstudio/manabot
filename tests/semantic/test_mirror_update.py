"""The shared optimizer preserves zero support and the two-KL optimum."""

import numpy as np
import pytest

from managym.mirror import regularized_update


def test_mirror_update_matches_fixed_numerical_oracle() -> None:
    base = np.array([1 - 2e-6, 1e-6, 1e-6])
    values = np.array([-1.0, 1.0, 1.0])
    reference = np.full(3, 1 / 3)
    result = regularized_update(base, values, reference, alpha=0.02, beta=1)
    np.testing.assert_allclose(
        result, [0.9999813693, 0.00000931535, 0.00000931535], rtol=1e-5
    )


def test_two_kl_optimum_and_shift_invariance() -> None:
    base = np.array([0.2, 0.8])
    reference = np.array([0.5, 0.5])
    values = np.array([-1.0, 1.0])
    result = regularized_update(base, values, reference, alpha=0.3, beta=2)
    # First-order simplex stationarity independently checks the optimizer.
    gradient = (
        values
        - 0.3 * (np.log(result / reference) + 1)
        - 2 * (np.log(result / base) + 1)
    )
    assert gradient[0] == pytest.approx(gradient[1])
    assert np.all(result > 0)
    np.testing.assert_allclose(
        result, regularized_update(base, values + 8, reference, alpha=0.3, beta=2)
    )
    np.testing.assert_array_equal(
        regularized_update(
            np.array([0.0, 1.0]), np.array([np.nan, 1.0]), reference, alpha=0.3, beta=2
        ),
        [0.0, 1.0],
    )


def test_zero_support_update_is_the_restricted_simplex_optimum() -> None:
    base = np.array([0.2, 0.0, 0.8])
    values = np.array([-0.7, np.nan, 0.4])
    reference = np.full(3, 1 / 3)
    target = regularized_update(base, values, reference, alpha=0.3, beta=2)
    active = base > 0
    gradient = (
        values[active]
        - 0.3 * (np.log(target[active] / reference[active]) + 1)
        - 2 * (np.log(target[active] / base[active]) + 1)
    )
    assert gradient[0] == pytest.approx(gradient[1])
    assert target[1] == 0
    assert target.sum() == pytest.approx(1)
    for inaccessible in (-1e10, 1e10, np.nan):
        values[1] = inaccessible
        np.testing.assert_array_equal(
            target, regularized_update(base, values, reference, alpha=0.3, beta=2)
        )


@pytest.mark.parametrize(
    ("base", "values", "reference", "alpha", "beta"),
    [
        ([0.5, 0.5], [0], [0.5, 0.5], 0.02, 1),
        ([0.5, 0.5], [0, np.nan], [0.5, 0.5], 0.02, 1),
        ([0.4, 0.5], [0, 1], [0.5, 0.5], 0.02, 1),
        ([0.5, 0.5], [0, 1], [0, 1], 0.02, 1),
        ([0.5, 0.5], [0, 1], [0.5, 0.5], -1, 1),
        ([0.5, 0.5], [0, 1], [0.5, 0.5], 0.02, 0),
    ],
)
def test_mirror_refuses_invalid_inputs(
    base: list[float],
    values: list[float],
    reference: list[float],
    alpha: float,
    beta: float,
) -> None:
    with pytest.raises(ValueError):
        regularized_update(
            np.array(base),
            np.array(values),
            np.array(reference),
            alpha=alpha,
            beta=beta,
        )
