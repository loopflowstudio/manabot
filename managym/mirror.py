"""Framework-free numerical update for magnetic-mirror search.

The detached simplex optimizer follows Ataraxos supplement S3.7 (7)-(8).
Arena execution and receipt verification use this same implementation; model
execution, world sampling and evidence ownership remain with their callers.
"""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def regularized_update(
    base: FloatArray,
    values: FloatArray,
    reference: FloatArray,
    *,
    alpha: float,
    beta: float,
) -> FloatArray:
    """Maximize p·Q - alpha KL(p||reference) - beta KL(p||base).

    All vectors have shape [legal_actions]. Base zeros restrict the feasible
    simplex; values outside its support may be NaN and are never consulted.
    The reference remains strictly positive. Values are signed expected outcomes;
    beta is the inverse step scale. This is one detached local simplex update,
    not a gradient through a determinized game or an argmax improvement claim.
    """
    if not np.isfinite([alpha, beta]).all() or alpha < 0 or beta <= 0:
        raise ValueError("local update requires finite alpha >= 0 and beta > 0")
    arrays = [np.asarray(item, dtype=np.float64) for item in (base, values, reference)]
    base, values, reference = arrays
    if base.ndim != 1 or not base.size or any(a.shape != base.shape for a in arrays):
        raise ValueError("local update vectors must align with legal offers")
    support = base > 0
    if not np.isfinite(values[support]).all() or any(
        not np.isfinite(a).all() for a in (base, reference)
    ):
        raise ValueError("local update vectors must be finite")
    for probabilities in (base, reference):
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
            raise ValueError(
                "base and reference require normalized nonnegative support"
            )
    if np.any(reference <= 0):
        raise ValueError("reference requires positive support")
    # beta > 0 restricts the feasible simplex to the base support. Missing Q
    # outside it is deliberately unused, never treated as a zero-valued rollout.
    logits = (
        values[support]
        + alpha * np.log(reference[support])
        + beta * np.log(base[support])
    ) / (alpha + beta)
    if not np.isfinite(logits).all():
        raise ValueError("local update logits are nonfinite")
    target = np.exp(logits - logits.max())
    target /= target.sum()
    if np.any(target == 0):
        raise ValueError("local update numerically lost mixed support")
    aligned = np.zeros_like(base)
    aligned[support] = target
    return aligned
