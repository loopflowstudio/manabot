"""Single-threaded pools for standalone native inspection children.

Managed evaluations use Etude's own CPU/thread allocation.
"""

import os

POOL_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


def child_environment() -> dict[str, str]:
    return {**os.environ, **dict.fromkeys(POOL_VARIABLES, "1")}
