"""Deprecated: moved to `manabot.training.updates.health`."""

from manabot.infra.deprecation import moved

__getattr__ = moved(
    __name__,
    dict.fromkeys(
        (
            "HEALTH_INTERVAL",
            "NumericalHealth",
            "batch_health",
            "OptimizerHealth",
        ),
        "manabot.training.updates.health",
    ),
)
