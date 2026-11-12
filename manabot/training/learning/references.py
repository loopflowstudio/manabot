"""Deprecated: moved to `manabot.training.updates.references`."""

from manabot.infra.deprecation import moved

__getattr__ = moved(
    __name__,
    {"reference_distribution": "manabot.training.updates.references"},
)
