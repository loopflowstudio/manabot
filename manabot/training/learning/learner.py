"""Deprecated: moved to `manabot.training.updates.learner`."""

from manabot.infra.deprecation import moved

__getattr__ = moved(
    __name__,
    dict.fromkeys(
        ("Learner", "AdmittedRollout", "observation_tensors"),
        "manabot.training.updates.learner",
    ),
)
