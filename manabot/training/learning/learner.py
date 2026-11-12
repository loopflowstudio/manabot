"""Deprecated: moved to `manabot.backends.torch.learner` and `manabot.training.updates.learner`."""

from manabot.infra.deprecation import moved

__getattr__ = moved(
    __name__,
    {"Learner": "manabot.backends.torch.learner"}
    | dict.fromkeys(
        ("AdmittedRollout", "observation_tensors"),
        "manabot.training.updates.learner",
    ),
)
