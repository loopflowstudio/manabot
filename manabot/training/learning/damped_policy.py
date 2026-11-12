"""Deprecated: the damped policy update moved to `manabot.training.updates.damped_policy`."""

from manabot.infra.deprecation import moved

__getattr__ = moved(
    __name__,
    dict.fromkeys(
        ("update_damped_iteration", "damped_policy_loss", "DampedTargets"),
        "manabot.training.updates.damped_policy",
    )
    | {"categorical_lambda_returns": "manabot.backends.torch.rl"},
)
