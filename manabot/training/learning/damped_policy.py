"""Deprecated: the damped policy rule moved to `manabot.training.loops.damped_policy`."""

from manabot.infra.deprecation import moved

__getattr__ = moved(
    __name__,
    {
        "update_damped_iteration": "manabot.training.loops.damped_policy",
        "damped_policy_loss": "manabot.training.loops.damped_policy",
        "DampedTargets": "manabot.training.loops.damped_policy",
        "categorical_lambda_returns": "manabot.rl.returns",
    },
)
