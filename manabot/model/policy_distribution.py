"""Deprecated: the policy distribution moved to `manabot.backends.torch.rl`."""

from manabot.infra.deprecation import moved

__getattr__ = moved(
    __name__,
    dict.fromkeys(
        (
            "legal_mask",
            "policy_logs",
            "check_probabilities",
            "check_logs",
            "check_behavior",
            "NumericalError",
            "require",
            "clipped_surrogate",
            "reverse_kl",
        ),
        "manabot.backends.torch.rl",
    ),
)
