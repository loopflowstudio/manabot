"""Deprecated: credit assignment moved to `manabot.rl.returns`."""

from manabot.infra.deprecation import moved

__getattr__ = moved(
    __name__,
    dict.fromkeys(
        ("terminal_distances", "reverse_waves", "transition_gae"), "manabot.rl.returns"
    ),
)
