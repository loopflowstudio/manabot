"""Deprecated: moved to `manabot.training.updates.selection` and `manabot.backends.torch.rl`."""

from manabot.infra.deprecation import moved

__getattr__ = moved(
    __name__,
    dict.fromkeys(
        ("selected_rows", "selected_moves", "selection_mask", "selected_mean"),
        "manabot.backends.torch.rl",
    )
    | dict.fromkeys(
        (
            "SelectionGroup",
            "selection_diagnostics",
            "ValueScale",
            "value_scale",
            "UpdateDiagnostics",
        ),
        "manabot.training.updates.selection",
    ),
)
