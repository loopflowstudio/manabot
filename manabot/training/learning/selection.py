"""Deprecated: moved to `manabot.training.updates.selection` and `manabot.rl.selection`."""

from manabot.infra.deprecation import moved

__getattr__ = moved(
    __name__,
    dict.fromkeys(
        ("selected_rows", "selected_moves", "selection_mask", "selected_mean"),
        "manabot.rl.selection",
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
