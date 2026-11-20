"""Deprecated policy inputs v9 and v10: the object rows before input v11.

Input v11 gives each game thing one row: cards outside the battlefield and
stack, permanents with their characteristics, and a stack segment. Checkpoints
trained before it read v9/v10 rows, in which a battlefield object or a stack
spell was also a card row and permanent rows had no characteristics. Native
`agent::legacy_input` writes those rows; this module holds their permanent
width and the deprecation warning.

Removal waits for a v11 pilot at least as strong as the best v9 pilot, from
release 0.4.0 onward, with the `policy_input_version` 9 and 10 branches in
`manabot.env.observation` and `manabot.backends.torch.policy`. MTG-353 owns
the general mechanism for checkpoints across worlds.
"""

from pathlib import Path
import warnings

import pydantic

import managym

# A v9/v10 permanent row ends at `dealt_damage_this_turn`, then validity.
LEGACY_PERMANENT_DIM = len(
    managym.observation_layout(legacy=True)["agent_permanents"]["columns"]
)
REMOVAL_RELEASE = "0.4.0"


def warn_deprecated_input(version: int) -> None:
    """Warn that a v9/v10 configuration selects the deprecated rows."""
    warnings.warn(
        f"policy input v{version} is deprecated until a v11 pilot matches the best v9 pilot; removal no earlier than "
        f"{REMOVAL_RELEASE}; it serves checkpoints trained before input v11",
        DeprecationWarning,
        # Validation reaches here through pydantic and manabot frames at
        # varying depths; attribute the warning to the first caller outside.
        skip_file_prefixes=(
            str(Path(pydantic.__file__).parent),
            str(Path(__file__).parents[1]),
        ),
    )
