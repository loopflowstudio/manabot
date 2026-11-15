"""Deprecated policy inputs v9 and v10: the object rows before input v11.

Input v11 gives each game thing one row: cards outside the battlefield and
stack, permanents with their characteristics, and a stack segment. Checkpoints
trained before it read v9/v10 rows, in which a battlefield object or a stack
spell was also a card row and permanent rows had no characteristics. The
encoder's v9/v10 path reads those rows through this module, mirroring native
`agent::legacy_input`; nothing else depends on it.

Removal release: 0.4.0, with the `policy_input_version` 9 and 10 branches in
`manabot.env.observation` and `manabot.backends.torch.policy`. MTG-353 owns
the general mechanism for checkpoints across worlds.
"""

from pathlib import Path
import warnings

import pydantic

import managym

# A v9/v10 permanent row ends at `dealt_damage_this_turn`, then validity.
LEGACY_PERMANENT_DIM = 38
REMOVAL_RELEASE = "0.4.0"


def warn_deprecated_input(version: int) -> None:
    """Warn that a v9/v10 configuration selects the deprecated rows."""
    warnings.warn(
        f"policy input v{version} is deprecated and will be removed in "
        f"{REMOVAL_RELEASE}; it serves checkpoints trained before input v11",
        DeprecationWarning,
        # Validation reaches here through pydantic and manabot frames at
        # varying depths; attribute the warning to the first caller outside.
        skip_file_prefixes=(
            str(Path(pydantic.__file__).parent),
            str(Path(__file__).parents[1]),
        ),
    )


def legacy_card_lists(
    obs: managym.Observation,
) -> tuple[list[managym.Card], list[managym.Card]]:
    """Each side's v9/v10 card rows: its v11 cards, then its stack spells (by
    owner, top first), then its battlefield objects (by controller)."""
    return obs.legacy_card_lists()
