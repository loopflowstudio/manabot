"""Restricted mana reaches the policy input exactly or is refused, never dropped."""

import json

import numpy as np
import pytest

from manabot.env.observation import PLAYER_RESTRICTED_INDEX, ObservationSpace
from manabot.infra.config import ObservationConfig
import managym


def _observation_with(
    env: managym.Env, restricted: list[dict[str, object]]
) -> managym.Observation:
    payload = json.loads(env.observation_for_player(0).toJSON())
    payload["agent"]["restricted_mana"] = restricted
    return managym.Observation.from_json(json.dumps(payload))


def _env() -> managym.Env:
    env = managym.Env(seed=163, skip_trivial=False)
    env.reset(
        [
            managym.authored_deck_setup("elemental-temur-v1", key)
            for key in ("temur_left", "temur_right")
        ]
    )
    return env


@pytest.mark.parametrize(
    "restricted,described",
    [
        ({"spells": ["Ogre"]}, "Ogre spells"),
        # The Ally column also pays abilities of Ally sources.
        ({"spells": ["Ally"]}, "Ally spells"),
        ({"abilities": ["Lesson"]}, "abilities of Lesson sources"),
        ({"spells": ["Lesson", "Ogre"]}, "Lesson or Ogre spells"),
        # Classes within the column subtypes can still share one encoding:
        # Lesson beside Shrine-or-Ally sums like Lesson-or-Shrine beside Ally.
        (
            {"spells": ["Ally", "Shrine"], "abilities": ["Ally"]},
            "Ally or Shrine spells or abilities of Ally sources",
        ),
        ({"spells": ["Ally", "Lesson"]}, "Ally or Lesson spells"),
    ],
)
def test_policy_input_refuses_restrictions_the_columns_cannot_state(
    restricted: dict[str, object], described: str
) -> None:
    env = _env()
    obs = _observation_with(
        env,
        [
            {"spells": ["Lesson"], "mana": [0, 0, 0, 1, 0, 0]},
            restricted | {"mana": [0, 0, 0, 1, 0, 0]},
        ],
    )
    message = f"cannot represent mana restricted to {described}"
    assert managym.unrepresentable_state(obs) == f"mana restricted to {described}"
    with pytest.raises(ValueError, match=message):
        ObservationSpace(ObservationConfig()).encode(obs)


def test_column_restrictions_encode_in_each_subtype_column() -> None:
    env = _env()
    obs = _observation_with(
        env,
        [
            {"spells": ["Lesson", "Shrine"], "mana": [0, 0, 0, 1, 0, 0]},
            {"spells": ["Ally"], "abilities": ["Ally"], "mana": [0, 0, 0, 0, 2, 0]},
        ],
    )
    assert managym.unrepresentable_state(obs) is None
    encoded = ObservationSpace(ObservationConfig()).encode(obs)
    columns = encoded["agent_player"][0, PLAYER_RESTRICTED_INDEX:][:18].reshape(3, 6)
    # Lesson, Shrine and Ally columns, each W U B R G C scaled by 1/10: the
    # Lesson-or-Shrine red counts for both subtypes, the Ally green for Ally.
    np.testing.assert_allclose(
        columns,
        [[0, 0, 0, 0.1, 0, 0], [0, 0, 0, 0.1, 0, 0], [0, 0, 0, 0, 0.2, 0]],
        rtol=1e-6,
    )
