"""Restricted mana reaches the policy input exactly or is refused, never dropped."""

import json

import numpy as np
import pytest

from manabot.env.observation import ObservationEncoder
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
    ],
)
def test_both_encoders_refuse_restrictions_the_columns_cannot_state(
    restricted: dict[str, object], described: str
) -> None:
    env = _env()
    obs = _observation_with(env, [restricted | {"mana": [0, 0, 0, 1, 0, 0]}])
    message = f"cannot represent mana restricted to {described}"
    assert managym.unrepresentable_state(obs) == f"mana restricted to {described}"
    for encode in (
        lambda: env.encode_observation(obs),
        lambda: ObservationEncoder(ObservationConfig()).encode(obs),
    ):
        with pytest.raises(ValueError, match=message):
            encode()


def test_column_restrictions_encode_identically_in_both_encoders() -> None:
    env = _env()
    obs = _observation_with(
        env,
        [
            {"spells": ["Lesson", "Shrine"], "mana": [0, 0, 0, 1, 0, 0]},
            {"spells": ["Ally"], "abilities": ["Ally"], "mana": [0, 0, 0, 0, 2, 0]},
        ],
    )
    assert managym.unrepresentable_state(obs) is None
    expected = ObservationEncoder(ObservationConfig()).encode(obs)
    actual = env.encode_observation(obs)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value, err_msg=key)
