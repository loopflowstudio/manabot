"""The published projection owns effective values independently of card identity."""

import json

import pytest

from manabot.verify.text_observation import priority_root
import managym


def test_permanent_keeps_its_effective_characteristics_on_round_trip() -> None:
    env = priority_root((), ("Gray Ogre",))
    obs = env.observation_for_player(env.current_agent_index())
    restored = managym.Observation.from_json(obs.toJSON())
    permanent = restored.agent_permanents[0]
    assert permanent.name is None  # No effect has overridden its name.
    assert permanent.characteristics.name == "Gray Ogre"
    assert permanent.characteristics.power == 2
    assert permanent.object_id == obs.agent_permanents[0].object_id
    assert restored.toJSON() == obs.toJSON()


def test_historical_permanent_without_reconstructible_identity_is_refused() -> None:
    env = priority_root((), ("Gray Ogre",))
    obs = env.observation_for_player(env.current_agent_index())
    wire = json.loads(obs.toJSON())
    del wire["agent_permanents"][0]["card"]
    with pytest.raises(ValueError, match="identity cannot be reconstructed"):
        managym.Observation.from_json(json.dumps(wire))


def test_card_characteristics_are_read_only() -> None:
    env = priority_root(("Lightning Bolt",), ())
    obs = env.observation_for_player(env.current_agent_index())
    card = obs.agent_cards[0]
    assert card.characteristics.name == "Lightning Bolt"
    with pytest.raises(AttributeError):
        card.characteristics.power = 99
