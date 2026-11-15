"""The published projection owns effective values independently of card identity."""

import json

import pytest

from manabot.verify.text_observation import priority_root
import managym


def test_permanent_characteristics_and_compatibility_getters_share_one_source() -> None:
    env = priority_root((), ("Gray Ogre",), opponent=("Gray Ogre",))
    obs = env.observation_for_player(env.current_agent_index())
    permanent = obs.agent_permanents[0]
    assert permanent.characteristics.name == "Gray Ogre"
    assert permanent.name is None  # The old name accessor still means override.
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        old = permanent.card
    assert old is not None
    assert old.id == permanent.object_id
    assert old.owner_id == permanent.owner_id
    assert old.is_token == permanent.is_token
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        permanent.power = 7
    assert permanent.characteristics.power == 7
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        assert permanent.power == 7
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        assert permanent.card.characteristics.power == 7
    obs.agent_permanents = [permanent]
    restored = managym.Observation.from_json(obs.toJSON())
    assert restored.agent_permanents[0].characteristics.power == 7
    assert restored.toJSON() == obs.toJSON()


def test_historical_permanent_without_reconstructible_identity_is_refused() -> None:
    env = priority_root((), ("Gray Ogre",))
    obs = env.observation_for_player(env.current_agent_index())
    wire = json.loads(obs.toJSON())
    del wire["agent_permanents"][0]["card"]
    with pytest.raises(ValueError, match="identity cannot be reconstructed"):
        managym.Observation.from_json(json.dumps(wire))


def test_card_characteristics_are_read_only_snapshots_with_derived_aliases() -> None:
    env = priority_root(("Lightning Bolt",), ())
    obs = env.observation_for_player(env.current_agent_index())
    card = obs.agent_cards[0]
    with pytest.warns(DeprecationWarning, match="0.4.0"):
        assert card.name == card.characteristics.name
    with pytest.raises(AttributeError):
        card.characteristics.power = 99
