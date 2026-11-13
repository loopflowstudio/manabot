"""Object names and deprecated spellings share one native representation."""

import json
import warnings

import pytest

from manabot.env.match import Match
from manabot.env.observation import ObservationConfig, ObservationEncoder
from manabot.semantic.text_observation import capture_text_decision, submit_response
from manabot.verify.text_observation import collision_branches
import managym
from managym.experience_projection import decision_surface_view, serialize_observation


def test_card_class_alias_warns_on_each_access() -> None:
    for _ in range(2):
        with pytest.warns(
            DeprecationWarning, match=r"managym.Card.*0\.3\.0.*managym.Object"
        ) as caught:
            old = managym.Card
        assert len(caught) == 1
        assert caught[0].filename == __file__
        assert old is managym.Object
    with pytest.raises(AttributeError):
        getattr(managym, "NotAnEngineType")


@pytest.mark.parametrize("seat", ["agent", "opponent"])
def test_object_list_alias_reads_and_writes_same_rows(seat: str) -> None:
    env = managym.Env(seed=42)
    obs, _ = env.reset(Match().to_rust())
    old, new = f"{seat}_cards", f"{seat}_objects"
    rows = obs.agent_objects
    assert isinstance(rows[0], managym.Object)
    with pytest.warns(
        DeprecationWarning, match=rf"Observation\.{old}.*0\.3\.0.*Observation\.{new}"
    ) as caught:
        setattr(obs, old, rows)
    assert len(caught) == 1
    assert [row.id for row in getattr(obs, new)] == [row.id for row in rows]
    with pytest.warns(
        DeprecationWarning, match=rf"Observation\.{old}.*0\.3\.0.*Observation\.{new}"
    ) as caught:
        legacy = getattr(obs, old)
    assert len(caught) == 1
    assert [row.id for row in legacy] == [row.id for row in rows]
    setattr(obs, new, [])
    with pytest.warns(DeprecationWarning):
        assert getattr(obs, old) == []


def test_stack_source_alias_reads_and_writes_same_value() -> None:
    _, env = collision_branches()
    submit_response(env, capture_text_decision(env), "A0")
    actor = env.current_agent_index()
    assert actor is not None
    obs = env.observation_for_player(actor)
    stack = obs.stack_objects[0]
    before = stack.source_registry_key
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert decision_surface_view(env, actor)["stack_objects"]
        assert "Stack (native order" in capture_text_decision(env).text
    with pytest.warns(
        DeprecationWarning,
        match=r"StackObject.source_card_registry_key.*0\.3\.0.*StackObject.source_registry_key",
    ) as caught:
        assert stack.source_card_registry_key == before
    assert len(caught) == 1
    with pytest.warns(
        DeprecationWarning, match=r"0\.3\.0.*StackObject.source_registry_key"
    ) as caught:
        stack.source_card_registry_key = before + 1
    assert len(caught) == 1
    assert stack.source_registry_key == before + 1
    stack.source_registry_key = before
    obs.stack_objects = [stack]
    encoded = json.loads(obs.toJSON())
    assert encoded["stack_objects"][0]["source_card_registry_key"] == before
    assert "source_registry_key" not in encoded["stack_objects"][0]
    assert managym.Observation.from_json(obs.toJSON()).toJSON() == obs.toJSON()


def test_current_consumers_use_object_names_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        env = managym.Env(seed=42)
        obs, _ = env.reset(Match().to_rust())
        encoded = ObservationEncoder(ObservationConfig()).encode(obs)
        assert encoded["agent_cards_valid"].sum() == len(obs.agent_objects)
        assert "agent" in serialize_observation(obs)
        decision = capture_text_decision(env)
        assert decision.representation_version == "manabot-text-7"
        assert "Visible objects" in decision.text
        payload = json.loads(obs.toJSON())
        assert "agent_cards" in payload and "agent_objects" not in payload
        assert managym.Observation.from_json(obs.toJSON()).toJSON() == obs.toJSON()
