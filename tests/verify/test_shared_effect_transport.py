"""Controlled Drill games cross the pilot capacity boundaries through native buffers."""

import json

import numpy as np
import pytest

from manabot.env.observation import ObservationSpace, encode_observation
from manabot.env.shared_effects import SHARED_EFFECT_SHAPES
from manabot.infra.config import ObservationConfig
from manabot.semantic.text_observation import capture_text_decision
from manabot.verify.representation import (
    advance_to_next_turn,
    aged_grant_root,
    source_effect_root,
)
import managym
from tests.current_capacity import capacity_position

EFFECTS = managym.observation_layout()["effects"]["columns"]


def _effect(name: str) -> int:
    return EFFECTS[name]["index"]


def _permanent_index(config: ObservationConfig, side: int, slot: int) -> int:
    """Zero-based object index of a v11 permanent row: players, cards, then permanents."""
    return (
        2
        + 2 * config.max_cards_per_player
        + side * config.max_permanents_per_player
        + slot
    )


@pytest.mark.parametrize("fragments", (140, 270))
@pytest.mark.parametrize("closed", (False, True))
def test_controlled_drill_prefix_admits_shared_effects(
    fragments: int, closed: bool
) -> None:
    game = capacity_position("drill140" if fragments == 140 else "drill270")
    payloads = 7 if fragments == 140 else 9
    memberships = fragments // 2
    players = game.setup(closed=closed)
    scalar = managym.Env(seed=game.seed, skip_trivial=True)
    scalar_obs, _ = scalar.reset(players)
    assert scalar_obs.decklists_closed is closed
    space = ObservationSpace()
    buffers = space.allocate(1)
    buffers.update(
        rewards=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )
    vector = managym.VectorEnv(
        num_envs=1,
        seed=game.seed,
        skip_trivial=True,
    )
    vector.set_buffers(buffers)
    ((vector_obs, _),) = vector.reset_all(players)
    assert vector_obs.decklists_closed is closed
    vector.reset_seeds_into_buffers([game.seed])
    assert scalar.observation_for_player(0).decklists_closed is closed
    assert vector.current_player_configs()[0].decklists == (
        "closed" if closed else "open"
    )
    for action in game.actions[:-1]:
        scalar.step(action)
        vector.step_into_buffers([action])
    assert (
        vector.semantic_decision_frames_json()[0]
        == scalar.semantic_decision_frame_json()
    )
    acting_observation, *_ = scalar.step(game.actions[-1])
    vector.step_into_buffers([game.actions[-1]])
    actor = scalar.current_agent_index()
    assert actor is not None
    for viewer in (0, 1):
        obs = (
            acting_observation
            if viewer == actor
            else scalar.observation_for_player(viewer)
        )
        assert obs.decklists_closed is closed
        assert obs.shared_effects is not None
        assert len(obs.shared_effects) == payloads
        assert sum(len(record.members) for record in obs.shared_effects) == memberships
        assert all(len(record.effect.keywords) == 2 for record in obs.shared_effects)
        encoded = encode_observation(space, obs)
        if closed:
            assert not obs.opponent.main_counts
            assert not encoded["decklists"][1].any()
        if viewer == actor:
            for key in encoded:
                np.testing.assert_array_equal(
                    encoded[key], buffers[key][0], err_msg=key
                )
        assert np.count_nonzero(encoded["effects"][:, -1]) == payloads
        assert np.count_nonzero(encoded["effect_members"][:, 0]) == memberships
        # Shared effects are effect rows, never per-object effect facts.
        assert not (encoded["facts"][:, 1] == 6).any()


def test_source_dependent_shared_transport_has_exact_public_join() -> None:
    env = source_effect_root()
    config = ObservationConfig()
    space = ObservationSpace(config)
    for viewer in (0, 1):
        obs = env.observation_for_player(viewer)
        assert obs.shared_effects is not None
        record = next(
            record
            for record in obs.shared_effects
            if record.effect.duration == "WhileYouControlSource"
        )
        assert record.source is not None and record.source.incarnation is not None
        encoded = encode_observation(space, obs)
        row = encoded["effects"][encoded["effects"][:, _effect("valid")] == 1][0]
        assert row[_effect("duration")] == 3
        # The source join is the one-based index of the source's permanent row.
        (side, slot), *_ = [
            (side, slot)
            for side, permanents in enumerate(
                (obs.agent_permanents, obs.opponent_permanents)
            )
            for slot, permanent in enumerate(permanents)
            if permanent.id == record.source.object_id
        ]
        assert row[_effect("source")] == _permanent_index(config, side, slot) + 1


def _history(env: managym.Env, viewer: int) -> np.ndarray:
    """The viewer's valid public history v1 rows."""
    encoded = encode_observation(
        ObservationSpace(ObservationConfig(policy_history_version=1)),
        env.observation_for_player(viewer),
    )
    return encoded["events"][encoded["events_valid"] == 1]


def _effect_durations(env: managym.Env, viewer: int) -> set[float]:
    headers = encode_observation(
        ObservationSpace(), env.observation_for_player(viewer)
    )["effects"]
    valid = headers[:, _effect("valid")] == 1
    return set(headers[valid, _effect("duration")].tolist())


def test_effect_older_than_history_window_survives_only_in_current_state() -> None:
    """History cannot say which grant cleanup will end; effect headers can.

    The roots differ in payload and target as well as duration, so this shows
    only that the window cannot separate them. Equal effective state with
    differing futures is the rules anchor
    `cr_514_2_an_end_of_turn_record_ends_at_cleanup`.
    """
    end_of_turn = aged_grant_root("EndOfTurn", fill=False)
    indefinite = aged_grant_root("Indefinite", fill=False)
    assert not np.array_equal(_history(end_of_turn, 0), _history(indefinite, 0))

    end_of_turn = aged_grant_root("EndOfTurn")
    indefinite = aged_grant_root("Indefinite")
    for viewer in (0, 1):
        window = _history(end_of_turn, viewer)
        assert window.shape[0] == 32
        np.testing.assert_array_equal(window, _history(indefinite, viewer))
        assert _effect_durations(end_of_turn, viewer) == {1.0}
        assert _effect_durations(indefinite, viewer) == {2.0}

    advance_to_next_turn(end_of_turn)
    advance_to_next_turn(indefinite)
    for viewer in (0, 1):
        assert _effect_durations(end_of_turn, viewer) == set()
        assert _effect_durations(indefinite, viewer) == {2.0}


def test_historical_projection_cannot_invent_shared_provenance() -> None:
    env = source_effect_root()
    payload = json.loads(env.observation_for_player(0).toJSON())
    payload.pop("shared_effects")
    old = managym.Observation.from_json(json.dumps(payload))
    with pytest.raises(ValueError, match="provenance"):
        encode_observation(ObservationSpace(), old)
    assert "shared_effects" not in json.loads(old.toJSON())


@pytest.mark.parametrize("field", ("effects", "effect_details", "effect_members"))
def test_shared_transport_rejects_each_capacity_without_truncation(field: str) -> None:
    limit = SHARED_EFFECT_SHAPES[field][0]
    env = source_effect_root()
    payload = json.loads(env.observation_for_player(0).toJSON())
    record = payload["shared_effects"][0]
    if field == "effects":
        payload["shared_effects"] = [record] * (limit + 1)
    elif field == "effect_details":
        record["effect"]["subtypes"] = ["Ally"] * (limit + 1)
    else:
        record["members"] = [record["members"][0]] * (limit + 1)
    obs = managym.Observation.from_json(json.dumps(payload))
    with pytest.raises(
        managym.ObservationCapacity,
        match=f"observation capacity exceeded: {field} {limit + 1} > {limit}",
    ):
        encode_observation(ObservationSpace(), obs)


def test_shared_payload_preserves_absence_zero_duration_and_every_detail() -> None:
    env = source_effect_root()
    payload = json.loads(env.observation_for_player(0).toJSON())
    record = payload["shared_effects"][0]
    effect = record["effect"]
    effect.pop("power", None)
    effect.update(
        toughness=0,
        amount=3,
        keywords=["Flying", "Hexproof"],
        subtypes=["Ally", "Bird"],
        color="Red",
    )
    obs = managym.Observation.from_json(json.dumps(payload))
    native = encode_observation(ObservationSpace(), obs)
    np.testing.assert_array_equal(native["effects"][0, 5:9], [0, 1, 1, 3])
    assert native["effects"][0, 14] == 5
    np.testing.assert_array_equal(native["effect_details"][:5, 0], [1, 1, 2, 2, 3])


def test_shared_capture_and_text_ignore_hidden_hand_permutations() -> None:
    env = source_effect_root()
    viewer = env.current_agent_index()
    assert viewer is not None
    space = ObservationSpace()
    original = capture_text_decision(env)
    assert "exact source" in original.text and "timestamp rank" in original.text
    assert original.text.count("Effect 1:") == 1
    changed = 0
    for seed in range(4):
        branch = env.clone_env()
        before = branch.state_digest()
        branch.determinize(seed, viewer)
        changed += branch.state_digest() != before
        decision = capture_text_decision(branch)
        assert decision.text == original.text
        assert decision.capture_json == original.capture_json
        determinized = encode_observation(space, branch.observation_for_player(viewer))
        for key, expected in encode_observation(
            space, env.observation_for_player(viewer)
        ).items():
            np.testing.assert_array_equal(expected, determinized[key], err_msg=key)
    assert changed > 0
