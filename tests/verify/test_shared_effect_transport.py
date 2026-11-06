"""Retained failures cross the prospective gate through real native buffers."""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel
import pytest

from manabot.env.failures import GameInputs
from manabot.env.observation import ObservationEncoder
from manabot.infra.config import ObservationConfig
from manabot.semantic.text_observation import capture_text_decision
from manabot.verify.representation import (
    advance_to_next_turn,
    aged_grant_root,
    source_effect_root,
)
import managym


class RetainedPosition(BaseModel):
    source: str
    sha256: str
    source_commit: str | None
    game: GameInputs
    skip_trivial: bool
    seed_stride: int
    fragments: int
    payloads: int
    memberships: int


@pytest.mark.parametrize("fragments", (140, 270))
@pytest.mark.parametrize("closed", (False, True))
def test_retained_drill_prefix_admits_shared_effects(
    fragments: int, closed: bool
) -> None:
    receipt = RetainedPosition.model_validate_json(
        (
            Path(__file__).parents[1]
            / "fixtures"
            / "shared_effects"
            / f"drill-{fragments}.json"
        ).read_bytes()
    )
    game = receipt.game
    players = [
        player.model_copy(update={"disclosure": "closed"}).native()
        if closed
        else player.native()
        for player in game.players
    ]
    scalar = managym.Env(seed=game.seed, skip_trivial=receipt.skip_trivial)
    scalar.reset(players)
    config = ObservationConfig(policy_input_version=7)
    encoder = ObservationEncoder(config)
    buffers = encoder.allocate(1)
    buffers.update(
        rewards=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )
    vector = managym.VectorEnv(
        num_envs=1,
        seed=game.seed,
        seed_stride=receipt.seed_stride,
        skip_trivial=receipt.skip_trivial,
    )
    vector.set_buffers(buffers)
    vector.reset_all(players)
    vector.reset_seeds_into_buffers([game.seed])
    for index, action in enumerate(game.actions):
        assert isinstance(action, int)
        if index == len(game.actions) - 1 and not closed:
            assert scalar.semantic_decision_frame_json() == game.frame
            assert vector.semantic_decision_frames_json()[0] == game.frame
        acting_observation, *_ = scalar.step(action)
        vector.step_into_buffers([action])
    actor = scalar.current_agent_index()
    assert actor is not None
    for viewer in (0, 1):
        obs = (
            acting_observation
            if viewer == actor
            else scalar.observation_for_player(viewer)
        )
        assert obs.shared_effects is not None
        assert len(obs.shared_effects) == receipt.payloads
        assert (
            sum(len(record.members) for record in obs.shared_effects)
            == receipt.memberships
        )
        assert all(len(record.effect.keywords) == 2 for record in obs.shared_effects)
        reference = encoder.encode(obs)
        if closed:
            assert not obs.opponent.main_counts
            assert not reference["decklists"][1].any()
        native = scalar.encode_observation(obs, 7)
        into = encoder.allocate()
        scalar.encode_observation_into(obs, into)
        for key in reference:
            np.testing.assert_array_equal(reference[key], native[key], err_msg=key)
            np.testing.assert_array_equal(reference[key], into[key], err_msg=key)
            if viewer == actor:
                np.testing.assert_array_equal(
                    reference[key], buffers[key][0], err_msg=key
                )
        assert np.count_nonzero(native["effects"][:, -1]) == receipt.payloads
        assert np.count_nonzero(native["effect_members"][:, 0]) == receipt.memberships
        assert not (native["facts"][:, 1] == 6).any()
        with pytest.raises(ValueError, match=f"facts {fragments} >"):
            scalar.encode_observation(
                obs,
                5 if fragments == 140 else 6,
                **({} if fragments == 140 else {"max_facts": 256}),
            )


def test_source_dependent_shared_transport_has_exact_public_join() -> None:
    env = source_effect_root()
    encoder = ObservationEncoder(ObservationConfig(policy_input_version=7))
    for viewer in (0, 1):
        obs = env.observation_for_player(viewer)
        assert obs.shared_effects is not None
        record = next(
            record
            for record in obs.shared_effects
            if record.effect.duration == "WhileYouControlSource"
        )
        assert record.source is not None and record.source.incarnation is not None
        expected = encoder.encode(obs)
        actual = env.encode_observation(obs, 7)
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key], err_msg=key)
        row = actual["effects"][actual["effects"][:, -1] == 1][0]
        assert row[8] == 3 and row[12] > 0
        assert row[12] == encoder.object_to_index[record.source.object_id] + 1


def _history(env: managym.Env, viewer: int) -> np.ndarray:
    capacity = ObservationEncoder(ObservationConfig()).cards_per_player
    return np.array(
        env.observation_for_player(viewer).encode_policy_history(32, capacity)
    )


def _effect_durations(env: managym.Env, viewer: int) -> set[float]:
    headers = env.encode_observation(env.observation_for_player(viewer), 7)["effects"]
    return set(headers[headers[:, -1] == 1, 8].tolist())


def test_effect_older_than_history_window_survives_only_in_current_state() -> None:
    """History cannot say which grant cleanup will end; v7 headers can.

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
    for encode in (
        lambda: env.encode_observation(old, 7),
        lambda: ObservationEncoder(ObservationConfig(policy_input_version=7)).encode(
            old
        ),
    ):
        with pytest.raises(ValueError, match="provenance"):
            encode()
    assert "shared_effects" not in json.loads(old.toJSON())


@pytest.mark.parametrize(
    "field,limit", (("effects", 128), ("effect_details", 512), ("effect_members", 2048))
)
def test_shared_transport_rejects_each_capacity_without_truncation(
    field: str, limit: int
) -> None:
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
    for encode in (
        lambda: env.encode_observation(obs, 7),
        lambda: ObservationEncoder(ObservationConfig(policy_input_version=7)).encode(
            obs
        ),
    ):
        with pytest.raises(ValueError, match=field):
            encode()


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
    native = env.encode_observation(obs, 7)
    reference = ObservationEncoder(ObservationConfig(policy_input_version=7)).encode(
        obs
    )
    for key in reference:
        np.testing.assert_array_equal(reference[key], native[key], err_msg=key)
    np.testing.assert_array_equal(native["effects"][0, 5:9], [0, 1, 1, 3])
    assert native["effects"][0, 14] == 5
    np.testing.assert_array_equal(native["effect_details"][:5, 0], [1, 1, 2, 2, 3])


def test_shared_capture_and_text_ignore_hidden_hand_permutations() -> None:
    env = source_effect_root()
    viewer = env.current_agent_index()
    assert viewer is not None
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
        for key, expected in env.encode_observation(
            env.observation_for_player(viewer), 7
        ).items():
            np.testing.assert_array_equal(
                expected,
                branch.encode_observation(branch.observation_for_player(viewer), 7)[
                    key
                ],
                err_msg=key,
            )
    assert changed > 0


@pytest.mark.parametrize("version", range(1, 7))
def test_frozen_inputs_ignore_shared_projection(version: int) -> None:
    env = source_effect_root()
    obs = env.observation_for_player(0)
    payload = json.loads(obs.toJSON())
    payload.pop("shared_effects")
    historical = managym.Observation.from_json(json.dumps(payload))
    encoder = ObservationEncoder(ObservationConfig(policy_input_version=version))
    for key, expected in encoder.encode(historical).items():
        np.testing.assert_array_equal(expected, encoder.encode(obs)[key], err_msg=key)
        np.testing.assert_array_equal(
            expected, env.encode_observation(obs, version)[key], err_msg=key
        )
