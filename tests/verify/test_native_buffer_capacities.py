"""Configured tensor axes govern encoding at live and historical capacity boundaries."""

import json
from pathlib import Path

import numpy as np
from pydantic import ValidationError
import pytest

from manabot.env.failures import (
    CaptureScope,
    RecordedVectorEnv,
    ScalarCapture,
    capture_scope,
)
from manabot.env.observation import (
    ACTION_MODE_INDEX,
    ObservationSpace,
    step_buffers,
)
from manabot.infra.config import ObservationConfig
from manabot.policy.spans import ObservationSelection
from manabot.training.rollout.producers import ShardedVectorEnv
import managym
from managym.recording import GameInputs
from tests.current_capacity import capacity_position
from tests.fixtures.observation_rows import observation_arrays, vector_arrays


def _players() -> managym.GameSetup:
    return managym.GameSetup(
        [
            managym.PlayerConfig(name, {"Mountain": 10, "Gray Ogre": 20})
            for name in ("a", "b")
        ]
    )


def _encode_everywhere(
    space: ObservationSpace,
    obs: managym.Observation,
    vector: managym.VectorEnv | None = None,
) -> dict[str, np.ndarray]:
    """The padded fields of `obs`, checked against the vector's first row."""
    expected = observation_arrays(space, obs)
    if vector is not None:
        published = vector_arrays(space, vector.observation_rows())
        for key, value in expected.items():
            np.testing.assert_array_equal(published[key][0], value, err_msg=key)
    return expected


def test_native_encoded_arrays_are_writable_and_independent() -> None:
    space = ObservationSpace()
    scalar = managym.Env(seed=213, skip_trivial=True)
    obs, _ = scalar.reset(_players())
    first = observation_arrays(space, obs)
    second = observation_arrays(space, obs)
    assert first.keys() == second.keys() == space.shapes.keys()
    for key, value in second.items():
        expected = value.copy()
        assert first[key].shape == space.shapes[key]
        assert first[key].dtype == space.dtypes[key]
        assert first[key].flags.writeable and first[key].flags.c_contiguous
        first[key].fill(19)
        np.testing.assert_array_equal(value, expected, err_msg=key)


@pytest.mark.parametrize("capacity", [40, 41, 64])
def test_land_growth_preserves_every_permanent(capacity: int) -> None:
    game = capacity_position("land")
    players = game.setup()
    scalar = managym.Env(seed=game.seed, skip_trivial=True)
    vector = managym.VectorEnv(
        num_envs=1,
        seed=game.seed,
        skip_trivial=True,
        player_policies=("agent", "agent"),
    )
    space = ObservationSpace(
        ObservationConfig(
            max_cards_per_player=128,
            max_permanents_per_player=capacity,
            max_actions=64,
            action_overflow="first-legal",
        )
    )
    vector.set_observation_layout(
        step_buffers(1), layout=space.layout, action_overflow="first-legal"
    )
    scalar.reset(players)
    vector.reset_all_into_buffers(players)
    for action in game.actions[:-1]:
        scalar.step(action)
        vector.step_into_buffers([action])
    frame = scalar.semantic_decision_frame_json()
    assert vector.semantic_decision_frames_json()[0] == frame
    last = game.actions[-1]
    assert isinstance(last, int)
    obs, *_ = scalar.step(last)
    assert max(len(obs.agent_permanents), len(obs.opponent_permanents)) == 41
    if capacity == 40:
        message = "observation capacity exceeded: opponent_permanents 41 > 40"
        with pytest.raises(managym.ObservationCapacity, match=message):
            observation_arrays(space, obs)
        with pytest.raises(Exception, match=message):
            vector.step_into_buffers([last])
    else:
        vector.step_into_buffers([last])
        reference = _encode_everywhere(space, obs, vector)
        assert reference["agent_permanents_valid"].sum() == len(obs.agent_permanents)
        assert reference["opponent_permanents_valid"].sum() == len(
            obs.opponent_permanents
        )


@pytest.mark.parametrize("workers", [1, 2])
def test_permanent_overflow_isolates_only_its_stream(workers: int) -> None:
    game = capacity_position("land")
    space = ObservationSpace(
        ObservationConfig(
            max_cards_per_player=128,
            max_permanents_per_player=40,
            max_actions=64,
            action_overflow="first-legal",
        )
    )
    vector = (
        RecordedVectorEnv(num_envs=2, seed=game.seed, skip_trivial=True)
        if workers == 1
        else ShardedVectorEnv(num_envs=2, seed=game.seed, workers=workers)
    )
    try:
        buffers = step_buffers(2)
        vector.set_observation_layout(
            buffers, layout=space.layout, action_overflow="first-legal"
        )
        vector.reset_each_into_buffers([game.setup(), _players()])
        scalar = managym.Env(seed=game.seed, skip_trivial=True)
        scalar.reset(game.setup())
        for action in game.actions[:-1]:
            scalar.step(action)
            vector.step_into_buffers([action, 0], [True, False])
        frame = scalar.semantic_decision_frame_json()
        if isinstance(vector, RecordedVectorEnv):
            assert vector.semantic_decision_frames_json()[0] == frame
        control = managym.Env(seed=game.seed + 1, skip_trivial=True)
        control.reset(_players())
        obs, *_ = control.step(0)
        vector.step_isolated_into_buffers([game.actions[-1], 0], [True, True], True)
        failed, healthy = vector.get_last_info()
        assert failed["game_error_kind"] == "resource_limit"
        assert (
            failed["game_error"]
            == "observation capacity exceeded: opponent_permanents 41 > 40"
        )
        assert failed["game_error_frame"] == frame
        assert "game_error" not in healthy
        assert not buffers["terminated"].any() and not buffers["truncated"].any()
        assert not buffers["rewards"].any()
        reference = observation_arrays(space, obs)

        def healthy_row() -> ObservationSelection:
            return ObservationSelection(space, (vector.observation_rows()[1],))

        for key, value in reference.items():
            np.testing.assert_array_equal(
                healthy_row().project(key)[0], value, err_msg=key
            )
        vector.load_slots_into_buffers([managym.SlotLoad.fresh(0)])
        assert vector.current_game_seeds() == [game.seed + 2, game.seed + 1]
        if isinstance(vector, RecordedVectorEnv):
            assert (
                vector.semantic_decision_frames_json()[1]
                == control.semantic_decision_frame_json()
            )
        for key, value in reference.items():
            np.testing.assert_array_equal(
                healthy_row().project(key)[0], value, err_msg=key
            )
    finally:
        vector.close()


@pytest.mark.parametrize("history", [0, 1])
def test_nondefault_axes_match_scalar_and_vector_rows(history: int) -> None:
    config = ObservationConfig(
        max_facts=17,
        max_cards_per_player=33,
        max_permanents_per_player=7,
        max_actions=37,
        max_focus_objects=5,
        max_events=3,
        policy_history_version=history,
    )
    space = ObservationSpace(config)
    scalar = managym.Env(seed=213, skip_trivial=True)
    vector = managym.VectorEnv(num_envs=1, seed=213, skip_trivial=True)
    vector.set_observation_layout(step_buffers(1), layout=space.layout)
    obs, _ = scalar.reset(_players())
    vector.reset_all_into_buffers(_players())
    for _ in range(12):
        encoded = _encode_everywhere(space, obs, vector)
        for key, shape in space.shapes.items():
            assert encoded[key].shape == shape, key
        assert encoded["actions_valid"].sum() == min(
            len(obs.action_space.actions), config.max_actions
        )
        obs, *_ = scalar.step(0)
        vector.step_into_buffers([0])


@pytest.mark.parametrize("capacity", [64, 255, 256, 777])
def test_historical_256_offer_capture_respects_configured_capacity(
    capacity: int,
) -> None:
    # These offers were an X-cost rules defect. Preserve their transport bytes;
    # do not require the repaired game to offer them or execute one of them.
    capture = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/tla-pilot-action-observation.json"
        ).read_bytes()
    )
    obs = managym.Observation.from_json(json.dumps(capture["observation"]))
    assert len(obs.action_space.actions) == 256
    space = ObservationSpace(
        ObservationConfig(
            max_cards_per_player=128,
            max_actions=capacity,
        )
    )
    if capacity < 256:
        message = f"observation capacity exceeded: actions 256 > {capacity}"
        with pytest.raises(managym.ObservationCapacity, match=message):
            observation_arrays(space, obs)
    else:
        reference = _encode_everywhere(space, obs)
        assert reference["actions_valid"].sum() == 256


@pytest.mark.parametrize("capacity,permanents", [(64, 40), (777, 40), (777, 1)])
def test_affordable_x_modes_reach_scalar_and_vector_rows(
    capacity: int, permanents: int
) -> None:
    game = capacity_position("wan")
    players = game.setup()
    scalar = managym.Env(seed=game.seed, skip_trivial=True)
    vector = managym.VectorEnv(
        num_envs=1,
        seed=game.seed,
        skip_trivial=True,
        player_policies=("agent", "agent"),
    )
    scalar.reset(players)
    vector.reset_all(players)
    for action in game.actions[:-1]:
        scalar.step(action)
        vector.step([action])
    frame = scalar.semantic_decision_frame_json()
    assert vector.semantic_decision_frames_json()[0] == frame
    last = game.actions[-1]
    assert isinstance(last, int)
    obs, *_ = scalar.step(last)
    assert not obs.game_over
    assert len(obs.action_space.actions) < 256
    assert [action.mode_index for action in obs.action_space.actions] == [0, 1, 2, 3]
    space = ObservationSpace(
        ObservationConfig(
            max_cards_per_player=128,
            max_actions=capacity,
            max_permanents_per_player=permanents,
        )
    )
    buffers = step_buffers(1)
    vector.set_observation_layout(buffers, layout=space.layout)
    if permanents == 1:
        assert max(len(obs.agent_permanents), len(obs.opponent_permanents)) > 1
        message = "observation capacity exceeded: .*permanents .* > 1"
        with pytest.raises(managym.ObservationCapacity, match=message):
            observation_arrays(space, obs)
        with pytest.raises(Exception, match="permanents"):
            vector.step_into_buffers([last])
    else:
        vector.step_into_buffers([last])
        encoded = _encode_everywhere(space, obs, vector)
        assert encoded["actions_valid"].sum() == 4
        # Mode selectors are one-based; zero means the action names no mode.
        np.testing.assert_array_equal(
            encoded["actions"][:4, ACTION_MODE_INDEX], [1, 2, 3, 4]
        )

        # The historical X=255 submission is no longer legal. Both native
        # stepping paths reject it without consuming the current decision or
        # publishing rows or outcomes; a subsequent legal step still agrees.
        frame = scalar.semantic_decision_frame_json()
        digest = scalar.state_digest()
        untouched = {key: value.copy() for key, value in buffers.items()}
        rows = vector.observation_rows()
        for submit in (
            lambda: scalar.step(255),
            lambda: vector.step([255]),
            lambda: vector.step_into_buffers([255]),
        ):
            with pytest.raises(Exception, match="out of bounds"):
                submit()
            assert scalar.state_digest() == digest
            assert scalar.semantic_decision_frame_json() == frame
            assert vector.semantic_decision_frames_json()[0] == frame
            assert vector.observation_rows() == rows
            for key, expected in untouched.items():
                np.testing.assert_array_equal(buffers[key], expected, err_msg=key)
        obs, *_ = scalar.step(0)
        vector.step_into_buffers([0])
        _encode_everywhere(space, obs, vector)


@pytest.mark.parametrize("history", [0, 1])
def test_event_capacity_keeps_legacy_window_and_public_history_bound(
    history: int,
) -> None:
    config = ObservationConfig(max_events=33, policy_history_version=history)
    if history:
        with pytest.raises(ValueError, match="1..32"):
            ObservationSpace(config)
        return
    space = ObservationSpace(config)
    scalar = managym.Env(seed=213)
    obs, _ = scalar.reset(_players())
    vector = managym.VectorEnv(num_envs=1, seed=213)
    vector.set_observation_layout(step_buffers(1), layout=space.layout)
    vector.reset_all_into_buffers(_players())
    encoded = _encode_everywhere(space, obs, vector)
    assert encoded["events"].shape == (33, space.shapes["events"][-1])


@pytest.mark.parametrize("capacity", [0, -1])
def test_focus_capacity_must_be_positive_at_configuration(capacity: int) -> None:
    with pytest.raises(ValidationError, match="max_focus_objects"):
        ObservationConfig(max_focus_objects=capacity)


@pytest.mark.parametrize("capacity", [2, 3, 5])
def test_focus_capacity_preserves_complete_action(capacity: int) -> None:
    scalar = managym.Env(seed=213)
    obs, _ = scalar.reset(_players())
    # Controlled transport fixture: three distinct public object references.
    space = obs.action_space
    actions = space.actions
    action = actions[0]
    action.focus = [obs.agent.id, obs.opponent.id, obs.agent_cards[0].id]
    actions[0] = action
    space.actions = actions
    obs.action_space = space
    space = ObservationSpace(ObservationConfig(max_focus_objects=capacity))
    if capacity < 3:
        message = f"observation capacity exceeded: action focus 3 > {capacity}"
        with pytest.raises(managym.ObservationCapacity, match=message):
            observation_arrays(space, obs)
    else:
        reference = _encode_everywhere(space, obs)
        # Players are objects 0 and 1; the agent's first card row is object 2.
        np.testing.assert_array_equal(reference["action_focus"][0, :3], [0, 1, 2])
        assert (reference["action_focus"][0, 3:] == -1).all()


def test_card_capacity_overflow_rejected_by_all_paths() -> None:
    space = ObservationSpace(ObservationConfig(max_cards_per_player=1))
    scalar = managym.Env(seed=213)
    obs, _ = scalar.reset(_players())
    vector = managym.VectorEnv(num_envs=1, seed=213)
    vector.set_observation_layout(step_buffers(1), layout=space.layout)
    message = "observation capacity exceeded: agent_cards.* > 1"
    with pytest.raises(managym.ObservationCapacity, match=message):
        observation_arrays(space, obs)
    with pytest.raises(Exception, match="cards.* > 1"):
        vector.reset_all_into_buffers(_players())


@pytest.mark.parametrize("capacity", [2, 64])
def test_explicit_legal_prefix_keeps_first_actions_in_scalar_and_vector_rows(
    capacity: int,
) -> None:
    space = ObservationSpace(
        ObservationConfig(max_actions=capacity, action_overflow="first-legal")
    )
    scalar = managym.Env(seed=213, skip_trivial=True)
    vector = managym.VectorEnv(num_envs=1, seed=213, skip_trivial=True)
    vector.set_observation_layout(
        step_buffers(1), layout=space.layout, action_overflow="first-legal"
    )
    obs, _ = scalar.reset(_players())
    vector.reset_all_into_buffers(_players())
    for _ in range(12):
        count = len(obs.action_space.actions)
        reference = _encode_everywhere(space, obs, vector)
        assert reference["actions_valid"].sum() == min(count, capacity)
        assert (
            vector.semantic_decision_frames_json()[0]
            == scalar.semantic_decision_frame_json()
        )
        obs, *_ = scalar.step(0)
        vector.step_into_buffers([0])


def test_first_64_legal_actions_preserve_membership_mapping_and_masks() -> None:
    scalar = managym.Env(seed=213, skip_trivial=True)
    players = [managym.PlayerConfig(name, {"Mountain": 100}) for name in ("a", "b")]
    scalar.reset(players)
    scalar.step(0)
    scalar.step(0)
    scalar.scenario_clear_hand(0)
    for _ in range(70):
        scalar.scenario_force_card_in_hand(0, "Mountain")
    obs = scalar.scenario_refresh()
    assert len(obs.action_space.actions) == 71  # 70 individually legal lands + pass.
    before = scalar.state_digest()
    frame = json.loads(scalar.semantic_decision_frame_json())
    full = observation_arrays(
        ObservationSpace(ObservationConfig(max_cards_per_player=128, max_actions=128)),
        obs,
    )
    space = ObservationSpace(
        ObservationConfig(
            max_cards_per_player=128, max_actions=64, action_overflow="first-legal"
        )
    )
    reference = _encode_everywhere(space, obs)
    for key, expected in reference.items():
        np.testing.assert_array_equal(
            expected,
            full[key][:64]
            if key in ("actions", "action_focus", "actions_valid", "name_candidates")
            else full[key],
            err_msg=key,
        )
    assert reference["actions_valid"].sum() == 64
    assert scalar.state_digest() == before
    assert len(json.loads(scalar.semantic_decision_frame_json())["offers"]) == 71
    # The retained final row is still the same native offer, not a page index.
    selected = obs.action_space.actions[63]
    command = {
        "command_id": "prefix-last",
        "expected_revision": frame["revision"],
        "offer_id": frame["offers"][63]["id"],
        "answers": [],
    }
    _, after, *_ = scalar.step_semantic_command(json.dumps(command))
    assert any(
        permanent.object_id in selected.focus
        for permanent in after.agent_permanents + after.opponent_permanents
    )


def test_cutoff_diagnostics_keep_exact_counts_and_bounded_replay_samples(
    tmp_path: Path,
) -> None:
    with capture_scope(CaptureScope(tmp_path)):
        capture = ScalarCapture(213)
        game = GameInputs(seed=213, players=[], actions=[0, 1])
        for count in (4, 64, *([71] * 9)):
            capture.record_policy_actions(count, "Priority", 64, game)
        capture.flush_policy_actions()
    (path,) = tmp_path.glob("action-cutoff-*.json")
    report = json.loads(path.read_bytes())
    assert report["counts_by_kind"]["Priority"] == {
        "decisions": 11,
        "overflows": 9,
        "dropped_actions": 63,
    }
    assert len(report["samples"]) == 4
    assert report["samples"][0] == {
        "original_count": 71,
        "retained_count": 64,
        "dropped_count": 7,
        "decision_kind": "Priority",
        "game": game.model_dump(mode="json"),
    }


def test_captured_vector_records_native_support_before_policy_step(
    tmp_path: Path,
) -> None:
    space = ObservationSpace(
        ObservationConfig(max_actions=2, action_overflow="first-legal")
    )
    with capture_scope(CaptureScope(tmp_path)):
        vector = RecordedVectorEnv(num_envs=1, seed=213)
        vector.set_observation_layout(
            step_buffers(1), "first-legal", layout=space.layout
        )
        vector.reset_all_into_buffers(
            [managym.PlayerConfig(name, {"Mountain": 30}) for name in ("a", "b")]
        )
        vector.step_into_buffers([0])
        vector.step_into_buffers([0])
        count, kind = vector.native.current_action_summaries()[0]
        assert count > 2
        vector.step_into_buffers([0])
        vector.close()
    (path,) = tmp_path.glob("action-cutoff-*.json")
    report = json.loads(path.read_bytes())
    sample = report["samples"][0]
    assert (
        sample["original_count"],
        sample["retained_count"],
        sample["decision_kind"],
    ) == (count, 2, kind)
    assert sample["game"]["seed"] == 213 and sample["game"]["actions"] == [0, 0]
    assert sample["game"]["frame"] and sample["game"]["players"]


def test_default_and_saved_observation_configs_keep_overflow_errors() -> None:
    saved = ObservationConfig().model_dump()
    assert "action_overflow" not in saved
    for config in (ObservationConfig(), ObservationConfig.from_checkpoint(saved)):
        assert config.action_overflow == "error"
        assert "action_overflow" not in config.model_dump()
