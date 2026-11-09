"""Supplied tensor axes govern native encoding, including retained pilot failures."""

import json
from pathlib import Path

import numpy as np
import pytest

from manabot.env.failures import (
    CapturedVector,
    CaptureScope,
    GameInputs,
    ScalarCapture,
    capture_scope,
    read_failure,
)
from manabot.env.observation import ObservationEncoder
from manabot.infra.config import ObservationConfig
from manabot.training.rollout.producers import ShardedVectorEnv
import managym
from tests.retained_decision import assert_retained_frame


def _vector_buffers(
    encoder: ObservationEncoder, num_envs: int = 1
) -> dict[str, np.ndarray]:
    buffers = encoder.allocate(num_envs)
    buffers.update(
        rewards=np.zeros(num_envs, dtype=np.float64),
        terminated=np.zeros(num_envs, dtype=np.uint8),
        truncated=np.zeros(num_envs, dtype=np.uint8),
    )
    return buffers


def _players() -> list[managym.PlayerConfig]:
    return [
        managym.PlayerConfig(name, {"Mountain": 10, "Gray Ogre": 20})
        for name in ("a", "b")
    ]


def _assert_encoding_parity(
    encoder: ObservationEncoder,
    scalar: managym.Env,
    obs: managym.Observation,
    vector_buffers: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Compare every configured input key, including optional versioned blocks."""
    reference = encoder.encode(obs)
    into = encoder.allocate()
    scalar.encode_observation_into(obs, into, encoder.config.action_overflow)
    for key, expected in reference.items():
        np.testing.assert_array_equal(into[key], expected, err_msg=key)
        if vector_buffers is not None:
            np.testing.assert_array_equal(vector_buffers[key][0], expected, err_msg=key)
    return reference


@pytest.mark.parametrize("version", [1, 6, 9])
def test_native_encoded_arrays_are_writable_and_independent(version: int) -> None:
    encoder = ObservationEncoder(ObservationConfig(policy_input_version=version))
    scalar = managym.Env(seed=213, skip_trivial=True)
    obs, _ = scalar.reset(_players())
    expected = encoder.encode(obs)
    first = scalar.encode_observation(obs, version, encoder.config.max_facts)
    second = scalar.encode_observation(obs, version, encoder.config.max_facts)
    assert first.keys() == second.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_array_equal(first[key], value, err_msg=key)
        assert first[key].dtype == value.dtype
        assert first[key].flags.writeable and first[key].flags.c_contiguous
        first[key].fill(19)
        np.testing.assert_array_equal(second[key], value, err_msg=key)


@pytest.mark.parametrize("capacity", [40, 41, 64])
def test_retained_airship_growth_preserves_every_permanent(capacity: int) -> None:
    evidence = read_failure(
        Path(__file__).parents[1] / "fixtures/tla-pilot-permanent-capacity.jsonl"
    )
    (game,) = evidence.games
    assert game.stream == 43 and game.seed == 32624 and len(game.actions) == 371
    players = [player.native() for player in game.players]
    scalar = managym.Env(seed=game.seed, skip_trivial=evidence.skip_trivial)
    vector = managym.VectorEnv(
        num_envs=1,
        seed=game.seed,
        seed_stride=evidence.seed_stride,
        skip_trivial=evidence.skip_trivial,
        opponent_policy=evidence.opponent_policy,
    )
    encoder = ObservationEncoder(
        ObservationConfig(
            policy_input_version=7,
            max_cards_per_player=128,
            max_permanents_per_player=capacity,
            max_actions=64,
            action_overflow="first-legal",
        )
    )
    buffers = _vector_buffers(encoder)
    vector.set_buffers(buffers, "first-legal")
    scalar.reset(players)
    vector.reset_all_into_buffers(players)
    for action in game.actions[:-1]:
        assert isinstance(action, int)
        scalar.step(action)
        vector.step_into_buffers([action])
    frame = scalar.semantic_decision_frame_json()
    assert_retained_frame(frame, game.frame)
    assert vector.semantic_decision_frames_json()[0] == frame
    last = game.actions[-1]
    assert isinstance(last, int)
    obs, *_ = scalar.step(last)
    assert len(obs.agent_permanents) == 41
    if capacity == 40:
        for encode in (
            lambda: encoder.encode(obs),
            lambda: scalar.encode_observation_into(
                obs, encoder.allocate(), "first-legal"
            ),
            lambda: vector.step_into_buffers([last]),
        ):
            with pytest.raises(Exception, match="permanents 41 > 40"):
                encode()
    else:
        vector.step_into_buffers([last])
        reference = _assert_encoding_parity(encoder, scalar, obs, buffers)
        assert reference["agent_permanents_valid"].sum() == 41
        assert reference["opponent_permanents_valid"].sum() == len(
            obs.opponent_permanents
        )


@pytest.mark.parametrize("workers", [1, 2])
def test_retained_airship_overflow_isolates_only_its_stream(workers: int) -> None:
    evidence = read_failure(
        Path(__file__).parents[1] / "fixtures/tla-pilot-permanent-capacity.jsonl"
    )
    (game,) = evidence.games
    encoder = ObservationEncoder(
        ObservationConfig(
            policy_input_version=7,
            max_cards_per_player=128,
            max_permanents_per_player=40,
            max_actions=64,
            action_overflow="first-legal",
        )
    )
    vector = (
        CapturedVector(num_envs=2, seed=game.seed, skip_trivial=True)
        if workers == 1
        else ShardedVectorEnv(num_envs=2, seed=game.seed, workers=workers)
    )
    try:
        buffers = _vector_buffers(encoder, num_envs=2)
        vector.set_buffers(buffers, "first-legal")
        vector.reset_each_into_buffers([[p.native() for p in game.players], _players()])
        for action in game.actions[:-1]:
            vector.step_into_buffers([action, 0], [True, False])
        if isinstance(vector, CapturedVector):
            assert_retained_frame(vector.semantic_decision_frames_json()[0], game.frame)
        control = managym.Env(seed=game.seed + 1, skip_trivial=True)
        control.reset(_players())
        obs, *_ = control.step(0)
        vector.step_isolated_into_buffers([game.actions[-1], 0], [True, True], True)
        failed, healthy = vector.get_last_info()
        assert failed["game_error_kind"] == "resource_limit"
        assert (
            failed["game_error"]
            == "observation capacity exceeded: agent_permanents 41 > 40"
        )
        assert_retained_frame(failed["game_error_frame"], game.frame)
        assert "game_error" not in healthy
        assert not buffers["terminated"].any() and not buffers["truncated"].any()
        assert not buffers["rewards"].any()
        reference = encoder.encode(obs)
        for key, value in reference.items():
            np.testing.assert_array_equal(buffers[key][1], value, err_msg=key)
        vector.replace_failed_into_buffers([True, False])
        assert vector.current_game_seeds() == [game.seed + 2, game.seed + 1]
        if isinstance(vector, CapturedVector):
            assert (
                vector.semantic_decision_frames_json()[1]
                == control.semantic_decision_frame_json()
            )
        for key, value in reference.items():
            np.testing.assert_array_equal(buffers[key][1], value, err_msg=key)
    finally:
        vector.close()


@pytest.mark.parametrize("version", range(1, 9))
@pytest.mark.parametrize("history", [0, 1])
def test_nondefault_axes_match_scalar_vector_and_python(
    version: int, history: int
) -> None:
    config = ObservationConfig(
        policy_input_version=version,
        max_facts=17 if version >= 6 else None,
        max_cards_per_player=33,
        max_permanents_per_player=7,
        max_actions=37,
        max_focus_objects=5,
        max_events=3,
        policy_history_version=history,
    )
    encoder = ObservationEncoder(config)
    scalar = managym.Env(seed=213, skip_trivial=True)
    vector = managym.VectorEnv(num_envs=1, seed=213, skip_trivial=True)
    buffers = _vector_buffers(encoder)
    vector.set_buffers(buffers)
    obs, _ = scalar.reset(_players())
    vector.reset_all_into_buffers(_players())
    for _ in range(12):
        _assert_encoding_parity(encoder, scalar, obs, buffers)
        obs, *_ = scalar.step(0)
        vector.step_into_buffers([0])


@pytest.mark.parametrize("capacity", [64, 255, 256, 777])
def test_historical_256_offer_capture_respects_supplied_capacity(capacity: int) -> None:
    # These offers were an X-cost rules defect. Preserve their transport bytes;
    # do not require the repaired game to offer them or execute one of them.
    capture = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/tla-pilot-action-observation.json"
        ).read_bytes()
    )
    obs = managym.Observation.from_json(json.dumps(capture["observation"]))
    assert len(obs.action_space.actions) == 256
    scalar = managym.Env(seed=213)
    encoder = ObservationEncoder(
        ObservationConfig(
            policy_input_version=7,
            max_cards_per_player=128,
            max_actions=capacity,
        )
    )
    into = encoder.allocate()
    if capacity < 256:
        with pytest.raises(ValueError, match=f"actions 256 > {capacity}"):
            encoder.encode(obs)
        with pytest.raises(ValueError, match=f"actions 256 > {capacity}"):
            scalar.encode_observation_into(obs, into)
    else:
        reference = _assert_encoding_parity(encoder, scalar, obs)
        assert reference["actions_valid"].sum() == 256


@pytest.mark.parametrize("capacity,permanents", [(64, 40), (777, 40), (777, 1)])
def test_repaired_retained_prefix_has_scalar_vector_python_parity(
    capacity: int, permanents: int
) -> None:
    evidence = read_failure(
        Path(__file__).parents[1] / "fixtures/tla-pilot-action-capacity.jsonl"
    )
    (game,) = evidence.games
    assert game.stream == 103 and len(game.actions) == 76
    players = [player.native() for player in game.players]
    scalar = managym.Env(seed=game.seed, skip_trivial=evidence.skip_trivial)
    vector = managym.VectorEnv(
        num_envs=1,
        seed=game.seed,
        seed_stride=evidence.seed_stride,
        skip_trivial=evidence.skip_trivial,
        opponent_policy=evidence.opponent_policy,
    )
    scalar.reset(players)
    vector.reset_all(players)
    for action in game.actions[:-1]:
        assert isinstance(action, int)
        scalar.step(action)
        vector.step([action])
    frame = scalar.semantic_decision_frame_json()
    assert vector.semantic_decision_frames_json()[0] == frame
    assert_retained_frame(frame, game.frame)
    last = game.actions[-1]
    assert isinstance(last, int)
    obs, *_ = scalar.step(last)
    assert not obs.game_over
    assert len(obs.action_space.actions) < 256
    assert all(action.mode_index is None for action in obs.action_space.actions)
    encoder = ObservationEncoder(
        ObservationConfig(
            policy_input_version=7,
            max_cards_per_player=128,
            max_actions=capacity,
            max_permanents_per_player=permanents,
        )
    )
    into = encoder.allocate()
    buffers = _vector_buffers(encoder)
    vector.set_buffers(buffers)
    if permanents == 1:
        assert max(len(obs.agent_permanents), len(obs.opponent_permanents)) > 1
        for encode in (
            lambda: encoder.encode(obs),
            lambda: scalar.encode_observation_into(obs, into),
            lambda: vector.step_into_buffers([last]),
        ):
            with pytest.raises(Exception, match="permanents"):
                encode()
    else:
        vector.step_into_buffers([last])
        _assert_encoding_parity(encoder, scalar, obs, buffers)

        # The historical X=255 submission is no longer legal. Both native
        # stepping paths reject it without consuming the current decision or
        # writing output buffers; a subsequent legal step still agrees.
        frame = scalar.semantic_decision_frame_json()
        digest = scalar.state_digest()
        untouched = {key: value.copy() for key, value in buffers.items()}
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
            for key, expected in untouched.items():
                np.testing.assert_array_equal(buffers[key], expected, err_msg=key)
        obs, *_ = scalar.step(0)
        vector.step_into_buffers([0])
        _assert_encoding_parity(encoder, scalar, obs, buffers)


@pytest.mark.parametrize(
    "key",
    [
        "opponent_cards",
        "agent_cards_valid",
        "semantic_cards",
        "known_hand",
        "decklists",
        "opponent_permanents",
        "agent_permanents_valid",
        "actions_valid",
        "action_focus",
        "events_valid",
        "relations",
        "facts",
        "effect_members",
    ],
)
def test_companion_shapes_rejected_before_scalar_writes(key: str) -> None:
    encoder = ObservationEncoder(ObservationConfig(policy_input_version=7))
    scalar = managym.Env(seed=213, skip_trivial=True)
    obs, _ = scalar.reset(_players())
    buffers = encoder.allocate()
    for value in buffers.values():
        value.fill(-7)
    shape = list(buffers[key].shape)
    # The first dimension is a companion axis (or a version-fixed axis).
    shape[1 if key == "facts" else 0] += 1
    buffers[key] = np.full(shape, -7, dtype=buffers[key].dtype)
    with pytest.raises(ValueError, match="shape"):
        scalar.encode_observation_into(obs, buffers)
    assert all(np.all(value == -7) for value in buffers.values())
    batched = {key: value[None, ...] for key, value in buffers.items()}
    batched.update(
        rewards=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )
    vector = managym.VectorEnv(num_envs=1, seed=213)
    with pytest.raises(ValueError, match="shape"):
        vector.set_buffers(batched)


@pytest.mark.parametrize(
    "key",
    ["agent_cards", "agent_permanents", "actions", "action_focus", "events", "facts"],
)
@pytest.mark.parametrize("batched", [False, True])
def test_zero_capacity_rejected(key: str, batched: bool) -> None:
    encoder = ObservationEncoder(ObservationConfig(policy_input_version=7))
    buffers = encoder.allocate()
    shape = list(buffers[key].shape)
    shape[1 if key == "action_focus" else 0] = 0
    buffers[key] = np.zeros(shape, dtype=buffers[key].dtype)
    if batched:
        vector = managym.VectorEnv(num_envs=1)
        with pytest.raises(ValueError, match="shape"):
            vector.set_buffers(
                {name: value[None, ...] for name, value in buffers.items()}
            )
    else:
        scalar = managym.Env(seed=213)
        obs, _ = scalar.reset(_players())
        with pytest.raises(ValueError, match="shape"):
            scalar.encode_observation_into(obs, buffers)


@pytest.mark.parametrize("history", [0, 1])
def test_event_capacity_keeps_legacy_window_and_public_history_bound(
    history: int,
) -> None:
    encoder = ObservationEncoder(
        ObservationConfig(max_events=33, policy_history_version=history)
    )
    scalar = managym.Env(seed=213)
    obs, _ = scalar.reset(_players())
    vector = managym.VectorEnv(num_envs=1, seed=213)
    buffers = _vector_buffers(encoder)
    if history:
        with pytest.raises(ValueError, match="1..32"):
            scalar.encode_observation_into(obs, encoder.allocate())
        with pytest.raises(ValueError, match="1..32"):
            vector.set_buffers(buffers)
    else:
        vector.set_buffers(buffers)
        vector.reset_all_into_buffers(_players())
        _assert_encoding_parity(encoder, scalar, obs, buffers)


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
    encoder = ObservationEncoder(ObservationConfig(max_focus_objects=capacity))
    if capacity < 3:
        with pytest.raises(ValueError, match="action focus"):
            encoder.encode(obs)
        with pytest.raises(ValueError, match="action focus"):
            scalar.encode_observation_into(obs, encoder.allocate())
    else:
        reference = _assert_encoding_parity(encoder, scalar, obs)
        assert (reference["action_focus"][0, :3] >= 0).all()


def test_card_capacity_overflow_rejected_by_all_paths() -> None:
    encoder = ObservationEncoder(ObservationConfig(max_cards_per_player=1))
    scalar = managym.Env(seed=213)
    obs, _ = scalar.reset(_players())
    vector = managym.VectorEnv(num_envs=1, seed=213)
    vector.set_buffers(_vector_buffers(encoder))
    for encode in (
        lambda: encoder.encode(obs),
        lambda: scalar.encode_observation_into(obs, encoder.allocate()),
        lambda: vector.reset_all_into_buffers(_players()),
    ):
        with pytest.raises(Exception, match="cards.* > 1"):
            encode()


@pytest.mark.parametrize("capacity", [2, 64])
def test_explicit_legal_prefix_matches_native_scalar_vector_and_python(
    capacity: int,
) -> None:
    encoder = ObservationEncoder(
        ObservationConfig(max_actions=capacity, action_overflow="first-legal")
    )
    scalar = managym.Env(seed=213, skip_trivial=True)
    vector = managym.VectorEnv(num_envs=1, seed=213, skip_trivial=True)
    buffers = _vector_buffers(encoder)
    vector.set_buffers(buffers, "first-legal")
    obs, _ = scalar.reset(_players())
    vector.reset_all_into_buffers(_players())
    for _ in range(12):
        count = len(obs.action_space.actions)
        reference = _assert_encoding_parity(encoder, scalar, obs, buffers)
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
    scalar.scenario_clear_hand(0)
    for _ in range(70):
        scalar.scenario_force_card_in_hand(0, "Mountain")
    obs = scalar.scenario_refresh()
    assert len(obs.action_space.actions) == 71  # 70 individually legal lands + pass.
    before = scalar.state_digest()
    frame = json.loads(scalar.semantic_decision_frame_json())
    full = ObservationEncoder(
        ObservationConfig(max_cards_per_player=128, max_actions=128)
    ).encode(obs)
    encoder = ObservationEncoder(
        ObservationConfig(
            max_cards_per_player=128, max_actions=64, action_overflow="first-legal"
        )
    )
    reference = _assert_encoding_parity(encoder, scalar, obs)
    for key, expected in reference.items():
        np.testing.assert_array_equal(
            expected,
            full[key][:64]
            if key in ("actions", "action_focus", "actions_valid")
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
        card.id in selected.focus
        and int(card.zone) == int(managym.ZoneEnum.BATTLEFIELD)
        for card in after.agent_cards + after.opponent_cards
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
    encoder = ObservationEncoder(
        ObservationConfig(max_actions=2, action_overflow="first-legal")
    )
    with capture_scope(CaptureScope(tmp_path)):
        vector = CapturedVector(num_envs=1, seed=213)
        vector.set_buffers(_vector_buffers(encoder), "first-legal")
        vector.reset_all_into_buffers(
            [managym.PlayerConfig(name, {"Mountain": 30}) for name in ("a", "b")]
        )
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
    assert sample["game"]["seed"] == 213 and sample["game"]["actions"] == []
    assert sample["game"]["frame"] and sample["game"]["players"]


def test_default_and_legacy_observation_configs_keep_overflow_errors() -> None:
    for config in (ObservationConfig(), ObservationConfig.from_checkpoint({})):
        assert config.action_overflow == "error"
        assert "action_overflow" not in config.model_dump()
