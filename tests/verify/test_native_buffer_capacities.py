"""Supplied tensor axes govern native encoding, including retained pilot failures."""

from pathlib import Path

import numpy as np
import pytest

from manabot.env.failures import read_failure
from manabot.env.observation import ObservationEncoder
from manabot.infra.config import ObservationConfig
import managym


def _vector_buffers(encoder: ObservationEncoder) -> dict[str, np.ndarray]:
    buffers = encoder.allocate(1)
    buffers.update(
        rewards=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )
    return buffers


def _players() -> list[managym.PlayerConfig]:
    return [
        managym.PlayerConfig(name, {"Mountain": 10, "Gray Ogre": 20})
        for name in ("a", "b")
    ]


@pytest.mark.parametrize("version", range(1, 8))
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
        reference = encoder.encode(obs)
        into = encoder.allocate()
        scalar.encode_observation_into(obs, into)
        for key, expected in reference.items():
            np.testing.assert_array_equal(into[key], expected, err_msg=key)
            np.testing.assert_array_equal(buffers[key][0], expected, err_msg=key)
        obs, *_ = scalar.step(0)
        vector.step_into_buffers([0])


@pytest.mark.parametrize(
    "capacity,permanents", [(64, 40), (255, 40), (256, 40), (777, 40), (777, 1)]
)
def test_retained_256_actions_respect_supplied_capacity(
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
    assert scalar.semantic_decision_frame_json() == game.frame
    assert vector.semantic_decision_frames_json()[0] == game.frame
    last = game.actions[-1]
    assert isinstance(last, int)
    obs, *_ = scalar.step(last)
    assert len(obs.action_space.actions) == 256
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
    if capacity < 256 or permanents == 1:
        error = f"actions 256 > {capacity}" if capacity < 256 else "permanents"
        if permanents == 1:
            assert max(len(obs.agent_permanents), len(obs.opponent_permanents)) > 1
        for encode in (
            lambda: encoder.encode(obs),
            lambda: scalar.encode_observation_into(obs, into),
            lambda: vector.step_into_buffers([last]),
        ):
            with pytest.raises(Exception, match=error):
                encode()
    else:
        reference = encoder.encode(obs)
        scalar.encode_observation_into(obs, into)
        vector.step_into_buffers([last])
        assert reference["actions_valid"].sum() == 256
        for key, expected in reference.items():
            np.testing.assert_array_equal(into[key], expected, err_msg=key)
            np.testing.assert_array_equal(buffers[key][0], expected, err_msg=key)


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
        into = encoder.allocate()
        scalar.encode_observation_into(obs, into)
        vector.set_buffers(buffers)
        vector.reset_all_into_buffers(_players())
        for key, expected in encoder.encode(obs).items():
            np.testing.assert_array_equal(into[key], expected, err_msg=key)
            np.testing.assert_array_equal(buffers[key][0], expected, err_msg=key)


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
        into = encoder.allocate()
        scalar.encode_observation_into(obs, into)
        reference = encoder.encode(obs)
        assert (reference["action_focus"][0, :3] >= 0).all()
        for key, expected in reference.items():
            np.testing.assert_array_equal(into[key], expected, err_msg=key)


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
