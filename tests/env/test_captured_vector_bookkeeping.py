"""CapturedVector replay prefixes follow native games, not seeds or raw delegation."""

import numpy as np
import pytest
import torch

from manabot import managym
from manabot.env.failures import CapturedVector, FailureEvidence
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.env.replay import replay_failure_inputs
from manabot.env.vector_env import VectorEnv as TorchVectorEnv
from manabot.infra.config import MatchConfig, RewardConfig

_MAX_SEED = 2**64 - 1


def _players() -> list[managym.PlayerConfig]:
    return [managym.PlayerConfig(name, {"Mountain": 8}) for name in ("a", "b")]


def _captured(
    seed: int,
    player_policies: tuple[str, str] | None = None,
    *,
    opponent_policy: str | None = None,
) -> tuple[CapturedVector, dict[str, np.ndarray]]:
    vector = CapturedVector(
        num_envs=1,
        seed=seed,
        player_policies=player_policies,
        opponent_policy=opponent_policy,
    )
    buffers = ObservationSpace().encoder.allocate(1)
    buffers.update(
        rewards=np.zeros(1, dtype=np.float64),
        terminated=np.zeros(1, dtype=np.uint8),
        truncated=np.zeros(1, dtype=np.uint8),
    )
    vector.set_buffers(buffers)
    return vector, buffers


def _seeds(vector: TorchVectorEnv) -> list[int]:
    return [snapshot.seed for snapshot in vector.hand_snapshots([0, 0])]


@pytest.mark.parametrize(
    "name",
    [
        "reset_all",
        "step",
        "take_exposure",
        "enable_exposure_capture",
        "scenario_naming_vocabulary_into_buffers",
    ],
)
def test_native_mutators_are_not_delegated(name: str) -> None:
    vector, _ = _captured(42)
    try:
        with pytest.raises(AttributeError, match="into_buffers"):
            getattr(vector, name)
        assert not hasattr(vector, "no_such_method")
    finally:
        vector.close()


def test_native_reads_are_delegated() -> None:
    vector, _ = _captured(42)
    try:
        vector.reset_all_into_buffers(_players())
        assert vector.current_game_seeds() == [42]
        assert vector.current_agent_indices() == [0]
        assert len(vector.current_game_generations()) == 1
        assert vector.semantic_decision_frames_json()[0] is not None
        assert len(vector.skip_trivial_counts()) == 1
    finally:
        vector.close()


@pytest.mark.parametrize("policy", ["none", "random", "heuristic"])
def test_legacy_capture_argument_warns_and_records_only_per_seat_players(
    policy: str,
) -> None:
    with pytest.warns(DeprecationWarning, match="0.3.0"):
        vector, _ = _captured(42, opponent_policy=policy)
    try:
        vector.reset_all_into_buffers(_players())
        vector.step_into_buffers([0])
        evidence = vector._evidence_for(vector.games, "step")
        restored = FailureEvidence.model_validate_json(
            evidence.model_dump_json(by_alias=True)
        )
        assert restored.player_policies == (
            "agent",
            "agent" if policy == "none" else policy,
        )
        assert restored.opponent_policy is None
        assert "opponent_policy" not in evidence.model_dump(mode="json", by_alias=True)
        assert replay_failure_inputs(restored)[0]["matches_binding"] is True
    finally:
        vector.close()


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize(
    ("policy", "player_policies"),
    [
        (None, None),
        ("none", None),
        ("random", None),
        ("heuristic", None),
        pytest.param("passive", ("agent", "random"), id="per-seat-takes-precedence"),
    ],
)
def test_retained_policy_replay_has_no_deprecation_warning(
    policy: str | None, player_policies: tuple[str, str] | None
) -> None:
    policies = player_policies or (
        "agent",
        "agent" if policy in (None, "none") else policy,
    )
    vector, _ = _captured(42, player_policies=policies)
    try:
        vector.reset_all_into_buffers(_players())
        vector.step_into_buffers([0])
        payload = vector._evidence_for(vector.games, "step").model_dump(
            mode="json", by_alias=True
        )
        if player_policies is None:
            del payload["player_policies"]
        if policy is not None:
            payload["opponent_policy"] = policy
        evidence = FailureEvidence.model_validate(payload)
        retained = evidence.model_dump_json(by_alias=True)
        result = replay_failure_inputs(evidence)[0]
        assert result["status"] == "advanced"
        assert result["matches_binding"] is True
        assert evidence.model_dump_json(by_alias=True) == retained
    finally:
        vector.close()


@pytest.mark.parametrize("policy", ["agent", "random", "heuristic"])
def test_terminal_replacement_at_saturated_seed_starts_a_new_prefix(
    policy: str,
) -> None:
    vector, buffers = _captured(_MAX_SEED, player_policies=("agent", policy))
    try:
        vector.reset_all_into_buffers(_players())
        generation = vector.current_game_generations()[0]
        for _ in range(120):
            vector.step_into_buffers([0])
            if buffers["terminated"][0] or buffers["truncated"][0]:
                break
        else:
            pytest.fail("bounded tiny-deck fixture did not terminate")
        assert vector.current_game_seeds() == [_MAX_SEED]
        assert vector.current_game_generations()[0] != generation
        (game,) = vector.games
        assert (game.seed, game.actions, game.frame) == (_MAX_SEED, [], None)
    finally:
        vector.close()


@pytest.mark.parametrize("policy", ["agent", "random", "heuristic"])
def test_same_seed_reset_starts_a_new_prefix(policy: str) -> None:
    vector, _ = _captured(42, player_policies=("agent", policy))
    try:
        vector.reset_all_into_buffers(_players())
        vector.step_into_buffers([0])
        assert vector.games[0].actions == [0]
        generation = vector.current_game_generations()[0]
        vector.reset_seeds_into_buffers([42])
        assert vector.current_game_generations()[0] != generation
        assert (vector.games[0].seed, vector.games[0].actions) == (42, [])
    finally:
        vector.close()


@pytest.mark.parametrize("policy", ["agent", "random", "heuristic"])
def test_torch_reset_with_a_seed_restarts_and_without_continues(
    policy: str,
) -> None:
    match = Match(MatchConfig(hero_deck={"Mountain": 12}, villain_deck={"Island": 12}))
    vector = TorchVectorEnv(
        2,
        match,
        ObservationSpace(),
        Reward(RewardConfig()),
        "cpu",
        seed=42,
        player_policies=("agent", policy),
    )
    try:
        first, _ = vector.reset(seed=42)
        first = {key: value.clone() for key, value in first.items()}
        assert _seeds(vector) == [42, 43]
        vector.step(torch.zeros(2, dtype=torch.int64))
        again, _ = vector.reset(seed=42)
        assert _seeds(vector) == [42, 43]
        for key, value in first.items():
            assert torch.equal(again[key], value), key
        vector.reset()
        assert _seeds(vector) == [44, 45]
    finally:
        vector.close()


@pytest.mark.parametrize(
    "policies", [("heuristic", "agent"), ("heuristic", "heuristic")]
)
def test_captured_per_seat_configuration_and_terminal_rewards(
    policies: tuple[str, str],
) -> None:
    vector, buffers = _captured(213, player_policies=policies)
    players = [
        managym.PlayerConfig(name, {"Forest": 8, "Llanowar Elves": 4})
        for name in ("a", "b")
    ]
    scalar = managym.Env(seed=213, player_policies=policies)
    scalar.reset(players)
    try:
        vector.reset_all_into_buffers(players)
        evidence = vector._evidence_for(vector.games, "step")
        assert evidence.player_policies == policies
        assert evidence.opponent_policy is None
        assert "opponent_policy" not in evidence.model_dump(mode="json", by_alias=True)
        for _ in range(1024):
            chooser = scalar.current_agent_index()
            assert chooser is not None
            external = policies[chooser] == "agent"
            _, reward, done, truncated, _ = scalar.step(0 if external else None)
            vector.step_into_buffers([0 if external else -1])
            assert buffers["rewards"][0] == reward
            assert bool(buffers["terminated"][0]) == done
            assert bool(buffers["truncated"][0]) == truncated
            if done or truncated:
                assert vector.games[0].actions == []
                break
            assert vector.current_agent_indices() == [scalar.current_agent_index()]
        else:
            pytest.fail("bounded per-seat fixture did not terminate")
    finally:
        vector.close()
