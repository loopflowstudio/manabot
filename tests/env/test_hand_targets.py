"""Privileged batch labels stay aligned with native seats and game boundaries."""

from collections import Counter
from typing import Literal

import numpy as np
import pytest
import torch

from manabot import managym
from manabot.env.failures import CapturedVector
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.env.vector_env import VectorEnv as TorchVectorEnv
from manabot.infra.config import MatchConfig


def _players(disclosure: Literal["open", "closed"] = "closed") -> list[managym.PlayerConfig]:
    return [
        managym.PlayerConfig("a", {"Mountain": 8, "Gray Ogre": 4}, disclosure=disclosure),
        managym.PlayerConfig("b", {"Island": 8, "Wind Drake": 4}, disclosure=disclosure),
    ]


def _buffers(space: ObservationSpace, size: int) -> dict[str, np.ndarray]:
    buffers = space.encoder.allocate(size)
    buffers.update(
        rewards=np.zeros(size, dtype=np.float64),
        terminated=np.zeros(size, dtype=np.uint8),
        truncated=np.zeros(size, dtype=np.uint8),
    )
    return buffers


def _hand(env: managym.Env, player: int) -> dict[str, int]:
    return dict(Counter(
        card.name for card in env.observation_for_player(player).agent_cards
        if int(card.zone) == 1
    ))


def _identity(target: managym.HandTarget) -> tuple[int, int, int, int, int]:
    return target.slot, target.player, target.game_id, target.seed, target.revision


@pytest.mark.parametrize("disclosure", ["open", "closed"])
def test_hand_targets_match_scalar_without_changing_observations(
    disclosure: Literal["open", "closed"],
) -> None:
    vector = managym.VectorEnv(2, seed=42)
    players = _players(disclosure)
    results = vector.reset_all(players)
    scalars = [managym.Env(seed=42 + slot) for slot in range(2)]
    for scalar in scalars:
        scalar.reset(players)
    space = ObservationSpace()
    before = vector.semantic_decision_frames_json()
    targets = vector.hand_targets([1, 0])
    assert vector.semantic_decision_frames_json() == before
    for slot, (target, scalar, (obs, _)) in enumerate(zip(targets, scalars, results, strict=True)):
        assert isinstance(target, managym.HandTarget)
        assert (target.slot, target.player, target.seed) == (slot, 1 - slot, 42 + slot)
        assert target.cards == _hand(scalar, target.player)
        assert sum(target.cards.values()) == 7
        assert max(target.cards.values()) > 1
        assert not any(int(card.zone) == 1 for card in obs.opponent_cards)
        assert obs.opponent.known_hand == {}
        expected = space.encode(scalar.observation_for_player(obs.agent.player_index))
        for key, actual in space.encode(obs).items():
            np.testing.assert_array_equal(actual, expected[key], err_msg=key)
        if disclosure == "closed":
            assert obs.opponent.main_counts is None
        copy = target.cards
        copy.clear()
        assert target.cards == _hand(scalar, target.player)
    old = [target.cards for target in targets]
    for _ in range(8):
        vector.step([0, 0])
        for scalar in scalars:
            scalar.step(0)
        for slot, target in enumerate(vector.hand_targets([0, 1])):
            assert target.cards == _hand(scalars[slot], slot)
            assert target.game_id == targets[slot].game_id
    assert [target.cards for target in targets] == old


def test_hand_targets_reject_invalid_requests_without_advancing() -> None:
    vector = managym.VectorEnv(2, seed=42)
    with pytest.raises(Exception, match="reset"):
        vector.hand_targets([0, 1])
    vector.reset_all(_players())
    expected = [_identity(target) for target in vector.hand_targets([0, 1])]
    for players in ([], [0], [0, 1, 0], [0, 2], [-1, 0]):
        with pytest.raises((managym.PyAgentError, OverflowError)):
            vector.hand_targets(players)
        assert [_identity(target) for target in vector.hand_targets([0, 1])] == expected


@pytest.mark.parametrize("captured", [False, True])
def test_hand_targets_mask_reset_and_failure_replacement(captured: bool) -> None:
    vector = CapturedVector(num_envs=2, seed=42) if captured else managym.VectorEnv(2, seed=42)
    buffers = _buffers(ObservationSpace(), 2)
    vector.set_buffers(buffers)
    try:
        vector.reset_each_into_buffers([_players(), list(reversed(_players()))])
        before = vector.hand_targets([1, 0])
        frozen = {key: value[1].copy() for key, value in buffers.items()}
        vector.step_into_buffers([0, -999], [True, False])
        after = vector.hand_targets([1, 0])
        assert _identity(before[1]) == _identity(after[1])
        assert before[1].cards == after[1].cards
        assert before[0].game_id == after[0].game_id
        assert after[0].revision > before[0].revision
        for key, expected in frozen.items():
            np.testing.assert_array_equal(buffers[key][1], expected)
        vector.reset_seeds_into_buffers([42, 43])
        reset = vector.hand_targets([1, 0])
        assert [t.cards for t in reset] == [t.cards for t in before]
        assert all(a.game_id != b.game_id for a, b in zip(before, reset, strict=True))
        vector.reassign_roots_into_buffers([list(reversed(_players())), None])
        reassigned = vector.hand_targets([1, 0])
        assert reassigned[0].game_id != reset[0].game_id
        assert reassigned[0].seed == reset[0].seed
        assert _identity(reassigned[1]) == _identity(reset[1])
        vector.step_isolated_into_buffers([-1, 0], [True, False])
        assert "game_error" in vector.get_last_info()[0]
        vector.replace_failed_into_buffers([True, False])
        replaced = vector.hand_targets([1, 0])
        assert replaced[0].game_id != reassigned[0].game_id
        assert replaced[0].seed == 44
        assert _identity(replaced[1]) == _identity(reassigned[1])
    finally:
        if isinstance(vector, CapturedVector):
            vector.close()


def test_hand_targets_after_done_label_the_returned_replacement() -> None:
    players = [managym.PlayerConfig(name, {"Mountain": 8}) for name in ("a", "b")]
    vector = managym.VectorEnv(1, seed=42)
    vector.reset_all(players)
    before = vector.hand_targets([1])[0]
    for _ in range(120):
        [(obs, _, terminated, truncated, _)] = vector.step([0])
        if terminated or truncated:
            break
    else:
        pytest.fail("bounded tiny-deck fixture did not terminate")
    after = vector.hand_targets([1])[0]
    assert after.game_id != before.game_id
    assert after.seed == 43
    assert after.cards == {"Mountain": 7}
    assert before.seed == 42
    assert not obs.game_over
    scalar = managym.Env(seed=after.seed)
    scalar.reset(players)
    assert after.cards == _hand(scalar, 1)


def test_torch_hand_targets_survive_native_rebuild_and_stay_off_policy_inputs() -> None:
    match = Match(MatchConfig(hero_deck={"Mountain": 12}, villain_deck={"Island": 12}))
    vector = TorchVectorEnv(2, match, ObservationSpace(), Reward(), "cpu", seed=42)
    try:
        observations, _ = vector.reset()
        before = {key: value.clone() for key, value in observations.items()}
        first = vector.hand_targets([1, 0])
        assert first[0].cards == {"Island": 7}
        for key in observations:
            assert torch.equal(observations[key], before[key])
        assert "hand_targets" not in observations
        vector.step(torch.zeros(2, dtype=torch.int64))
        assert vector.hand_targets([1, 0])[0].game_id == first[0].game_id
        for seed in (99, 42):
            vector.reset(seed=seed)
            current = vector.hand_targets([1, 0])
            assert current[0].game_id != first[0].game_id
            assert current[0].seed == seed
            assert current[0].cards == first[0].cards
    finally:
        vector.close()
