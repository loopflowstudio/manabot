"""Installed native selector, Python adapter, and retained pre-port decisions.

Rule-group fixtures live alongside the native selector in heuristic/tests.rs.
Golden decisions come from the Python owner at 757bb6257 on the fixture's
engine world. A world change needs a new comparison with that reference,
never relabeled actions.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from manabot import managym
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.env.vector_env import VectorEnv
from manabot.eval.admission import register_anchor
from manabot.infra.config import MatchConfig, RewardConfig
from manabot.infra.identity import file_sha256
from manabot.sim import anchors
from manabot.sim.heuristic import HeuristicPlayer, heuristic_source_paths
from manabot.sim.players import build_code_player
from tests.eval.arena_fixture import arena_fixture

PARITY = Path(__file__).parents[1] / "fixtures/native_heuristic_parity.json"


def _players(decks: list[str]) -> list[managym.PlayerConfig]:
    if decks[0].startswith("elemental_"):
        return [
            managym.authored_deck_setup("elemental-world-v2", name) for name in decks
        ]
    return [
        managym.PlayerConfig(
            "red", {"Mountain": 20, "Gray Ogre": 12, "Lightning Bolt": 8}
        ),
        managym.PlayerConfig("green", {"Forest": 20, "Llanowar Elves": 20}),
    ]


def test_native_heuristic_matches_retained_python_decisions() -> None:
    fixture = json.loads(PARITY.read_text())
    assert fixture["world"] == managym.world_identity()
    for case in fixture["cases"]:
        env = managym.Env(seed=case["seed"])
        raw, _ = env.reset(_players(case["decks"]))
        players = [HeuristicPlayer(max_actions=case["max_actions"]) for _ in range(2)]
        for decision, expected in enumerate(case["actions"]):
            view = cast(Env, SimpleNamespace(last_raw_obs=raw))
            chosen = players[raw.agent.player_index].act(view, {})
            assert chosen == expected, (case["seed"], case["decks"], decision)
            raw, _, done, truncated, _ = env.step(chosen)
            if done or truncated:
                assert decision == len(case["actions"]) - 1
                break
        assert (done, truncated) == (case["terminated"], case["truncated"])


@pytest.mark.parametrize("seed", [213, 2**64 - 1])
def test_native_batch_matches_scalar_choices_rewards_and_reset(seed: int) -> None:
    configs = _players(["red", "green"])
    batch = managym.VectorEnv(
        num_envs=1, seed=seed, player_policies=("agent", "heuristic")
    )
    scalar = managym.Env(seed=seed)
    selector = managym.Heuristic()
    raw, _ = scalar.reset(configs)
    ((batched, _),) = batch.reset_all(configs)
    episodes = 0
    for _ in range(96):
        while raw.agent.player_index == 1:
            raw, _, done, _, _ = scalar.step(selector.select_action(raw))
            assert not done  # This fixed fixture reaches a learner choice at reset.
        assert batched.toJSON() == raw.toJSON()
        raw, expected_reward, done, truncated, _ = scalar.step(0)
        while not (done or truncated) and raw.agent.player_index == 1:
            raw, opponent_reward, done, truncated, _ = scalar.step(
                selector.select_action(raw)
            )
            if done or truncated:
                expected_reward = -opponent_reward
        ((batched, reward, batch_done, batch_truncated, _),) = batch.step([0])
        assert (reward, batch_done, batch_truncated) == (
            expected_reward,
            done,
            truncated,
        )
        if done or truncated:
            episodes += 1
            scalar = managym.Env(seed=min(seed + episodes, 2**64 - 1))
            raw, _ = scalar.reset(configs)
            selector.reset()
    assert episodes > 0


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_adapter_rejects_invalid_capacity(capacity: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        HeuristicPlayer(max_actions=cast(int, capacity))


def test_heuristic_registration_versions_native_sources() -> None:
    candidate = arena_fixture().anchors[0]
    scripted = register_anchor(candidate, "scripted_greedy", player_id="scripted")
    native = register_anchor(candidate, "heuristic", player_id="native")
    assert scripted.source_sha256 == file_sha256(Path(anchors.__file__))
    assert native.player_spec == {"kind": "heuristic", "version": "native-v2"}
    assert native.source_sha256 != scripted.source_sha256
    assert any(path.suffix == ".rs" for path in heuristic_source_paths())
    assert all(path.exists() for path in heuristic_source_paths())
    assert isinstance(build_code_player(native, seed=42), HeuristicPlayer)


@pytest.mark.parametrize("capacity", [None, True, 0, -1, "64"])
def test_registered_heuristic_rejects_invalid_capacity(capacity: object) -> None:
    registration = register_anchor(
        arena_fixture().anchors[0], "heuristic", player_id="bounded"
    ).model_copy(
        update={
            "player_spec": {
                "kind": "heuristic",
                "max_actions": capacity,
                "action_overflow": "first-legal",
            }
        }
    )
    with pytest.raises(ValueError, match="explicit first-legal support"):
        build_code_player(registration, seed=213)


def test_python_vector_wrapper_accepts_native_heuristic() -> None:
    deck = {"Forest": 20, "Llanowar Elves": 20}
    env = VectorEnv(
        2,
        Match(MatchConfig(hero_deck=deck, villain_deck=deck)),
        ObservationSpace(),
        Reward(RewardConfig()),
        device="cpu",
        player_policies=("agent", "heuristic"),
    )
    try:
        obs, _ = env.reset()
        assert obs["actions_valid"].shape[0] == 2
        env.step(torch.zeros(2, dtype=torch.long))
    finally:
        env.close()


def test_closed_decklists_stay_closed_for_native_selector() -> None:
    configs = _players(["red", "green"])
    configs[0].disclosure = "closed"
    configs[1].disclosure = "closed"
    env = managym.Env(seed=42)
    raw, _ = env.reset(configs)
    assert raw.opponent.main_counts is None
    assert not any(card.zone == managym.ZoneEnum.HAND for card in raw.opponent_cards)
    before = env.state_digest()
    selector = managym.Heuristic()
    selected = selector.select_action(raw)
    assert env.state_digest() == before
    assert 0 <= selected < len(raw.action_space.actions)
    env.step(selected)
