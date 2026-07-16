"""Tests for the search-supervised policy/value training substrate."""

from pathlib import Path

import numpy as np
import torch

from manabot.env import Env, Match, ObservationSpace, Reward
from manabot.infra.hypers import AgentHypers, MatchHypers, RewardHypers
from manabot.model.agent import Agent
from manabot.sim.distill import (
    OBS_KEYS,
    ROOT_VALUE_KEY,
    SCORE_KEY,
    VISIT_COUNT_KEY,
    save_bc_checkpoint,
)
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.sim.search_supervised import (
    CHOSEN_ACTION_TARGET,
    ROOT_VALUE_TARGET,
    VISIT_DISTRIBUTION_TARGET,
    outcome_targets,
    train_search_supervised,
)
from manabot.verify.util import INTERACTIVE_DECK


def _dataset(seed: int = 7) -> dict[str, np.ndarray]:
    obs_space = ObservationSpace()
    match = Match(
        MatchHypers(
            hero="teacher-a",
            villain="teacher-b",
            hero_deck=dict(INTERACTIVE_DECK),
            villain_deck=dict(INTERACTIVE_DECK),
        )
    )
    env = Env(match, obs_space, Reward(RewardHypers()), seed=seed, auto_reset=False)
    rng = np.random.default_rng(seed)
    observations: dict[str, list[np.ndarray]] = {key: [] for key in OBS_KEYS}
    scores: list[np.ndarray] = []
    actions: list[int] = []
    game_indices: list[int] = []
    seats: list[int] = []
    num_valid: list[int] = []
    winners: list[int] = []

    for game_index in range(8):
        obs, _ = env.reset(seed=seed + game_index)
        winner = game_index % 2
        for _ in range(8):
            valid = np.flatnonzero(obs["actions_valid"] > 0)
            if not len(valid):
                break
            for key in OBS_KEYS:
                observations[key].append(np.asarray(obs[key], dtype=np.float32))
            row = np.full(obs_space.encoder.max_actions, -1.0, dtype=np.float32)
            row[valid] = np.linspace(0.2, 0.8, len(valid), dtype=np.float32)
            scores.append(row)
            actions.append(int(valid[-1]))
            game_indices.append(game_index)
            seats.append(int(env.last_raw_obs.agent.player_index))
            num_valid.append(len(valid))
            winners.append(winner)
            obs, _, terminated, truncated, _ = env.step(int(rng.choice(valid)))
            if terminated or truncated:
                break

    dataset = {key: np.stack(values) for key, values in observations.items()}
    dataset.update(
        {
            SCORE_KEY: np.stack(scores),
            "action": np.asarray(actions, dtype=np.int16),
            "game_index": np.asarray(game_indices, dtype=np.int32),
            "seat": np.asarray(seats, dtype=np.int8),
            "num_valid": np.asarray(num_valid, dtype=np.int16),
            "winner": np.asarray(winners, dtype=np.int8),
        }
    )
    return dataset


def test_outcome_targets_follow_deciding_player_perspective() -> None:
    usable, targets = outcome_targets(
        np.asarray([0, 1, -1, 0]), np.asarray([0, 0, 1, 1])
    )
    assert usable.tolist() == [True, True, False, True]
    assert targets.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_policy_only_and_joint_arms_isolate_value_gradient() -> None:
    dataset = _dataset()
    seed = 11
    torch.manual_seed(seed)
    initial = Agent(ObservationSpace(), AgentHypers())
    initial_state = {
        name: value.clone() for name, value in initial.state_dict().items()
    }

    policy_only, _, initial_metrics, policy_history = train_search_supervised(
        dataset,
        value_weight=0.0,
        epochs=2,
        batch_size=16,
        val_fraction=0.25,
        seed=seed,
    )
    joint, _, _, joint_history = train_search_supervised(
        dataset,
        value_weight=1.0,
        epochs=2,
        batch_size=16,
        val_fraction=0.25,
        seed=seed,
    )

    assert len(policy_history) == len(joint_history) == 2
    assert np.isfinite(initial_metrics.policy_loss)
    assert np.isfinite(initial_metrics.policy_kl)
    assert initial_metrics.policy_kl >= 0.0
    assert np.isfinite(joint_history[-1].validation.value_brier)
    assert joint_history[-1].validation.value_rows > 0
    for name, value in policy_only.state_dict().items():
        if name.startswith("value_head"):
            assert torch.equal(value.cpu(), initial_state[name]), name
    assert any(
        not torch.equal(value.cpu(), initial_state[name])
        for name, value in joint.state_dict().items()
        if name.startswith("value_head")
    )


def test_joint_checkpoint_round_trips(tmp_path: Path) -> None:
    dataset = _dataset(seed=19)
    agent, obs_space, _, history = train_search_supervised(
        dataset, epochs=1, batch_size=32, val_fraction=0.25, seed=3
    )
    path = tmp_path / "search_supervised.pt"
    save_bc_checkpoint(
        agent,
        obs_space,
        path,
        extra={
            "search_supervised": True,
            "value_brier": history[-1].validation.value_brier,
        },
    )
    loaded, loaded_space = load_checkpoint_agent(str(path))
    assert loaded_space == obs_space
    for (name, expected), (_, actual) in zip(
        agent.state_dict().items(), loaded.state_dict().items()
    ):
        assert torch.equal(expected.cpu(), actual.cpu()), name


def test_visit_and_root_value_columns_are_supported_for_future_mcts() -> None:
    dataset = _dataset(seed=23)
    dataset[VISIT_COUNT_KEY] = np.where(dataset[SCORE_KEY] >= 0, 1.0, 0.0).astype(
        np.float32
    )
    dataset[ROOT_VALUE_KEY] = (
        dataset["winner"].astype(np.int64) == dataset["seat"].astype(np.int64)
    ).astype(np.float32)
    _, _, initial, history = train_search_supervised(
        dataset,
        policy_target_kind=VISIT_DISTRIBUTION_TARGET,
        value_target_kind=ROOT_VALUE_TARGET,
        epochs=1,
        batch_size=32,
        val_fraction=0.25,
        seed=5,
    )
    assert initial.value_rows > 0
    assert np.isfinite(history[-1].validation.policy_loss)
    assert np.isfinite(history[-1].validation.value_brier)


def test_teacher_action_must_be_represented_by_legal_mask() -> None:
    dataset = _dataset(seed=29)
    dataset["action"][0] = dataset["actions_valid"].shape[1]
    with np.testing.assert_raises_regex(
        ValueError, "teacher action must be present in the encoded legal mask"
    ):
        train_search_supervised(dataset, epochs=1)


def test_chosen_action_target_supports_matched_mcts_ablation() -> None:
    dataset = _dataset(seed=31)
    _, _, initial, history = train_search_supervised(
        dataset,
        policy_target_kind=CHOSEN_ACTION_TARGET,
        epochs=1,
        batch_size=32,
        val_fraction=0.25,
        seed=7,
    )
    assert initial.policy_target_entropy == 0.0
    assert np.isfinite(history[-1].validation.policy_loss)
