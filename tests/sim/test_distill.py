"""Tests for search-as-teacher distillation (exp-03 / wave C4)."""

from pathlib import Path

import numpy as np
import torch

from manabot.sim.distill import (
    META_KEYS,
    OBS_KEYS,
    evaluate_bc,
    generate_selfplay_shard,
    load_shards,
    save_bc_checkpoint,
    split_by_game,
    train_bc,
)
from manabot.sim.flat_mc import load_checkpoint_agent


def _tiny_dataset(tmp_path: Path, num_games: int = 3) -> dict[str, np.ndarray]:
    shard = tmp_path / "shard_00.npz"
    summary = generate_selfplay_shard(
        num_games=num_games,
        sims=2,
        seed=11,
        out_path=shard,
    )
    assert summary["decisions"] > 0
    return load_shards([shard])


def test_generate_selfplay_shard_records_both_seats(tmp_path):
    dataset = _tiny_dataset(tmp_path)
    decisions = len(dataset["action"])
    for key in OBS_KEYS + META_KEYS:
        assert len(dataset[key]) == decisions
    # Both players' decisions are captured.
    assert set(np.unique(dataset["seat"]).tolist()) == {0, 1}
    # Every recorded action is valid in its own observation.
    for i, action in enumerate(dataset["action"]):
        assert dataset["actions_valid"][i, int(action)] > 0
    # skip_trivial means every surfaced decision has >= 2 valid actions.
    assert int(dataset["num_valid"].min()) >= 2


def test_generate_selfplay_shard_publishes_only_complete_atomic_file(tmp_path):
    shard = tmp_path / "shard_00.npz"
    fingerprint = "test-run-fingerprint"
    summary = generate_selfplay_shard(
        num_games=1,
        sims=1,
        seed=17,
        out_path=shard,
        dataset_run_fingerprint=fingerprint,
    )
    assert shard.is_file()
    assert not list(tmp_path.glob(".*.tmp"))
    assert summary["provenance"]["dataset_run_fingerprint"] == fingerprint
    assert summary["provenance"]["num_games"] == 1


def test_split_by_game_has_no_leakage(tmp_path):
    dataset = _tiny_dataset(tmp_path)
    train_idx, val_idx = split_by_game(dataset, val_fraction=0.34, seed=0)
    assert len(train_idx) + len(val_idx) == len(dataset["action"])
    train_games = set(dataset["game_index"][train_idx].tolist())
    val_games = set(dataset["game_index"][val_idx].tolist())
    assert train_games and val_games
    assert train_games.isdisjoint(val_games)


def test_train_bc_learns_and_checkpoint_roundtrips(tmp_path):
    dataset = _tiny_dataset(tmp_path, num_games=4)
    agent, obs_space, history = train_bc(
        dataset,
        lr=1e-3,
        epochs=2,
        batch_size=64,
        val_fraction=0.25,
        seed=0,
    )
    assert len(history) == 2
    assert history[-1].train_loss < history[0].train_loss * 1.5  # not diverging
    assert 0.0 <= history[-1].val_accuracy <= 1.0

    path = tmp_path / "bc_policy.pt"
    save_bc_checkpoint(agent, obs_space, path, extra={"lr": 1e-3})
    loaded, loaded_space = load_checkpoint_agent(str(path))
    assert loaded_space == obs_space
    # Loaded policy reproduces the trained policy's logits exactly.
    idx = np.arange(min(8, len(dataset["action"])))
    obs = {
        key: torch.as_tensor(dataset[key][idx], dtype=torch.float32) for key in OBS_KEYS
    }
    with torch.no_grad():
        original_logits, _ = agent.forward(obs)
        loaded_logits, _ = loaded.forward(obs)
    assert torch.allclose(original_logits, loaded_logits)

    loss, acc, acc_nontrivial = evaluate_bc(agent, dataset, idx)
    assert loss > 0.0
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= acc_nontrivial <= 1.0
