"""Search-generated datasets retain seat ownership and whole-game partitions."""

from pathlib import Path

import numpy as np

from manabot.sim.data import META_KEYS, OBS_KEYS, load_shards
from manabot.sim.datagen import generate_selfplay_shard
from manabot.training.data import split_by_game


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
