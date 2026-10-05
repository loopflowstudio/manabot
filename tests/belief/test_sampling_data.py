"""Whole-game immutable evidence and real frozen-policy collection checks."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from manabot.belief.sampling import SamplerInput, SamplerSchema
from manabot.belief.sampling_data import (
    SamplerDataset,
    SamplerExample,
    SamplerGame,
    Split,
    collect_frozen_policy,
    read_dataset,
    save_dataset,
)
from manabot.env import ObservationSpace
from manabot.env.match import Match
from manabot.infra.hypers import AgentSpec, MatchHypers, ObservationSpaceHypers
from manabot.model.agent import Agent
from manabot.model.world import checkpoint_world


def _dataset() -> SamplerDataset:
    schema = SamplerSchema("test-vocabulary", ("Island", "Mountain"), 0)
    row = SamplerExample(
        SamplerInput("test-vocabulary", (2, 2), (0, 0), 2, ()),
        (1, 1),
        0,
        1,
        "observation",
    )
    splits: tuple[Split, ...] = ("train", "validation", "test")
    return SamplerDataset(
        schema,
        "policy",
        "world",
        tuple(
            SamplerGame(str(index), index, 0, split, (row,))
            for index, split in enumerate(splits)
        ),
    )


def test_dataset_round_trip_and_fail_closed_integrity(tmp_path: Path) -> None:
    dataset = _dataset()
    path = tmp_path / "dataset.json"
    save_dataset(dataset, path)
    assert read_dataset(path) == dataset
    with pytest.raises(FileExistsError):
        save_dataset(dataset, path)
    payload = json.loads(path.read_text())
    payload["dataset"]["games"][0]["examples"][0]["target_hand"] = [2, 0]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        read_dataset(path)
    with pytest.raises(ValueError, match="duplicate"):
        replace(
            dataset,
            games=(
                *dataset.games,
                replace(dataset.games[0], game_id="other", split="test"),
            ),
        )


@pytest.mark.parametrize(
    "compound,max_actions", [(False, 10), (True, 10), (True, 1), (False, 1)]
)
def test_real_frozen_collection_keeps_both_viewers_and_three_game_splits(
    tmp_path: Path,
    compound: bool,
    max_actions: int,
) -> None:
    torch.set_num_threads(1)
    match_hypers = MatchHypers(
        hero_deck={"Mountain": 4, "Raging Goblin": 4},
        villain_deck={"Mountain": 4, "Raging Goblin": 4},
    )
    obs_hypers = ObservationSpaceHypers(max_actions=max_actions)
    obs_space = ObservationSpace(obs_hypers)
    agent_hypers = AgentSpec(hidden_dim=8, compound_decisions=compound)
    agent = Agent(obs_space, agent_hypers)
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "world_binding": checkpoint_world(Match(match_hypers).to_rust(), obs_space),
            "hypers": {
                "observation_hypers": obs_hypers.model_dump(),
                "agent_hypers": agent_hypers.model_dump(),
            },
            "model_state_dict": agent.state_dict(),
        },
        checkpoint,
    )
    if max_actions == 1:
        with pytest.raises(ValueError, match="Observation capacity exceeded: actions"):
            collect_frozen_policy(
                checkpoint=checkpoint,
                match_hypers=match_hypers,
                games=3,
                seed=51,
                max_steps=300,
            )
        return
    dataset = collect_frozen_policy(
        checkpoint=checkpoint,
        match_hypers=match_hypers,
        games=3,
        seed=51,
        max_steps=300,
    )
    assert len(dataset.games) == 3
    assert {game.split for game in dataset.games} == {"train", "validation", "test"}
    assert {game.assignment for game in dataset.games} == {0, 1}
    assert all(
        {row.viewer for row in game.examples} == {0, 1} for game in dataset.games
    )
    assert all(
        sum(row.target_hand) == row.inputs.hand_size
        for game in dataset.games
        for row in game.examples
    )
    path = tmp_path / "real.json"
    save_dataset(dataset, path)
    assert read_dataset(path) == dataset
