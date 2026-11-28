from pathlib import Path

import pytest

from manabot.sim.replay import read_trace, record_actions, replay_games, write_trace
from managym.recording import GameInputs, PlayerSetup


@pytest.fixture
def setup() -> GameInputs:
    return GameInputs(
        seed=7,
        players=[
            PlayerSetup(name="first", decklist={"Mountain": 2}, sideboard={}),
            PlayerSetup(name="second", decklist={"Forest": 2}, sideboard={}),
        ],
    )


def test_record_actions_round_trip(setup: GameInputs, tmp_path: Path) -> None:
    actions = [0, 0, 0]
    game = record_actions(
        setup, actions, players=("a", "b"), source={"kind": "evaluation"}
    )
    path = tmp_path / "game.commands.jsonl.gz"
    write_trace(path, [game])
    games = read_trace(path)
    assert [row["command"]["offer_id"] for row in games[0]["decisions"]] == actions
    receipt = replay_games(games)
    assert receipt.passed and receipt.decisions == len(actions)


def test_record_actions_refuses_unoffered_action(setup: GameInputs) -> None:
    with pytest.raises(ValueError, match="action not offered: 1000"):
        record_actions(setup, [1000], players=("a", "b"), source={})
