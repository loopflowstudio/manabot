"""One Elemental evaluation match: failed groups stay in the denominator, rows mirror seats."""

import hashlib
from pathlib import Path
from typing import Literal

from etude.checkpoints import CheckpointRef, checkpoint_id
from etude.eval.records import Inference, Match
from etude.records import ArtifactRef
import pytest

from experiments.elemental_grid import DECKS, GameRow, Group
from experiments.elemental_play import (
    INFERENCE_CODE,
    Conditions,
    game_rows,
    heuristic_code,
    play,
)


def test_failed_group_is_stored_as_invalid_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that cannot load the checkpoint fails its group, not the run."""
    monkeypatch.setenv("ETUDE_OUTPUT_DIR", str(tmp_path))
    body = b"not a checkpoint"
    path = tmp_path / "updates-0.pt"
    path.write_bytes(body)
    checkpoint = CheckpointRef(
        key=checkpoint_id(run="seed-1", name=path.name),
        artifact=ArtifactRef(
            uri=path.as_uri(),
            sha256=hashlib.sha256(body).hexdigest(),
            size_bytes=len(body),
        ),
        at={"updates": 0},
    )
    row = Conditions(
        kind="full",
        deck=DECKS[0],
        groups=(
            Group(
                model_deck=DECKS[0], heuristic_deck=DECKS[1], model_seat=0, deals=(1,)
            ),
        ),
    )
    match = Match(
        players=(
            Inference(source=list(INFERENCE_CODE)).player(checkpoint),
            heuristic_code(),
        ),
        config=row,
    )
    result = play(match)
    games = result.tables["games"]
    assert [(game["seat"], game["valid"], game["score"]) for game in games] == [
        (0, False, None),
        (1, False, None),
    ]
    assert str(games[0]["failure"]).startswith("group failed")
    assert not result.blobs
    assert list((tmp_path / "arena" / match.key).glob("attempt-*/failed.txt"))


@pytest.mark.parametrize("game_seat", [0, 1])
def test_game_rows_mirror_each_game_for_both_players(game_seat: Literal[0, 1]) -> None:
    row = GameRow(
        model_deck="elemental_ur",
        heuristic_deck="elemental_bg",
        model_seat=game_seat,
        deal=7,
        score=1.0,
        valid=True,
        terminated=True,
        truncated=False,
        failure=None,
        decisions=40,
        model_decisions=20,
        seconds=1.5,
    )
    first, second = game_rows([row], "full")
    assert (first["deck"], first["seat"], first["game_seat"], first["score"]) == (
        "elemental_ur",
        0,
        game_seat,
        1.0,
    )
    assert (second["deck"], second["seat"], second["game_seat"], second["score"]) == (
        "elemental_bg",
        1,
        1 - game_seat,
        0.0,
    )
    assert first["unit"] == second["unit"] == "full"
