"""One Elemental evaluation match: failed groups stay in the denominator, rows mirror seats."""

from typing import Literal

import pytest

from experiments.elemental_grid import GameRow
from experiments.elemental_play import (
    game_rows,
)


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
