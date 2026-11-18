"""A released pilot must still compute the recorded policy on its complete game."""

import os
from pathlib import Path

import pytest
import torch

from scripts.pinned_pilot import BEST_PLAYER, FIXTURES, PILOTS, Recording, verify


@pytest.mark.parametrize("name", [*PILOTS, "u1952"])
def test_recorded_game(name: str) -> None:
    directory = FIXTURES / name
    if name == "u1952":
        if BEST_PLAYER not in os.environ:
            pytest.skip(f"set {BEST_PLAYER} to replay the optional frontier checkpoint")
        path = Path(os.environ[BEST_PLAYER])
    else:
        path = directory / PILOTS[name]
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        verify(
            path, Recording.model_validate_json((directory / "game.json").read_bytes())
        )
    finally:
        torch.set_num_threads(threads)
