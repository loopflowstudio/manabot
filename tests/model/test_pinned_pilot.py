"""A released pilot must still compute the recorded policy on its complete game."""

import pytest
import torch

from scripts.pinned_pilot import (
    BEST_PLAYER,
    FIXTURES,
    PILOTS,
    Recording,
    pilot_path,
    verify,
)


@pytest.mark.parametrize("name", [*PILOTS, "u1952"])
def test_recorded_game(name: str) -> None:
    path = pilot_path(name)
    if path is None:
        pytest.skip(f"set {BEST_PLAYER} to replay the optional frontier checkpoint")
    recording = (FIXTURES / name / "game.json").read_bytes()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        verify(path, Recording.model_validate_json(recording))
    finally:
        torch.set_num_threads(threads)
