"""A released pilot must still compute the recorded policy on its complete game."""

import torch

from scripts.pinned_pilot import PILOT, Recording, verify


def test_recorded_game() -> None:
    recording = PILOT.with_name("game.json").read_bytes()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        verify(PILOT, Recording.model_validate_json(recording))
    finally:
        torch.set_num_threads(threads)
