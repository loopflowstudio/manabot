"""Bounded game execution and retained Commands, independent of evaluation policy.

`play_game` supervises one native game child (`manabot.sim.native_game`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from manabot.sim.registration import PlayerRegistration
from manabot.sim.serving_protocol import InferenceRuntime
from manabot.sim.supervision import supervise_game

GAME_WORKER = "manabot.sim.native_game:game_worker"


def play_game(
    game: dict[str, Any],
    registrations: list[PlayerRegistration],
    checkpoint_paths: dict[str, str],
    max_commands: int,
    game_seconds: float,
    evidence_path: Path | None = None,
    keep_completed: bool = False,
    *,
    inference: InferenceRuntime | None = None,
) -> dict[str, Any]:
    return supervise_game(
        game,
        registrations,
        checkpoint_paths,
        max_commands,
        game_seconds,
        evidence_path,
        worker=GAME_WORKER,
        inference=inference,
        keep_completed=keep_completed,
    )
