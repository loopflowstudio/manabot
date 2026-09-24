from __future__ import annotations

from pathlib import Path
import time

import pytest

from etude.live_advice import LiveBeliefUnavailable
from etude.replay_index import (
    project_replay,
    projection_with_addresses,
)
from etude.server import GameSession


def _selected_game(trace_dir: Path) -> GameSession:
    game = GameSession(
        trace_dir=trace_dir,
        id_factory=lambda kind: f"{kind}.live-advice",
    )
    game.new_game(
        {
            "hero_deck": "interactive",
            "villain_deck": "interactive",
            "villain_type": "passive",
            "seed": 197,
            "auto_pass": False,
        }
    )
    return game


def _await_posterior(game: GameSession, address: str):
    deadline = time.monotonic() + 60.0
    while True:
        try:
            return game.resolve_live_tracked_posterior(address, 0)
        except LiveBeliefUnavailable as error:
            if error.__cause__ is not None or time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def test_current_ed2_promotes_to_the_exact_committed_replay_identity(
    tmp_path: Path,
) -> None:
    game = _selected_game(tmp_path)
    pending = game._pending_decision
    assert pending is not None
    live_address = pending.address.serialize()
    state_before = game.env.state_digest()

    game.hero_action(0)

    projection = project_replay(game.canonical_replay(), 0)
    committed = projection_with_addresses(projection)["decisions"][0]
    assert live_address.startswith("ed2.")
    assert committed["address"] == live_address
    assert game._study_roots[0].state_digest() == state_before
    game.close("test")


def test_selected_live_root_rejects_retired_likelihood_checkpoint(
    tmp_path: Path,
) -> None:
    game = _selected_game(tmp_path)
    pending = game._pending_decision
    assert pending is not None
    authority_digest = game.env.state_digest()

    try:
        with pytest.raises(LiveBeliefUnavailable, match="failed") as failure:
            _await_posterior(game, pending.address.serialize())
        assert isinstance(failure.value.__cause__, ValueError)
        assert "max_conditions" in str(failure.value.__cause__)
        assert game.env.state_digest() == authority_digest
    finally:
        game.close("test")


def test_unsupported_match_fails_without_substituting_a_posterior(
    tmp_path: Path,
) -> None:
    game = GameSession(
        trace_dir=tmp_path,
        id_factory=lambda kind: f"{kind}.unsupported",
    )
    game.new_game({"seed": 197, "auto_pass": False})
    pending = game._pending_decision
    assert pending is not None
    with pytest.raises(LiveBeliefUnavailable, match="unavailable"):
        game.resolve_live_tracked_posterior(pending.address.serialize(), 0)
    game.close("test")
