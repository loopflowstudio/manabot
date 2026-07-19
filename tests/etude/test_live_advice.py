from __future__ import annotations

from pathlib import Path
import time

import pytest

from etude.live_advice import LiveBeliefUnavailable
from etude.replay_index import (
    DecisionAddressV2,
    project_replay,
    projection_with_addresses,
)
from etude.server import GameSession
from managym.possible_worlds import WorldQuery


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
        except LiveBeliefUnavailable:
            if time.monotonic() >= deadline:
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


def test_selected_live_root_resolves_tracked_posterior_and_rules_conditions(
    tmp_path: Path,
) -> None:
    game = _selected_game(tmp_path)
    pending = game._pending_decision
    assert pending is not None
    authority_digest = game.env.state_digest()

    resolved = _await_posterior(game, pending.address.serialize())
    posterior = resolved.posterior.posterior
    assert resolved.address == DecisionAddressV2.parse(pending.address.serialize())
    assert resolved.source_sha256 == pending.source_sha256
    assert posterior.space.identity == resolved.posterior.posterior.space.identity
    has_indexes, has_receipt = posterior.space.condition_indexes(
        WorldQuery.has("Counterspell")
    )
    lacks_indexes, lacks_receipt = posterior.space.condition_indexes(
        WorldQuery.lacks("Counterspell")
    )
    assert len(has_indexes) == has_receipt.support_size == 4_820
    assert len(lacks_indexes) == lacks_receipt.support_size == 6_012
    assert set(has_indexes).isdisjoint(lacks_indexes)
    assert len(has_indexes) + len(lacks_indexes) == posterior.support_size == 10_832
    assert game.env.state_digest() == authority_digest
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
