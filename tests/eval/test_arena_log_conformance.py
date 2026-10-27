"""Independent native-metric and pinned-public-parser conformance, offline."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from pydantic import JsonValue
import pytest

from manabot.eval.arena_log import ArenaPrinting, CardMapping, export_arena_log
from manabot.eval.card_evaluation import verification_spec
from manabot.eval.card_metrics import reduce_cards
from manabot.infra.config import MatchConfig
from manabot.infra.identity import canonical_sha256
from manabot.sim.card_evidence import GameTrace, capture_verified_game
from manabot.sim.match import play_game
from tests.eval.test_card_evaluation import card_fixture

FIXTURES = Path(__file__).parents[1] / "fixtures"


def mapping() -> CardMapping:
    return CardMapping(
        printings=tuple(
            ArenaPrinting(
                card=f"{'a' * 64}:{i}",
                scryfall=json.loads(
                    (FIXTURES / f"arena-card-log/{name}.json").read_text()
                ),
            )
            for i, name in ((1, "mountain"), (2, "forest"))
        )
    )


def _parse_log(path: Path) -> dict[str, JsonValue]:
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("arena_log_harness.py")),
            str(path),
        ],
        cwd=path.parent,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    payload = json.loads(result.stdout)
    assert payload["errors"] == []
    assert len(payload["results"]) == 1
    return payload["results"][0]


@pytest.mark.parametrize("viewer", [0, 1])
def test_export_matches_pinned_parser_fields_without_network(
    tmp_path: Path, viewer: int
) -> None:
    assert (
        sha256((FIXTURES / "seventeenlands/mtga_follower.py").read_bytes()).hexdigest()
        == "71117bdd8b1ef75cc019bbddfaeab9a69a02c5ef9efd741ec05a2e660918f77d"
    )
    game = card_fixture()
    log = export_arena_log(game, viewer, mapping())
    path = tmp_path / "game.log"
    path.write_text(log)
    parsed = _parse_log(path)
    printing = 70504 if viewer == 0 else 70505
    assert parsed["match_id"] == game.game
    assert parsed["opening_hand"] == [printing] * (2 if viewer == 0 else 1)
    assert parsed["maindeck_card_ids"] == [printing] * (3 if viewer == 0 else 2)
    assert parsed["sideboard_card_ids"] == []
    assert parsed["drawn_hands"] == [] and parsed["mulligans"] == []
    assert parsed["mulligan_count"] == parsed["opponent_mulligan_count"] == -1
    assert parsed["turns"] == 8
    assert parsed["on_play"] == (viewer == 0)
    assert parsed["won"] == (viewer == 0)
    assert parsed["win_type"] == "ResultType_WinLoss"
    assert parsed["history"]["seat_id"] == viewer + 1
    assert Counter(parsed["drawn_cards"]) == {printing: 6 if viewer == 0 else 2}
    # A bounce and tutor appear in parser hand sightings; native GIH excludes them.
    rows = reduce_cards(game, game.seat_players)
    assert next(r for r in rows if r.seat == viewer).gih_copies == (
        4 if viewer == 0 else 2
    )
    assert parsed["opponent_card_ids"] == ([] if viewer == 0 else [70504])
    for event in parsed["history"]["events"]:
        state = event["gameStateMessage"]
        for zone in state["zones"]:
            if (
                zone["type"] == "ZoneType_Library"
                or zone["type"] == "ZoneType_Hand"
                and zone["ownerSeatId"] != viewer + 1
            ):
                assert zone["objectInstanceIds"] == []


def test_export_refuses_unmapped_or_mislabeled_printings() -> None:
    with pytest.raises(ValueError, match="printing unavailable"):
        export_arena_log(card_fixture(), 0, CardMapping(printings=()))
    wrong = mapping().printings[1].model_copy(update={"card": "a" * 64 + ":1"})
    with pytest.raises(ValueError, match="printing unavailable"):
        export_arena_log(card_fixture(), 0, CardMapping(printings=(wrong,)))


def test_native_game_replay_exports_both_viewers(tmp_path: Path) -> None:
    """Play the real bounded arena path with a small, fully mapped land fixture."""
    players = verification_spec(8).players
    setup = MatchConfig(
        hero=players[0].player_id,
        villain=players[1].player_id,
        hero_deck={"Mountain": 12},
        villain_deck={"Forest": 12},
    )
    data = dict(card_fixture().trace.data)
    data.pop("game_trace_sha256")
    data.update(
        match_id="native-log-fixture",
        seat_players=[p.player_id for p in players],
        match_hypers=setup.model_dump(),
        winner=None,
        terminated=False,
        failure=None,
        failed_player_id=None,
        player_seeds={p.player_id: 23 for p in players},
        integrity={
            name: 0
            for name in (
                "private_exposures",
                "root_mutations",
                "illegal_actions",
                "truncations",
            )
        },
    )
    trace = play_game(data, list(players), {}, 1000, 30, tmp_path / "native.jsonl")
    assert trace["terminated"] and trace["failure"] is None
    trace["game_trace_sha256"] = canonical_sha256(trace)
    game = capture_verified_game(GameTrace(data=trace))
    assert game.replay["passed"]
    assert game.cards.turn == 12
    assert [row.gih_copies for row in reduce_cards(game, game.seat_players)] == [12, 12]
    identities = {card.definition: card.name for card in game.cards.capture.initial}
    card_mapping = CardMapping(
        printings=tuple(
            ArenaPrinting(
                card=f"{game.cards.capture.content_digest}:{definition}",
                scryfall=json.loads(
                    (FIXTURES / f"arena-card-log/{name.lower()}.json").read_text()
                ),
            )
            for definition, name in identities.items()
        )
    )
    for viewer in (0, 1):
        log = tmp_path / f"native-{viewer}.log"
        log.write_text(export_arena_log(game, viewer, card_mapping))
        parsed = _parse_log(log)
        printing = 70504 if viewer == 0 else 70505
        assert parsed["opening_hand"] == [printing] * 7
        assert parsed["maindeck_card_ids"] == [printing] * 12
        assert parsed["drawn_cards"] == [printing] * 12
        assert parsed["turns"] == 12
        assert parsed["won"] == (game.winner == viewer)
        assert parsed["on_play"] == (viewer == 0)
