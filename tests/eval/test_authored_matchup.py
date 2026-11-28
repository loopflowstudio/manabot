"""The shared arena executes and replays explicitly named authored deck roles."""

from pathlib import Path

import pytest

from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.eval.match import play_cell
from manabot.eval.monitor import MonitorProtocol
from manabot.infra.config import MatchConfig
from manabot.infra.identity import canonical_sha256
from manabot.sim.matchup import AuthoredMatchup
from manabot.sim.replay import read_trace
from managym import Env, authored_deck_setup
from tests.eval.arena_fixture import arena_fixture
from tests.fixtures.observation_rows import observation_arrays

ROOT = Path(__file__).resolve().parents[2]


def roster() -> AuthoredMatchup:
    return AuthoredMatchup(
        pack_key="tla-jeong-increment-v1",
        role_a="ur_lessons",
        role_b="gw_allies_jeong",
    )


def test_authored_matrix_has_four_roles_and_exact_native_selection() -> None:
    matrix = roster()
    assert [matrix.decks(i) for i in range(0, 8, 2)] == [
        (matrix.role_a, matrix.role_b),
        (matrix.role_b, matrix.role_a),
        (matrix.role_a, matrix.role_a),
        (matrix.role_b, matrix.role_b),
    ]
    for leg in range(8):
        assert matrix.match(leg).world is None
    with pytest.raises(ValueError, match="distinct"):
        AuthoredMatchup(pack_key="anything", role_a="same", role_b="same")
    with pytest.raises(ValueError, match="between"):
        matrix.decks(8)
    assert "authored_matchup" not in MonitorProtocol().model_dump()
    protocol = MonitorProtocol(authored_matchup=matrix, include_mirrors=True)
    assert MonitorProtocol.model_validate_json(protocol.model_dump_json()) == protocol
    assert protocol.legs == 8


def test_open_authored_matrix_preserves_serialized_identities() -> None:
    matrix = roster()
    original = {
        "pack_key": matrix.pack_key,
        "role_a": matrix.role_a,
        "role_b": matrix.role_b,
    }
    assert matrix.model_dump() == original
    assert matrix.content_suite == (f"w17-authored-matrix-{canonical_sha256(original)}")
    explicit = AuthoredMatchup(**original, decklists="open")
    assert explicit.model_dump_json() == matrix.model_dump_json()
    for leg in range(8):
        first, second = (
            authored_deck_setup(matrix.pack_key, role) for role in matrix.decks(leg)
        )
        original_match = MatchConfig(
            hero="arena-seat-0",
            villain="arena-seat-1",
            hero_deck=first.decklist,
            villain_deck=second.decklist,
            hero_sideboard=first.sideboard,
            villain_sideboard=second.sideboard,
        )
        assert matrix.match(leg).model_dump_json() == original_match.model_dump_json()


def test_closed_authored_matrix_hides_opponent_decklists_in_every_leg() -> None:
    opened = roster()
    closed = AuthoredMatchup(**opened.model_dump(), decklists="closed")
    assert closed.content_suite != opened.content_suite
    assert closed.model_dump()["decklists"] == "closed"
    assert AuthoredMatchup.model_validate_json(closed.model_dump_json()) == closed
    space = ObservationSpace()
    env = Env(seed=122501)
    for leg in range(8):
        assert closed.match(leg).decklists == "closed"
        env.reset(Match(closed.match(leg)).to_rust())
        for viewer in (0, 1):
            observation = env.observation_for_player(viewer)
            assert observation.decklists_closed
            rows = observation_arrays(space, observation)["decklists"]
            assert rows[0].any()
            assert not rows[1].any()
    with pytest.raises(ValueError):
        AuthoredMatchup.model_validate({**opened.model_dump(), "decklists": "mixed"})


def test_explicit_matrix_replays_all_cells_without_selected_world_alias(
    tmp_path: Path,
) -> None:
    matrix = roster()
    contract = arena_fixture()
    key = contract.key.model_copy(update={"content_suite": matrix.content_suite})
    players = [
        p.model_copy(
            update={
                "content_suite": matrix.content_suite,
                "matchup_sha256": canonical_sha256(matrix.match().model_dump()),
            }
        )
        for p in contract.anchors[:2]
    ]
    rows, trace, replay = play_cell(
        key=key,
        player_a=players[0],
        player_b=players[1],
        deal_seeds=(122501,),
        out_dir=tmp_path,
        authored_matchup=matrix,
        matchup_mode="all",
        game_seconds=30,
    )
    assert len(rows) == 8
    assert replay["passed"]
    assert all(row["terminated"] and row["replay_passed"] for row in rows)
    for row in rows:
        assert row["seat_decks"] == list(matrix.decks(row["leg"]))
        assert row["player_a_seat"] == row["leg"] % 2
    for game in read_trace(Path(trace["path"])):
        assert "world" not in game["match_hypers"]
    with pytest.raises(ValueError, match="suite differs"):
        play_cell(
            key=contract.key,
            player_a=players[0],
            player_b=players[1],
            deal_seeds=(122501,),
            out_dir=tmp_path / "rejected",
            authored_matchup=matrix,
        )
