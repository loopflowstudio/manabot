"""The shared arena executes and replays explicitly named authored deck roles."""

from pathlib import Path

import pytest

from manabot.arena.match import play_cell
from manabot.arena.matchup import AuthoredMatchup
from manabot.arena.models import ArenaContract, canonical_sha256
from manabot.arena.replay import read_trace
from manabot.training.monitor_evaluation import MonitorProtocol
from managym import WORLD_VERSION

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
        assert matrix.match(leg).content_pack == matrix.pack_key
    with pytest.raises(ValueError, match="distinct"):
        AuthoredMatchup(pack_key="anything", role_a="same", role_b="same")
    with pytest.raises(ValueError, match="between"):
        matrix.decks(8)
    assert "authored_matchup" not in MonitorProtocol().model_dump()
    protocol = MonitorProtocol(authored_matchup=matrix, include_mirrors=True)
    assert MonitorProtocol.model_validate_json(protocol.model_dump_json()) == protocol
    assert protocol.legs == 8


def test_explicit_matrix_replays_all_cells_without_selected_world_alias(
    tmp_path: Path,
) -> None:
    matrix = roster()
    contract = ArenaContract.model_validate_json(
        (ROOT / "tests/fixtures/int-6-skill-arena-v1.json").read_text()
    )
    key = contract.key.model_copy(
        update={"world": WORLD_VERSION, "content_suite": matrix.content_suite}
    )
    players = [
        p.model_copy(
            update={
                "world": WORLD_VERSION,
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
        assert game["match_hypers"]["content_pack"] == matrix.pack_key
    with pytest.raises(ValueError, match="suite differs"):
        play_cell(
            key=contract.key,
            player_a=players[0],
            player_b=players[1],
            deal_seeds=(122501,),
            out_dir=tmp_path / "rejected",
            authored_matchup=matrix,
        )
