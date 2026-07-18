import json
from pathlib import Path

from manabot.arena.match import play_cell
from manabot.arena.models import (
    ArenaContract,
    PlayerRegistration,
    ProfileRoots,
    canonical_sha256,
)
from manabot.arena.profile import profile_players, select_profile_roots, verify_profile
from manabot.arena.replay import read_trace

ROOT = Path(__file__).resolve().parents[2]


def test_profile_roots_are_canonical_and_shared() -> None:
    games = [
        {
            "match_id": "later",
            "deal_seed": 2,
            "leg": 0,
            "game_trace_sha256": "b" * 64,
            "decisions": [{"revision": 0}, {"revision": 1}],
        },
        {
            "match_id": "earlier",
            "deal_seed": 1,
            "leg": 1,
            "game_trace_sha256": "a" * 64,
            "decisions": [{"revision": 0}, {"revision": 1}],
        },
    ]
    roots = select_profile_roots(games, warmup=1, measured=2)
    assert [root["game"]["deal_seed"] for root in roots] == [1, 1, 2]
    assert roots[0]["root_id"] == canonical_sha256(["a" * 64, 0])


def test_random_player_has_isolated_matched_root_cost(tmp_path: Path) -> None:
    contract = ArenaContract.model_validate(
        json.loads(
            (ROOT / "experiments/contracts/int-6-skill-arena-v1.json").read_text()
        )
    )
    _, trace, _ = play_cell(
        key=contract.key,
        player_a=contract.anchors[0],
        player_b=contract.anchors[1],
        deal_seeds=(779,),
        out_dir=tmp_path,
    )
    candidate = PlayerRegistration.model_validate(
        json.loads(
            (ROOT / "experiments/candidates/int-6-dpuct-32-w4-v1.json").read_text()
        )
    )
    profile = profile_players(
        [contract.anchors[0], candidate],
        source_games=read_trace(Path(trace["path"])),
        profile_roots=ProfileRoots(
            source_cell="random-v1__scripted-greedy-v1",
            selection="deal-leg-revision-canonical-v1",
            warmup=0,
            measured=1,
            sampler_interval_ms=5,
        ),
    )
    verify_profile(profile)
    player = profile["players"]["random-v1"]
    assert len(player["samples"]) == 1
    assert player["p95_seconds"] >= 0.0
    assert player["peak_rss_delta_bytes"] >= 0
    assert player["root_mutations"] == 0
    assert player["illegal_actions"] == 0
    dpuct = profile["players"][candidate.player_id]
    assert dpuct["samples"][0]["simulations"] == 32
    assert dpuct["playout_cap_rate"] <= 0.001
    assert dpuct["root_mutations"] == 0
