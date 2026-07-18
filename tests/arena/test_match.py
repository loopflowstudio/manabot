import json
from pathlib import Path

from experiments.runners.run_skill_arena import play_cells
from manabot.arena.match import derive_seed, play_cell
from manabot.arena.models import ArenaContract, MatchRow
from manabot.arena.replay import read_trace, replay_games

ROOT = Path(__file__).resolve().parents[2]


def test_comparison_alias_shares_seed_without_changing_default_identity() -> None:
    contract = ArenaContract.model_validate(
        json.loads(
            (ROOT / "experiments/contracts/int-6-skill-arena-v1.json").read_text()
        )
    )
    key = contract.key
    default = derive_seed(key, ("candidate-a", "random-v1"), 61001, "candidate-a")
    assert default != derive_seed(
        key, ("candidate-b", "random-v1"), 61001, "candidate-b"
    )
    aliases = {"candidate-a": "guidance-arm", "candidate-b": "guidance-arm"}
    assert derive_seed(
        key,
        ("candidate-a", "random-v1"),
        61001,
        "candidate-a",
        comparison_seed_aliases=aliases,
    ) == derive_seed(
        key,
        ("candidate-b", "random-v1"),
        61001,
        "candidate-b",
        comparison_seed_aliases=aliases,
    )


def test_same_deal_seat_swap_retains_and_replays_commands(tmp_path: Path) -> None:
    contract = ArenaContract.model_validate(
        json.loads(
            (ROOT / "experiments/contracts/int-6-skill-arena-v1.json").read_text()
        )
    )
    random, scripted = contract.anchors[:2]
    rows, trace, replay = play_cell(
        key=contract.key,
        player_a=random,
        player_b=scripted,
        deal_seeds=(777,),
        out_dir=tmp_path,
    )
    assert len(rows) == 2
    assert {row["leg"] for row in rows} == {0, 1}
    assert {row["deal_seed"] for row in rows} == {777}
    assert {row["player_a_seat"] for row in rows} == {0, 1}
    assert all(row["replay_passed"] for row in rows)
    assert all(MatchRow.model_validate(row) for row in rows)
    assert replay["passed"]
    games = read_trace(Path(trace["path"]))
    assert replay_games(games).passed
    assert all(game["decisions"] for game in games)
    assert all(
        decision["command_sha256"] and decision["chosen_offer"]
        for game in games
        for decision in game["decisions"]
    )


def test_replay_rejects_a_command_that_was_not_exactly_retained(tmp_path: Path) -> None:
    contract = ArenaContract.model_validate(
        json.loads(
            (ROOT / "experiments/contracts/int-6-skill-arena-v1.json").read_text()
        )
    )
    _, trace, _ = play_cell(
        key=contract.key,
        player_a=contract.anchors[0],
        player_b=contract.anchors[1],
        deal_seeds=(778,),
        out_dir=tmp_path,
    )
    games = read_trace(Path(trace["path"]))
    games[0]["decisions"][0]["command"]["command_id"] = "fabricated"
    receipt = replay_games(games)
    assert not receipt.passed
    assert receipt.command_mismatches > 0
    assert receipt.trace_mismatches > 0


def test_outcome_worker_runs_a_registered_cell(tmp_path: Path) -> None:
    contract = ArenaContract.model_validate(
        json.loads(
            (ROOT / "experiments/contracts/int-6-skill-arena-v1.json").read_text()
        )
    )
    results = play_cells(
        contract=contract,
        pairs=[(contract.anchors[0], contract.anchors[1])],
        deal_seeds=(783,),
        out_dir=tmp_path,
    )
    assert len(results) == 1
    rows, _, replay = results[0]
    assert len(rows) == 2
    assert replay["passed"]
