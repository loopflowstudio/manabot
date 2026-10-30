import json
from multiprocessing.connection import Connection
from pathlib import Path
import time

import pytest

from manabot.eval.match import derive_seed, play_cell
from manabot.eval.models import EvaluationContract, MatchRow
from manabot.sim.replay import read_trace, replay_games
from managym import WORLD_VERSION

ROOT = Path(__file__).resolve().parents[2]


def test_comparison_alias_shares_seed_without_changing_default_identity() -> None:
    contract = EvaluationContract.model_validate(
        json.loads((ROOT / "tests/fixtures/int-6-skill-arena-v1.json").read_text())
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
    contract = EvaluationContract.model_validate(
        json.loads((ROOT / "tests/fixtures/int-6-skill-arena-v1.json").read_text())
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
    counters = json.loads((tmp_path / "performance.json").read_text())["counters"]
    assert counters["threads/group_worker"] >= 1
    assert counters["threads_budget/group_worker"] == 8


def test_replay_rejects_a_command_that_was_not_exactly_retained(tmp_path: Path) -> None:
    contract = EvaluationContract.model_validate(
        json.loads((ROOT / "tests/fixtures/int-6-skill-arena-v1.json").read_text())
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


def selected_players():
    from manabot.infra.identity import canonical_sha256
    from manabot.sim.matchup import SELECTED_SUITE, selected_match

    contract = EvaluationContract.model_validate(
        json.loads((ROOT / "tests/fixtures/int-6-skill-arena-v1.json").read_text())
    )
    key = contract.key.model_copy(
        update={"world": WORLD_VERSION, "content_suite": SELECTED_SUITE}
    )
    players = [
        player.model_copy(
            update={
                "world": WORLD_VERSION,
                "content_suite": SELECTED_SUITE,
                "matchup_sha256": canonical_sha256(selected_match().model_dump()),
            }
        )
        for player in contract.anchors[:2]
    ]
    return key, players


def test_selected_four_game_block_preserves_decks_sideboards_and_replay(tmp_path):
    from manabot.eval.rating import bootstrap_population, payoff_matrix
    from manabot.sim.matchup import selected_match
    from manabot.sim.replay import replay_environment

    key, (random, scripted) = selected_players()
    rows, trace, receipt = play_cell(
        key=key,
        player_a=random,
        player_b=scripted,
        deal_seeds=(83002,),
        out_dir=tmp_path,
    )
    assert len(rows) == 4
    assert {
        (r["seat_decks"][r["player_a_seat"]], r["player_a_seat"]) for r in rows
    } == {(deck, seat) for deck in ("ur_lessons", "gw_allies") for seat in (0, 1)}
    assert all(MatchRow.model_validate(row) for row in rows)
    assert all(row["failure"] is None for row in rows)
    assert receipt["passed"]
    games = read_trace(Path(trace["path"]))
    setup = selected_match()
    for game in games:
        env, _ = replay_environment(game)
        ur_seat = game["seat_decks"].index("ur_lessons")
        assert env.match.hero_sideboard == (
            setup.hero_sideboard if ur_seat == 0 else setup.villain_sideboard
        )
        assert env.match.villain_sideboard == (
            setup.villain_sideboard if ur_seat == 0 else setup.hero_sideboard
        )
        assert env._engine.state_digest() == game["initial_state_digest"]
    matrix = payoff_matrix(rows)["random-v1__scripted-greedy-v1"]
    assert matrix["games"] == 4
    assert sum(matrix["paired_blocks"].values()) == 1
    assert bootstrap_population(rows, seed=83, replicates=5)["failures"] == 0
    games[0]["match_hypers"]["hero_sideboard"] = {}
    assert not replay_games(games).passed


def test_selected_registration_cannot_claim_other_matchup(tmp_path):
    import pytest

    key, (first, second) = selected_players()
    wrong = first.model_copy(update={"matchup_sha256": "0" * 64})
    with pytest.raises(ValueError, match="not bound"):
        play_cell(
            key=key, player_a=wrong, player_b=second, deal_seeds=(83,), out_dir=tmp_path
        )
    assert not list(tmp_path.iterdir())


def test_game_cap_retains_all_four_failed_prefixes_without_draw_credit(tmp_path):
    import pytest

    from manabot.eval.rating import fit_population

    key, (first, second) = selected_players()
    rows, trace, receipt = play_cell(
        key=key,
        player_a=first,
        player_b=second,
        deal_seeds=(83,),
        out_dir=tmp_path,
        max_commands=1,
    )
    assert len(rows) == 4
    assert all(MatchRow.model_validate(row) for row in rows)
    assert all(
        row["termination_reason"] == "command_cap" and row["score_a"] is None
        for row in rows
    )
    assert receipt["passed"]  # Prefix replay is distinct from successful completion.
    assert all(len(game["decisions"]) == 1 for game in read_trace(Path(trace["path"])))
    with pytest.raises(ValueError, match="do not drop"):
        fit_population(rows)
    draw = {
        **rows[0],
        "failure": None,
        "termination_reason": "draw",
        "terminated": True,
        "score_a": 0.5,
    }
    assert MatchRow.model_validate(draw).score_a == 0.5
    with pytest.raises(ValueError):
        MatchRow.model_validate({**draw, "terminated": False})
    forfeit = {
        **rows[0],
        "termination_reason": "crash",
        "failed_player_id": first.player_id,
        "score_a": 0.0,
    }
    assert MatchRow.model_validate(forfeit).score_a == 0.0
    assert (
        fit_population([forfeit, {**forfeit, "leg": 1, "player_a_seat": 1}]).rows[0][
            "score_a"
        ]
        == 0.0
    )


def _dead_worker(
    connection, game, registrations, checkpoint_paths, max_commands, *_endpoint
):
    import os

    connection.send(("phase", game["seat_players"][0]))
    os._exit(7)


@pytest.mark.parametrize("exit_between_polls", [False, True])
def test_native_worker_crash_and_timeout_are_retained(
    monkeypatch: pytest.MonkeyPatch, exit_between_polls: bool
) -> None:
    from copy import deepcopy

    import manabot.sim.match as match

    _, players = selected_players()
    game = {"seat_players": [p.player_id for p in players], "decisions": []}
    monkeypatch.setattr(
        match, "GAME_WORKER", "tests.eval.test_evaluation_match:_dead_worker"
    )
    if exit_between_polls:
        original_poll = Connection.poll

        def poll_before_worker_exit(
            connection: Connection, timeout: float | None = 0.0
        ) -> bool:
            ready = original_poll(connection, timeout)
            if ready and timeout:
                # Model a timed poll that missed the final phase just before
                # the worker exited. The next nonblocking poll sees its bytes.
                time.sleep(0.1)
                return False
            return ready

        monkeypatch.setattr(Connection, "poll", poll_before_worker_exit)
    crashed = match.play_game(deepcopy(game), players, {}, 10000, 20)
    assert crashed["termination_reason"] == "crash", crashed
    assert crashed["failed_player_id"] == players[0].player_id, crashed
    assert crashed["winner"] is None
    timed_out = match.play_game(deepcopy(game), players, {}, 10000, 0.001)
    assert timed_out["termination_reason"] == "timeout"
    assert timed_out["failed_player_id"] is None
    assert timed_out["winner"] is None


def test_policy_exception_retains_player_failure(monkeypatch):
    from manabot.sim.native_game import execute_game

    _, players = selected_players()
    game = {"player_seeds": {p.player_id: 1 for p in players}}

    def fail(*args, **kwargs):
        raise RuntimeError("policy failed")

    monkeypatch.setattr("manabot.sim.native_game.build_code_player", fail)
    messages = []
    execute_game(game, players, {}, 1, messages.append)
    result = messages[-1][1]
    assert result["termination_reason"] == "crash"
    assert result["failed_player_id"] == players[0].player_id
    assert "policy failed" in result["failure"]


def test_checkpoint_observation_bounds_survive_selected_replay(tmp_path):
    import torch

    from manabot.env.observation import ObservationSpace
    from manabot.eval.evaluate import validate_registered_player
    from manabot.infra.config import ModelConfig, ObservationConfig
    from manabot.infra.identity import file_sha256
    from manabot.infra.provenance import runtime_fingerprints
    from manabot.model.actor_critic import ActorCritic
    from manabot.model.checkpoint import save_checkpoint
    from manabot.sim.matchup import selected_match
    from manabot.sim.registration import PlayerRegistration
    from manabot.sim.replay import replay_environment

    key, (random, _) = selected_players()
    space = ObservationSpace(
        ObservationConfig(
            max_actions=128, max_cards_per_player=96, max_permanents_per_player=64
        )
    )
    torch.manual_seed(83)
    agent = ActorCritic(space, ModelConfig(world="ur-lessons-vs-gw-allies"))
    path = tmp_path / "fixture.pt"
    from manabot.env.match import Match

    save_checkpoint(
        agent, space, path, player_configs=Match(selected_match()).to_rust()
    )
    runtime = runtime_fingerprints(
        match_config=selected_match(), observation_space=space, world=WORLD_VERSION
    )
    candidate = PlayerRegistration.model_validate(
        {
            **random.model_dump(),
            "player_id": "checkpoint-fixture-83",
            "role": "challenger",
            "runner_kind": "checkpoint",
            "source_sha256": None,
            "checkpoint_sha256": file_sha256(path),
            "checkpoint_bytes": path.stat().st_size,
            "parameter_count": sum(p.numel() for p in agent.parameters()),
            "training_seed": 83,
            "artifact_id": "untrained-fixture-seed-83",
            "evidence_class": "fixture",
            "player_spec": {
                "kind": "checkpoint",
                "deterministic": False,
                "device": "cpu",
                "batch_size": 1,
            },
            "observation_abi_sha256": runtime["observation_abi_sha256"],
            "action_abi_sha256": runtime["action_abi_sha256"],
        }
    )
    paths, loaded_space = validate_registered_player(candidate, path)
    assert loaded_space.shapes == space.shapes
    rows, trace, receipt = play_cell(
        key=key,
        player_a=candidate,
        player_b=random,
        deal_seeds=(831,),
        out_dir=tmp_path,
        checkpoint_paths=paths,
        max_commands=2,
    )
    assert receipt["passed"]
    for game in read_trace(Path(trace["path"])):
        env, _ = replay_environment(game)
        assert env.obs_space.shapes == space.shapes
        assert game["termination_reason"] == "command_cap"
    path.write_bytes(b"changed")
    import pytest

    with pytest.raises(RuntimeError, match="SHA-256"):
        validate_registered_player(candidate, path)
