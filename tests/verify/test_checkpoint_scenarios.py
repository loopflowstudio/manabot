"""Ordinary fixture checkpoints exercise tactical scoring without training."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from manabot.arena.models import file_sha256
from manabot.arena.replay import replay_games
from manabot.env import Match, ObservationSpace
from manabot.infra.hypers import AgentSpec, MatchHypers, ObservationSpaceHypers
from manabot.model.agent import Agent
from manabot.sim.distill import save_bc_checkpoint
from manabot.sim.flat_mc import make_player
from manabot.verify.checkpoint_scenarios import main, score_checkpoint
from manabot.verify.competency import SCENARIOS, build_scenario_env


def _checkpoint(
    tmp_path: Path, name: str, *, max_actions: int = 64, compound: bool = False
) -> Path:
    torch.set_num_threads(1)
    torch.manual_seed(17)
    scenario = SCENARIOS[name]
    space = ObservationSpace(ObservationSpaceHypers(max_actions=max_actions))
    agent = Agent(
        space,
        AgentSpec(hidden_dim=8, num_attention_heads=2, compound_decisions=compound),
    )
    path = tmp_path / f"{name}.pt"
    save_bc_checkpoint(
        agent,
        space,
        path,
        player_configs=Match(
            MatchHypers(
                hero_deck=scenario.hero_deck,
                villain_deck=scenario.villain_deck,
            )
        ).to_rust(),
    )
    return path


@pytest.mark.parametrize("name", SCENARIOS)
def test_bound_checkpoint_scores_and_replays(tmp_path: Path, name: str) -> None:
    checkpoint = _checkpoint(tmp_path, name)
    score, game = score_checkpoint(name, checkpoint)
    assert score.status == "scored", score.error
    assert score.replay_passed and score.decisions > 0
    assert score.correct is not None
    assert game is not None
    repeated, repeated_game = score_checkpoint(name, checkpoint)
    assert repeated.model_dump(exclude={"seconds"}) == score.model_dump(
        exclude={"seconds"}
    )
    assert repeated_game == game
    assert replay_games([game], require_terminal=False).passed
    if not game["terminated"]:
        assert not replay_games([game]).passed
    changed = deepcopy(game)
    changed["decisions"][0]["command"]["expected_revision"] += 1
    assert not replay_games([changed], require_terminal=False).passed
    changed = deepcopy(game)
    changed["scenario_root"]["world"] = "wrong"
    with pytest.raises(ValueError, match="version/world/source"):
        replay_games([changed], require_terminal=False)


def test_wrong_world_setup_and_capacity_fail_admission(tmp_path: Path) -> None:
    name = next(iter(SCENARIOS))
    path = _checkpoint(tmp_path, name)
    other = list(SCENARIOS)[1]
    score, _ = score_checkpoint(other, path)
    assert score.status == "failed" and score.correct is None
    assert "setup differs" in (score.error or "")
    data = torch.load(path, weights_only=False)
    data["world_binding"]["world"] = "wrong-world"
    torch.save(data, path)
    score, game = score_checkpoint(name, path)
    assert score.status == "failed" and game is None
    assert "world binding" in (score.error or "")
    path = _checkpoint(tmp_path, name, max_actions=1)
    score, _ = score_checkpoint(name, path)
    assert score.status == "failed" and score.correct is None
    assert "capacity" in (score.error or "").lower()


def test_cli_report_regenerates_without_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignments = {name: str(_checkpoint(tmp_path, name)) for name in SCENARIOS}
    manifest = tmp_path / "checkpoints.json"
    manifest.write_text(json.dumps(assignments))
    out = tmp_path / "scores"
    monkeypatch.setattr(
        "sys.argv", ["scenarios", "--checkpoints", str(manifest), "--out", str(out)]
    )
    main()
    result = json.loads((out / "results.json").read_text())
    assert all(row["status"] == "scored" for row in result["scores"])
    original = file_sha256(out / "report.md")
    for path in assignments.values():
        Path(path).unlink()
    monkeypatch.setattr("sys.argv", ["scenarios", "--report-only", "--out", str(out)])
    main()
    assert file_sha256(out / "report.md") == original
    assert "Premise validation" in (out / "report.md").read_text()


def test_hidden_hand_swap_preserves_checkpoint_inputs_and_decision(
    tmp_path: Path,
) -> None:
    name = next(iter(SCENARIOS))
    path = _checkpoint(tmp_path, name)
    player, space = make_player(
        {"kind": "checkpoint", "path": str(path), "deterministic": True},
        seed=2,
    )
    env, observation, _ = build_scenario_env(SCENARIOS[name], space, 2)
    try:
        action = player.act(env, observation)
        before = env._engine.state_digest()
        env._engine.scenario_clear_hand(1)
        for _ in range(2):
            env._engine.scenario_force_card_in_hand(1, "Mountain")
        changed, _ = env.scenario_refresh()
        assert env._engine.state_digest() != before
        for key in observation:
            np.testing.assert_array_equal(observation[key], changed[key])
        assert player.act(env, changed) == action
    finally:
        env.close()


def test_command_cap_retains_failed_prefix(tmp_path: Path) -> None:
    name = next(iter(SCENARIOS))
    score, game = score_checkpoint(name, _checkpoint(tmp_path, name), max_steps=1)
    assert score.status == "failed" and score.correct is None
    assert score.decisions == 1 and score.replay_passed
    assert "Command cap" in (score.error or "")
    assert game is not None and game["failure"]


def test_compound_checkpoint_combat_prefix_replays(tmp_path: Path) -> None:
    name = "s4_race_vs_block"
    checkpoint = _checkpoint(tmp_path, name, compound=True)
    score, game = score_checkpoint(name, checkpoint)
    assert score.status == "scored", score.error
    assert score.replay_passed and game is not None
    repeated, repeated_game = score_checkpoint(name, checkpoint)
    assert repeated.trace_sha256 == score.trace_sha256
    assert repeated_game == game
