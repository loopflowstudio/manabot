"""The configured attack runner produces replayable four-leg learning curves."""

import json
from pathlib import Path
import signal

import pytest
import torch

from manabot.arena.match import selected_match
from manabot.arena.models import file_sha256
from manabot.env import Match, ObservationSpace
from manabot.infra.hypers import AgentHypers
from manabot.model.agent import Agent
from manabot.sim.distill import save_bc_checkpoint
from manabot.training import attack_execution
from manabot.training.attack_execution import execute_attack_plan
from manabot.training.attacks import AttackPlan, AttackTarget
from manabot.training.models import (
    FrozenOpponent,
    Learning,
    TrainingRegime,
    TrainSelfPlay,
)
import managym


def _plan(tmp_path: Path) -> AttackPlan:
    torch.set_num_threads(1)
    match = selected_match()
    space = ObservationSpace()
    hypers = AgentHypers(
        hidden_dim=8, num_attention_heads=2, semantic_pack="ur-lessons-vs-gw-allies"
    )
    target = tmp_path / "target.pt"
    torch.manual_seed(17)
    save_bc_checkpoint(
        Agent(space, hypers), space, target, player_configs=Match(match).to_rust()
    )
    plan = AttackPlan(
        id="attack-smoke",
        targets=(
            AttackTarget(
                id="target-a",
                policy=FrozenOpponent(path=str(target), sha256=file_sha256(target)),
                producer_seeds=(17,),
            ),
        ),
        attacker_seeds=(301, 401),
        cumulative_updates=(1, 2),
        final_deal_seeds=(950701,),
        template=TrainingRegime(
            id="attacker",
            world=managym.WORLD_VERSION,
            match=match,
            agent=hypers,
            wall_seconds=30,
            stages=[
                TrainSelfPlay(
                    id="policy",
                    operation="train_self_play",
                    streams=2,
                    transitions=8,
                    learning=Learning(epochs=1, minibatches=1, retained_fraction=1),
                )
            ],
        ),
        total_seconds=180,
        evaluation_seconds=120,
        game_seconds=10,
        prediction="Workflow proof only; no strength prediction at this tiny budget.",
    )
    return plan


def test_selected_world_attack_cohort_replays_all_rungs(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    result = execute_attack_plan(plan, tmp_path / "attack")
    assert result.status == "completed"
    assert len(result.runs) == 2
    assert len(result.points) == 6
    assert sum(point.games for point in result.points) == 24
    assert all(
        point.replay_passed and point.score is not None for point in result.points
    )
    assert {point.updates for point in result.points} == {0, 1, 2}
    assert "2 independent attacker seeds" in (tmp_path / "attack/report.md").read_text()
    assert file_sha256(plan.targets[0].policy.path) == plan.targets[0].policy.sha256


def test_failed_training_is_retained_and_deadline_is_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    handler = signal.getsignal(signal.SIGALRM)

    def failed_training(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected training failure")

    monkeypatch.setattr(attack_execution, "execute_regime", failed_training)
    with pytest.raises(RuntimeError, match="injected"):
        execute_attack_plan(plan, tmp_path / "failed")
    result = json.loads((tmp_path / "failed/result.json").read_text())
    assert result["status"] == "failed"
    assert len(result["run_paths"]) == 1
    assert result["runs"] == []
    assert "injected training failure" in result["error"]
    assert result["seconds"] > 0
    assert signal.getsignal(signal.SIGALRM) == handler
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
