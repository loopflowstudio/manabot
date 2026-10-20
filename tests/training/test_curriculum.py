"""Curriculum arms continue one learning state across a declared deck switch."""

from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from experiments.runners.complexity_campaign import (
    TESTED_SCHEDULE,
    Anchor,
    parameter_digest,
    read_complexity_plan,
)
from experiments.runners.curriculum_analysis import paired_difference
from experiments.runners.curriculum_campaign import (
    BOUNDARY_DEALS,
    COMPLEXITY_PLAN,
    SEEDS,
    Arm,
    CurriculumPlan,
    Fit,
    Recipe,
    _reference,
    evaluate_retained,
    evaluate_seed,
    fit_checkpoints,
    initial_checkpoint,
    initialization_digest,
    source_recipe,
    target_recipe,
    train_fit,
)
from experiments.runners.environment_complexity import roster
from manabot.env import Match
from manabot.model.world import validate_agent_setup
from manabot.sim.flat_mc import load_checkpoint_agent
from manabot.training.learning_state import admit_learning_state
from manabot.training.models import IterationSchedule, TrainingRegime, TrainSelfPlay
from manabot.training.monitor_evaluation import MonitorProtocol, MonitorResult

TINY = Recipe(
    updates=4,
    switch=1,
    streams=4,
    device="cpu",
    schedule=IterationSchedule(steps=4, reference_steps=64),
    seconds_per_fit=1200,
)
SEED = 12271


@pytest.fixture(scope="module")
def fits(tmp_path_factory: pytest.TempPathFactory) -> dict[Arm, Fit]:
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("curriculum")
    digest = initialization_digest(TINY, SEED)
    return {
        arm: train_fit(arm, TINY, SEED, root / arm, digest)
        for arm in ("curriculum", "scratch")
    }


def _tensors(path: str) -> dict[str, torch.Tensor]:
    agent, _ = load_checkpoint_agent(path)
    return agent.state_dict()


def test_both_arms_share_initialization_and_continue_one_learning_state(
    fits: dict[Arm, Fit],
) -> None:
    initial = {
        arm: parameter_digest(
            load_checkpoint_agent(
                fit.source.stages[0].artifacts["initial_raw"]["path"]
            )[0]
        )
        for arm, fit in fits.items()
    }
    assert initial["curriculum"] == initial["scratch"]
    for arm, fit in fits.items():
        source, target = fit.source.stages[0], fit.target.stages[0]
        origin = target.learning_state_origin
        assert origin is not None and origin.iteration == TINY.switch
        assert [d["iteration"] for d in target.diagnostics] == [2, 3, 4]
        # The boundary export is exactly the source's final raw policy.
        exported = _tensors(target.artifacts["initial_raw"]["path"])
        for key, value in _tensors(source.artifacts["raw"]["path"]).items():
            torch.testing.assert_close(exported[key], value, rtol=0, atol=0)
        # ...rebound to L3, so ordinary L3 evaluation admits it.
        boundary, _ = load_checkpoint_agent(target.artifacts["initial_raw"]["path"])
        validate_agent_setup(boundary, Match(roster("l3").match()).to_rust())
        definition = fit.target.regime.stages[0]
        assert isinstance(definition, TrainSelfPlay)
        assert definition.learning_state is not None
        declared = definition.learning_state.source_match
        assert (declared is not None) == (arm == "curriculum")
    source_policy, _ = load_checkpoint_agent(
        fits["curriculum"].source.stages[0].artifacts["raw"]["path"]
    )
    with pytest.raises(ValueError, match="setup differs"):
        validate_agent_setup(source_policy, Match(roster("l3").match()).to_rust())


def test_checkpoints_cover_every_milestone_with_absolute_clocks(
    fits: dict[Arm, Fit],
) -> None:
    for fit in fits.values():
        found = fit_checkpoints(fit)
        assert {TINY.switch, TINY.updates // 2, TINY.updates} <= set(found)
        boundary = found[TINY.switch].coordinates
        assert boundary.updates == TINY.switch
        assert boundary.learner_transitions == TINY.switch * TINY.streams * 64
        assert found[TINY.updates].coordinates.learner_transitions == (
            TINY.updates * TINY.streams * 64
        )
    initial = initial_checkpoint(fits["scratch"]).coordinates
    assert initial.updates == 0 and initial.active_training_seconds == 0


def _admit(regime: TrainingRegime, out: Path) -> None:
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    out.mkdir(exist_ok=True)
    admit_learning_state(stage, regime, out)


def _target(fit: Fit, arm: Arm) -> TrainingRegime:
    return target_recipe(
        arm,
        TINY,
        fit.source,
        _reference(
            Path(fit.source.stages[0].artifacts["raw"]["path"]).parent / "run.json"
        ),
    )


def test_setup_transfer_admits_only_the_declared_deck_switch(
    fits: dict[Arm, Fit], tmp_path: Path
) -> None:
    curriculum = fits["curriculum"]
    _admit(_target(curriculum, "curriculum"), tmp_path / "ok")

    def with_source_match(**update: object) -> TrainingRegime:
        regime = _target(curriculum, "curriculum")
        stage = regime.stages[0]
        assert isinstance(stage, TrainSelfPlay) and stage.learning_state is not None
        assert stage.learning_state.source_match is not None
        stage.learning_state.source_match = (
            stage.learning_state.source_match.model_copy(update=update)
        )
        return regime

    # Undeclared: the ordinary unchanged-match rule still applies.
    with pytest.raises(ValueError, match="setup/schedule mismatch"):
        _admit(_target(curriculum, "scratch"), tmp_path / "undeclared")
    # Declared, but not what the parent trained on.
    l2 = roster("l2").match()
    with pytest.raises(ValueError, match="setup/schedule mismatch"):
        _admit(
            with_source_match(hero_deck=l2.hero_deck, villain_deck=l2.villain_deck),
            tmp_path / "other-decks",
        )
    with pytest.raises(ValueError, match="setup/schedule mismatch"):
        _admit(
            with_source_match(hero_sideboard={"Mountain": 1}), tmp_path / "sideboard"
        )
    with pytest.raises(ValueError, match="semantic pack"):
        _admit(
            with_source_match(content_pack="ur-lessons-vs-gw-allies"), tmp_path / "pack"
        )
    # A transfer must change decks; a scratch continuation cannot declare one.
    scratch = _target(fits["scratch"], "curriculum")
    with pytest.raises(ValueError, match="must change the match"):
        _admit(scratch, tmp_path / "same")
    # Architecture still binds the transfer.
    regime = _target(curriculum, "curriculum")
    regime.agent.hidden_dim = 32
    with pytest.raises(ValueError, match="setup/schedule mismatch"):
        _admit(TrainingRegime.model_validate(regime.model_dump()), tmp_path / "agent")


def _protocol(anchor: Anchor, milestone: int) -> MonitorProtocol:
    return MonitorProtocol(
        authored_matchup=roster("l3"),
        include_mirrors=True,
        deal_seeds=(1_227_990_000 + milestone,),
        opponent=anchor,
    )


def _score(fits: dict[Arm, Fit], directory: Path) -> int:
    return evaluate_seed(
        fits, (0, TINY.switch), _protocol, directory, lambda: False, anchors=("random",)
    )


@pytest.fixture(scope="module")
def scored(fits: dict[Arm, Fit], tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("cells")
    # Initialization once, then both arms and the control at the boundary.
    assert _score(fits, directory) == 1 + 3
    return directory


def _cell(directory: Path, name: str) -> MonitorResult:
    return MonitorResult.model_validate_json(
        (directory / name / "monitor.json").read_text()
    )


def test_scoring_runs_each_arm_and_control_once_and_keeps_failures(
    fits: dict[Arm, Fit], scored: Path, tmp_path: Path
) -> None:
    copy = tmp_path / "cells"
    shutil.copytree(scored, copy)
    assert _score(fits, copy) == 0
    (copy / f"{TINY.switch}-random-control/monitor.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete attempt"):
        _score(fits, copy)


def test_arms_pair_on_identical_deals_and_never_across_milestones(
    scored: Path,
) -> None:
    boundary = TINY.switch
    curriculum = _cell(scored, f"{boundary}-random-curriculum")
    scratch = _cell(scored, f"{boundary}-random-scratch")
    control = _cell(scored, f"{boundary}-random-control")
    assert curriculum.run_id != scratch.run_id
    between = paired_difference(curriculum, scratch, bootstrap_replicates=100)
    assert between.milestone == boundary and len(between.cells) == 4
    assert between.difference.mean == pytest.approx(
        np.mean([c.improvement.mean for c in between.cells])
    )
    gain = paired_difference(curriculum, control, bootstrap_replicates=100)
    assert curriculum.score is not None and control.score is not None
    assert gain.difference.mean == pytest.approx(
        curriculum.score.mean - control.score.mean
    )
    with pytest.raises(ValueError, match="differ: protocol"):
        paired_difference(curriculum, _cell(scored, "0-random-initial"))


def _plan(**update: object) -> CurriculumPlan:
    fields: dict[str, object] = {
        "total_seconds": 12 * 1740 + 600,
        "max_usd_per_hour": 0.69,
        "prior_usd": 4.85,
        "evaluation_usd": 10,
        "complexity_plan_sha256": read_complexity_plan(
            COMPLEXITY_PLAN.read_text()
        ).digest,
    }
    return CurriculumPlan.model_validate({**fields, **update})


def test_plan_freezes_cohort_budget_and_reuses_complexity_protocols() -> None:
    plan = _plan()
    assert plan.seed_order == SEEDS and plan.evaluation_updates == (0, 32, 64, 128)
    assert plan.recipe.schedule == TESTED_SCHEDULE
    complexity = plan.complexity()
    for milestone in (0, 64, 128):
        assert plan.protocol("heuristic", milestone) == complexity.protocol(
            "l3", "heuristic", milestone
        )
    families = {s for m in (0, 64, 128) for s in complexity.deals(m)}
    boundary = plan.protocol("scripted_greedy", 32)
    assert boundary.deal_seeds == BOUNDARY_DEALS and not families & set(BOUNDARY_DEALS)
    for arm in ("curriculum", "scratch"):
        source = source_recipe(arm, plan.recipe)
        stage = source.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        assert stage.updates == 32 and stage.streams == 64
        assert stage.execution.device == "cuda"
        assert stage.matchup_curriculum == "cross-balanced"
    with pytest.raises(ValueError, match="inclusive"):
        _plan(evaluation_usd=30)
    with pytest.raises(ValueError, match="twelve bounded fits"):
        _plan(total_seconds=3600)
    with pytest.raises(ValueError, match="fixed six seeds"):
        _plan(seed_order=SEEDS[:3])
    with pytest.raises(ValueError, match="complexity plan differs"):
        _plan(complexity_plan_sha256="0" * 64).complexity()


def test_scoring_refuses_local_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ETUDE_RUN", raising=False)
    with pytest.raises(ValueError, match="rented allocation"):
        evaluate_retained(tmp_path)
