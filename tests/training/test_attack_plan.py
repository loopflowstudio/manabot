"""Attack plans reject leaked evaluation seeds and unaffordable full cohorts."""

import pytest

from manabot.infra.hypers import MatchHypers
from manabot.training.attacks import AttackPlan, AttackTarget
from manabot.training.models import FrozenOpponent, TrainingRegime, TrainSelfPlay
import managym


def _plan() -> AttackPlan:
    return AttackPlan(
        id="attack-proof",
        targets=(
            AttackTarget(
                id="target-a",
                policy=FrozenOpponent(path="target.pt", sha256="a" * 64),
                producer_seeds=(17,),
            ),
        ),
        attacker_seeds=(101, 201),
        cumulative_updates=(2, 5, 9),
        final_deal_seeds=(950001,),
        template=TrainingRegime(
            id="attacker",
            world=managym.WORLD_VERSION,
            match=MatchHypers(),
            wall_seconds=30,
            stages=[TrainSelfPlay(id="policy", operation="train_self_play")],
        ),
        total_seconds=90,
        evaluation_seconds=30,
        prediction="Attacker strength increases with declared training compute.",
    )


def test_attack_ladder_continues_one_optimizer_against_identical_target() -> None:
    plan = _plan()
    regime = plan.regime_for(plan.targets[0])
    stages = regime.stages
    assert all(isinstance(stage, TrainSelfPlay) for stage in stages)
    assert [stage.updates for stage in stages] == [2, 3, 4]
    assert [stage.initial for stage in stages] == [None, "attack-0", "attack-1"]
    assert all(stage.opponent == plan.targets[0].policy for stage in stages)


@pytest.mark.parametrize("seed", [17, 10017, 101, 10101, 20201, 30201])
def test_attack_plan_rejects_training_namespace_in_final_deals(seed: int) -> None:
    payload = _plan().model_dump()
    payload["final_deal_seeds"] = [seed]
    with pytest.raises(ValueError, match="overlap"):
        AttackPlan.model_validate(payload)


def test_attack_plan_charges_every_seed_before_start() -> None:
    payload = _plan().model_dump()
    payload["total_seconds"] = 89
    with pytest.raises(ValueError, match="full attack cohort"):
        AttackPlan.model_validate(payload)
