"""Plan-only checks: contrasts change declared settings and cannot execute work."""

from pathlib import Path
from typing import NoReturn, get_args

from pydantic import ValidationError
import pytest

from experiments.runners.run_training_regimes import STUDIES, smoke_recipe
from experiments.runners.technique_screen import (
    ContrastName,
    experiment,
    export_plan,
    smoke_plan,
)
from experiments.runners.training_protocol import ResolvedStudy
from manabot.training.models import (
    AtaraxosMoveLearning,
    Execution,
    TrainCompound,
    TrainSelfPlay,
)

# This expected scientific intervention is intentionally independent of the
# generator: accidental extra changes invalidate the interpretation of a pair.
RULE_DELTAS: dict[str, set[str]] = {
    "policy-trace": {"policy_lambda"},
    "value-trace": {"value_lambda"},
    "reference": {"reference"},
    "collection-kl": {"collection_kl"},
    "ratio-clip": {"clip"},
    "gradient-clip": {"max_grad_norm"},
    "filter-quantile": {"advantage_quantile"},
    "filter-minimum": {"min_advantage"},
    "filter-off": {"advantage_quantile", "min_advantage"},
    "filter-scope": {"filter_scope"},
    "filter-ties": {"filter_kind"},
    "lr-constant": {"learning_rate_scale", "learning_rate_power"},
    "tau-constant": {"tau_power"},
    "schedules-constant": {
        "learning_rate_scale",
        "learning_rate_power",
        "tau_power",
    },
    "discount": {"gamma"},
}


@pytest.mark.parametrize("name", get_args(ContrastName))
def test_contrast_admission_and_exact_intervention(name: ContrastName) -> None:
    cells = experiment(name).resolve()
    plan = smoke_plan(name, cells)
    assert ResolvedStudy.model_validate_json(plan.model_dump_json()) == plan
    assert plan.protocol.purpose == "workflow-smoke"
    assert plan.protocol.training_seeds == (1051,)
    assert plan.protocol.process_seconds == 900
    assert all(r["wall_seconds"] <= 180 for r in plan.recipes)
    if name == "evaluation-ema":
        assert len(cells.cases) == 1
        assert plan.protocol.evaluation_variants == ("raw", "ema")
        return
    control, treatment = [c.regime for c in cells.cases]
    assert control.agent == treatment.agent
    assert control.match == treatment.match
    assert control.observation == treatment.observation
    if name == "ppo-package":
        assert control.agent.value_kind == "scalar"
        assert control.stages[0].learning.gradient == "ataraxos_move"
        assert treatment.stages[0].learning.gradient == "ppo"
    for left, right in zip(control.stages, treatment.stages, strict=True):
        assert isinstance(left, TrainSelfPlay) and isinstance(right, TrainSelfPlay)
        if name == "behavior-ema":
            assert left.behavior == "current-self" and right.behavior == "ema-self"
            right.behavior = left.behavior
        else:
            if name in RULE_DELTAS:
                a, b = left.learning.model_dump(), right.learning.model_dump()
                changed = {key for key in a if a[key] != b[key]}
                assert changed == RULE_DELTAS[name]
            right.learning = left.learning
        assert left == right
    # All non-stage settings, including workload, resource and run controls agree.
    control.id = treatment.id
    control.stages = treatment.stages
    assert control == treatment


def test_rate_contrast_has_equal_start_and_real_late_difference() -> None:
    base, held = [c.regime.stages[0] for c in experiment("lr-constant").resolve().cases]
    assert isinstance(base, TrainSelfPlay) and isinstance(held, TrainSelfPlay)
    assert isinstance(base.learning, AtaraxosMoveLearning)
    assert isinstance(held.learning, AtaraxosMoveLearning)
    assert base.learning.rates(1) == held.learning.rates(1)
    assert base.learning.rates(2) == held.learning.rates(2)
    assert base.learning.rates(10000)[0] < held.learning.rates(10000)[0]
    assert base.learning.rates(10000)[1] == held.learning.rates(10000)[1]


def test_plan_export_cannot_train_or_replace_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("planning attempted execution")

    monkeypatch.setattr("manabot.training.execution.execute_regime", forbidden)
    monkeypatch.setattr("manabot.model.agent.Agent.__init__", forbidden)
    monkeypatch.setattr("manabot.verify.store.VerifyStore.__init__", forbidden)
    monkeypatch.setattr("managym.Env", forbidden)
    for name in get_args(ContrastName):
        out = tmp_path / name
        export_plan(name, out)
        before = (out / "plan.json").read_bytes()
        with pytest.raises(FileExistsError):
            export_plan(name, out)
        assert (out / "plan.json").read_bytes() == before
        assert "unexecuted" in (out / "admission.md").read_text()


def test_mislabeled_or_unsupported_plans_fail_closed() -> None:
    with pytest.raises(ValueError, match="does not match"):
        smoke_plan("policy-trace", experiment("value-trace").resolve())
    with pytest.raises(ValidationError):
        AtaraxosMoveLearning.model_validate(
            {"gradient": "ataraxos_move", "gamma": 0.99}
        )
    with pytest.raises(ValidationError):
        Execution.model_validate({"precision": "bfloat16"})
    plan = smoke_plan("reference", experiment("reference").resolve()).model_dump()
    plan["protocol"]["purpose"] = "scientific"
    with pytest.raises(ValidationError, match="scientific"):
        ResolvedStudy.model_validate(plan)


def test_compound_factorial_reuses_delivered_credit_and_estimator_contract() -> None:
    recipes = [smoke_recipe(name) for name in STUDIES["compound-decisions"]]
    pairs: set[tuple[str, str]] = set()
    for recipe in recipes:
        assert recipe.agent.compound_decisions
        assert recipe.agent == recipes[0].agent
        assert recipe.match == recipes[0].match
        for stage in recipe.stages:
            assert isinstance(stage, TrainCompound)
            assert stage.learning.gamma == 1
            assert stage.learning.ema is None
            pairs.add((stage.grouping, stage.estimator))
        # Credit unit and estimator are the only non-identity differences.
        normalized = recipe.model_dump()
        reference = recipes[0].model_dump()
        normalized["id"] = reference["id"]
        for stage, control in zip(
            normalized["stages"], reference["stages"], strict=True
        ):
            stage["grouping"], stage["estimator"] = (
                control["grouping"],
                control["estimator"],
            )
        assert normalized == reference
    assert pairs == {
        (grouping, estimator)
        for grouping in ("sequential", "grouped")
        for estimator in ("outcome", "bootstrapped")
    }
