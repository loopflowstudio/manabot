"""Declarative authoring preserves configurations and exposes intentional changes."""

from dataclasses import replace
import json
from pathlib import Path
from typing import NoReturn

from pydantic import JsonValue, ValidationError
import pytest

from experiments.runners.model_capacity import experiment as capacity_experiment
from experiments.runners.run_value_models import (
    experiment as value_experiment,
    smoke_baseline,
    smoke_plan,
)
from manabot.arena.models import canonical_sha256
from manabot.infra.hypers import AgentSpec, MatchHypers, ObservationSpaceHypers
from manabot.training.experiments import (
    Axis,
    Baseline,
    Case,
    Environment,
    Experiment,
    LearningRule,
    Model,
    Pipeline,
    Resources,
    RunControl,
    SettingProvenance,
)
from manabot.training.models import (
    AtaraxosMoveLearning,
    CollectSearch,
    Execution,
    Learning,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.presets import ataraxos_mtg_v1


def _baseline() -> Baseline:
    return Baseline.capture("control-v1", smoke_baseline())


def _flatten(value: JsonValue, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if isinstance(value, dict) and value:
        return set().union(*(_flatten(v, (*prefix, k)) for k, v in value.items()))
    if isinstance(value, list) and value:
        return set().union(
            *(_flatten(v, (*prefix, str(i))) for i, v in enumerate(value))
        )
    return {prefix}


def test_provenance_covers_complete_configuration_and_explicit_equal_values() -> None:
    base = ataraxos_mtg_v1()
    resolved = (
        Experiment(
            "audit",
            base,
            overrides=(
                Model(
                    AgentSpec(
                        hidden_dim=64, semantic_pack=None, compound_decisions=False
                    )
                ),
                Resources(execution=Execution(threads=2), wall_seconds=200),
                LearningRule(
                    AtaraxosMoveLearning(gradient="ataraxos_move", policy_lambda=0.3)
                ),
                Environment(
                    observation=ObservationSpaceHypers(max_actions=128),
                    match=MatchHypers(hero="new-hero"),
                ),
                RunControl(
                    schedule_clock="iteration_fraction", recovery_max_microsteps=None
                ),
            ),
        )
        .resolve()
        .cases[0]
    )
    settings = {p.path: p for p in resolved.provenance}
    effective = resolved.regime.model_dump(mode="json")
    effective["agent"]["compound_decisions"] = False
    assert set(settings) == _flatten(effective)
    assert len(settings) == len(resolved.provenance)
    assert settings[("agent", "hidden_dim")].origin == "override"
    assert settings[("agent", "hidden_dim")].value == 64
    assert settings[("agent", "semantic_pack")].value is None
    assert settings[("agent", "compound_decisions")].origin == "override"
    assert settings[("agent", "value_kind")].origin == "preset"
    assert (
        settings[("stages", "0", "learning", "policy_lambda")].component == "learning"
    )
    assert settings[("stages", "1", "execution", "threads")].component == "resources"
    assert settings[("match", "hero")].origin == "override"
    assert settings[("match", "villain")].origin == "preset"
    assert settings[("id",)].origin == "identity"
    assert settings[("recovery_max_microsteps",)].value is None
    assert {p.component for p in settings.values()} == {
        "environment",
        "model",
        "learning",
        "resources",
        "run",
        "pipeline",
        "identity",
    }
    assert base.regime().agent.semantic_pack == "ur-lessons-vs-gw-allies"


def test_baseline_and_resolved_artifacts_are_isolated() -> None:
    regime = smoke_baseline()
    base = Baseline.capture("immutable-v1", regime)
    before = base.digest
    regime.match.hero_deck.clear()
    base.regime().agent.hidden_dim = 128
    cells = value_experiment().resolve()
    identity = cells.identity
    cells.regimes[cells.cases[0].name].match.hero_deck.clear()
    receipt = cells.receipt()
    receipt.clear()
    assert base.digest == before
    assert base.regime().match.hero_deck
    assert cells.identity == identity
    assert cells.cases[1].regime.match.hero_deck
    empty = SettingProvenance(("empty",), "model", "baseline", "test", "{}")
    value = empty.value
    assert isinstance(value, dict)
    value["mutated"] = True
    assert empty.value == {}


def test_snapshot_rejects_missing_defaults_and_preset_is_pinned() -> None:
    base = ataraxos_mtg_v1()
    assert (
        base.digest
        == "43e7ece151480504a795f3d355c792080d8088215151bf9fe24d64e6c22166cb"
    )
    raw = json.loads(base.configuration)
    del raw["agent"]["attention_layers"]
    with pytest.raises(ValueError, match="incomplete"):
        replace(base, configuration=json.dumps(raw)).regime()
    assert ataraxos_mtg_v1().regime().agent.value_kind == "categorical_wdl"


@pytest.mark.parametrize(
    "overrides",
    [
        (Model(AgentSpec(hidden_dim=64)), Model(AgentSpec(hidden_dim=64))),
        (Resources(wall_seconds=100), Resources(wall_seconds=200)),
        (
            Pipeline(tuple(smoke_baseline().stages)),
            Resources(execution=Execution(threads=2)),
        ),
        (Pipeline(tuple(smoke_baseline().stages)), LearningRule(Learning())),
    ],
)
def test_conflicts_rejected_in_either_order(
    overrides: tuple[Model | Resources | Pipeline | LearningRule, ...],
) -> None:
    for ordered in (overrides, tuple(reversed(overrides))):
        with pytest.raises(ValueError, match="conflicting writes"):
            Experiment("conflict", _baseline(), overrides=ordered).resolve()


def test_case_axis_and_common_overrides_cannot_shadow_each_other() -> None:
    change = (Model(AgentSpec(value_kind="scalar")),)
    declaration = Experiment(
        "conflict",
        _baseline(),
        overrides=change,
        matrix=(Axis("output", (Case("scalar", change),)),),
    )
    with pytest.raises(ValueError, match="experiment.*axis:output"):
        declaration.resolve()
    with pytest.raises(ValueError, match="case:scalar.*axis:output"):
        replace(declaration, overrides=(), cases=(Case("scalar", change),)).resolve()


def test_disjoint_writes_are_order_independent_and_provenance_is_identity() -> None:
    overrides = (
        Model(AgentSpec(value_aggregation="value_token")),
        Resources(wall_seconds=120),
    )
    declaration = Experiment("independent", _baseline(), overrides=overrides)
    assert (
        declaration.resolve()
        == replace(declaration, overrides=tuple(reversed(overrides))).resolve()
    )
    implicit = Experiment("same", _baseline()).resolve().cases[0]
    explicit = (
        Experiment("same", _baseline(), overrides=(Model(AgentSpec(hidden_dim=64)),))
        .resolve()
        .cases[0]
    )
    assert explicit.digest == implicit.digest
    assert explicit.identity != implicit.identity
    assert explicit.digest == canonical_sha256(explicit.regime.model_dump(mode="json"))


def test_matrix_order_labels_and_deterministic_resolution() -> None:
    declaration = Experiment(
        "cross",
        _baseline(),
        cases=(Case("a"), Case("b")),
        matrix=(
            Axis(
                "pool",
                (
                    Case("mean", label="Mean"),
                    Case(
                        "token",
                        (Model(AgentSpec(value_aggregation="value_token")),),
                        "Token",
                    ),
                ),
            ),
            Axis(
                "output",
                (
                    Case("scalar"),
                    Case(
                        "wdl", (Model(AgentSpec(value_kind="categorical_wdl")),), "WDL"
                    ),
                ),
            ),
        ),
    )
    result = declaration.resolve()
    assert [c.name for c in result.cases] == [
        f"cross-{case}-{pool}-{output}"
        for case in ("a", "b")
        for pool in ("mean", "token")
        for output in ("scalar", "wdl")
    ]
    assert result.cases[-1].label == "b / Token / WDL"
    assert result.receipt() == declaration.resolve().receipt()
    assert result.identity == declaration.resolve().identity


@pytest.mark.parametrize(
    "changes",
    [
        {"cases": (Case("same"), Case("same"))},
        {"cases": (Case("a", label="Same"), Case("b", label="Same"))},
        {"matrix": (Axis("empty", ()),)},
        {"matrix": (Axis("x", (Case("a"),)), Axis("x", (Case("b"),)))},
        {"cases": (Case("bad_name"),)},
        {
            "cases": (Case("a-b"), Case("a")),
            "matrix": (Axis("x", (Case("c"), Case("b-c"))),),
        },
    ],
)
def test_ambiguous_or_empty_names_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(Experiment("bad", _baseline()), **changes).resolve()


def test_compatibility_is_checked_after_independent_components_compose() -> None:
    no_attention = Model(AgentSpec(attention_on=False))
    token = Model(AgentSpec(value_aggregation="value_token"))
    with pytest.raises(ValidationError, match="require attention"):
        Experiment("invalid", _baseline(), overrides=(no_attention, token)).resolve()
    with pytest.raises(ValidationError, match="requires ataraxos_move"):
        Experiment(
            "invalid", ataraxos_mtg_v1(), overrides=(LearningRule(Learning()),)
        ).resolve()
    with pytest.raises(ValidationError, match="compound"):
        Experiment(
            "invalid",
            _baseline(),
            overrides=(Model(AgentSpec(compound_decisions=True)),),
        ).resolve()
    valid = Experiment(
        "ppo",
        ataraxos_mtg_v1(),
        overrides=(LearningRule(Learning()), Model(AgentSpec(value_kind="scalar"))),
    ).resolve()
    assert valid.cases[0].regime.agent.value_kind == "scalar"


def test_pipeline_replacement_and_learning_boundary() -> None:
    result = (
        Experiment(
            "teacher",
            _baseline(),
            overrides=(
                Pipeline(
                    (CollectSearch(operation="collect_search", id="labels", games=2),)
                ),
            ),
        )
        .resolve()
        .cases[0]
    )
    assert len(result.regime.stages) == 1
    assert result.regime.stages[0].operation == "collect_search"
    assert all(
        p.origin == "override" for p in result.provenance if p.path[0] == "stages"
    )
    with pytest.raises(ValueError, match="only self-play"):
        Experiment(
            "bad",
            Baseline.capture("teacher-v1", result.regime),
            overrides=(LearningRule(Learning()),),
        ).resolve()


def test_resolution_allocates_no_model_training_or_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("resolution attempted execution")

    monkeypatch.setattr("manabot.training.execution.execute_regime", forbidden)
    monkeypatch.setattr("experiments.runners.run_value_models.run_study", forbidden)
    monkeypatch.setattr("manabot.model.agent.Agent.__init__", forbidden)
    monkeypatch.setattr("manabot.verify.store.VerifyStore.__init__", forbidden)
    monkeypatch.setattr("managym.Env", forbidden)
    monkeypatch.chdir(tmp_path)
    result = value_experiment().resolve()
    plan = smoke_plan(result)
    assert plan.protocol.regime_digests == result.digests
    assert len(plan.recipes) == 8
    assert list(tmp_path.iterdir()) == []
    assert all("seed" not in p.path for c in result.cases for p in c.provenance)


def test_capacity_migration_preserves_registered_identities() -> None:
    result = capacity_experiment(smoke_baseline()).resolve()
    assert result.digests == (
        "3aa8175c21095df6620fdb3485ca55b0eb2259d58a1bcc4705d769e8a5b4ce35",
        "01050b55349236b68c96bb7ecdb5dd4a60d20f9b904192057546ca1f786ed4a1",
        "b3d98735210e95988a941ae01547ee54ae093433d8d61310af91d335036f3a80",
    )
    for case in result.cases:
        assert TrainingRegime.model_validate_json(case.configuration) == case.regime
        assert all(isinstance(s, TrainSelfPlay) for s in case.regime.stages)


@pytest.mark.parametrize(
    "component",
    [
        Environment(world="unsupported-world"),
        Model(AgentSpec(belief_count_buckets=4)),
    ],
)
def test_resolution_uses_existing_runtime_admission(
    component: Environment | Model,
) -> None:
    with pytest.raises(ValueError, match="world|belief inputs"):
        Experiment("unsupported", _baseline(), overrides=(component,)).resolve()


def test_explicit_false_default_remains_in_provenance() -> None:
    # An explicitly written default must survive AgentSpec's serialization omission.
    base = _baseline()
    cell = (
        Experiment(
            "flat", base, overrides=(Model(AgentSpec(compound_decisions=False)),)
        )
        .resolve()
        .cases[0]
    )
    field = next(
        p for p in cell.provenance if p.path == ("agent", "compound_decisions")
    )
    assert field.origin == "override" and field.value is False
    assert not cell.regime.agent.compound_decisions
