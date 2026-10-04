"""Complete-game estimator semantics and retained real execution evidence."""

import json
from pathlib import Path

import pytest

from experiments.runners.selection_diagnostic import diagnostic_recipe
from manabot.arena.models import canonical_sha256, file_sha256
from manabot.training.analysis import report_selection_run
from manabot.training.execution import execute_regime
from manabot.training.models import (
    AtaraxosMoveLearning,
    CollectSelection,
    TrainingRegime,
    TrainSelfPlay,
)
from manabot.training.selection_analysis import analyze_selection
from manabot.training.selection_data import (
    SelectionDataset,
    SelectionDecision,
    SelectionGame,
)
from manabot.verify.store import VerifyStore


def _dataset(stage: CollectSelection) -> SelectionDataset:
    return SelectionDataset(
        run_id="fixture",
        stage_id="selection",
        policy_run_id="fixture",
        policy_stage_id="policy-0",
        policy_sha256="a" * 64,
        weights="raw",
        world_binding_sha256="b" * 64,
        population_sha256=canonical_sha256(
            [s.model_dump(mode="json") for s in stage.population]
        ),
        games=tuple(
            SelectionGame(
                spec=spec,
                winner=0,
                terminal_digest="terminal",
                rows=tuple(
                    SelectionDecision(
                        step=i,
                        actor=i % 2,
                        observation_identity=str(i),
                        action=0,
                        action_type=i % 2,
                        value=0.2 if i < 2 else 0.4,
                        action_probability=0.5,
                        entropy=0.69,
                        reference_kl=0,
                    )
                    for i in range(4)
                ),
            )
            for spec in stage.population
        ),
    )


def test_outcome_residual_is_separate_from_lambda_target() -> None:
    stage = diagnostic_recipe().stages[-1]
    assert isinstance(stage, CollectSelection)
    stage.learning.gamma = 0.5
    stage.learning.policy_lambda = 0
    stage.learning.value_lambda = 0
    stage.learning.retained_fraction = 0.5
    report = analyze_selection(_dataset(stage), stage)
    first = report.rows[0]
    assert first.terminal_distance == 1
    assert first.advantage == pytest.approx(0)  # .5 * .4 - .2
    assert first.lambda_residual == pytest.approx(0)
    assert first.terminal_residual == pytest.approx(0.8)
    assert first.discounted_terminal_residual == pytest.approx(0.3)
    assert [r.terminal_distance for r in report.rows[:4]] == [1, 0, 1, 0]
    assert report.rows[2].terminal_residual == pytest.approx(-1.2)
    assert all(g["terminal_distance"] != "censored" for g in report.selection_groups)
    assert report.optimizer_exposures == 0
    assert all(g.terminal_abs_difference_ci95 is None for g in report.associations)
    # Whole-game membership survives analysis; no row or seat is a split unit.
    assert all(r.split == stage.population[r.game].split for r in report.rows)
    stage.learning = AtaraxosMoveLearning(
        gradient="ataraxos_move", policy_lambda=1, value_lambda=1
    )
    report = analyze_selection(_dataset(stage), stage)
    for row in report.rows:
        assert row.lambda_residual == pytest.approx(row.terminal_residual)
        assert row.advantage == pytest.approx(row.terminal_residual)


def test_uncertainty_clusters_whole_games_and_keeps_empty_groups() -> None:
    stage = diagnostic_recipe().stages[-1]
    assert isinstance(stage, CollectSelection)
    stage.learning.filter_kind = "top_count"
    stage.learning.retained_fraction = 0.25
    stage.learning.policy_lambda = 0
    dataset = _dataset(stage)
    dataset = dataset.model_copy(
        update={
            "games": tuple(
                g.model_copy(
                    update={
                        "rows": tuple(
                            r.model_copy(update={"action_type": 0}) for r in g.rows
                        )
                    }
                )
                for g in dataset.games
            )
        }
    )
    report = analyze_selection(dataset, stage)
    terminal = next(
        g
        for g in report.associations
        if g.split == "held_out" and g.terminal_distance == "terminal"
    )
    assert terminal.retained.games == terminal.excluded.games == 2
    assert terminal.retained.rows == terminal.excluded.rows == 2
    assert terminal.terminal_abs_difference == pytest.approx(0.8)
    assert terminal.terminal_abs_difference_ci95 == pytest.approx([0.8, 0.8])
    assert terminal.bootstrap_defined == stage.bootstrap_samples
    stage.learning.min_advantage = 100
    empty = analyze_selection(dataset, stage)
    assert all(g.retained.rows == 0 for g in empty.associations)
    assert all(g.terminal_abs_difference is None for g in empty.associations)
    assert all(g.terminal_abs_difference_ci95 is None for g in empty.associations)


def test_population_rejects_duplicate_deals_and_missing_assignments() -> None:
    recipe = diagnostic_recipe()
    raw = recipe.model_dump()
    population = raw["stages"][-1]["population"]
    population = list(population)
    population[1]["seed"] = population[0]["seed"]
    raw["stages"][-1]["population"] = population
    with pytest.raises(ValueError, match="unique deal"):
        TrainingRegime.model_validate(raw)
    raw = recipe.model_dump()
    raw["stages"][-1]["population"][0]["split"] = "held_out"
    with pytest.raises(ValueError, match="both deck assignments"):
        TrainingRegime.model_validate(raw)


@pytest.mark.parametrize("categorical", [False, True])
def test_frozen_complete_games_and_offline_report(
    tmp_path: Path, categorical: bool
) -> None:
    recipe = diagnostic_recipe()
    training = recipe.stages[0]
    stage = recipe.stages[-1]
    assert isinstance(training, TrainSelfPlay)
    assert isinstance(stage, CollectSelection)
    if categorical:
        recipe.agent.value_kind = "categorical_wdl"
        training.learning = AtaraxosMoveLearning(gradient="ataraxos_move")
        stage.learning = training.learning.model_copy(deep=True)
        stage.weights = "ema"
    with VerifyStore(tmp_path / "training.sqlite") as store:
        run = execute_regime(recipe, 693, tmp_path / "run", store)
        assert store.training_run(run.id).status == "completed"
        # A new diagnostic can reuse the admitted Run without retraining.
        reuse = diagnostic_recipe(recipe, source_run=run.id)
        reused_stage = reuse.stages[0]
        assert isinstance(reused_stage, CollectSelection)
        reused_stage.weights = stage.weights
        reused_stage.learning = stage.learning.model_copy(deep=True)
        repeated = execute_regime(reuse, 693, tmp_path / "repeat", store)
    record = run.stages[-1]
    assert record.games == 4
    assert record.optimizer_exposures == 0
    assert record.collection_seconds > 0 and record.diagnostic_seconds > 0
    assert (
        record.inputs["policy"]["sha256"]
        == run.stages[0].artifacts[stage.weights]["sha256"]
    )
    assert (
        file_sha256(record.inputs["policy"]["path"])
        == record.inputs["policy"]["sha256"]
    )
    dataset = SelectionDataset.model_validate_json(
        Path(record.artifacts["dataset"]["path"]).read_text()
    )
    repeat = SelectionDataset.model_validate_json(
        Path(repeated.stages[0].artifacts["dataset"]["path"]).read_text()
    )
    assert dataset.games == repeat.games
    assert dataset.policy_sha256 == repeat.policy_sha256
    assert all(len(g.rows) > 1 for g in dataset.games)
    assert {r.actor for g in dataset.games for r in g.rows} == {0, 1}
    report_selection_run(
        tmp_path / "run/run.json", "selection", tmp_path / "regenerated"
    )
    assert (tmp_path / "regenerated/analysis.json").read_bytes() == Path(
        record.artifacts["analysis"]["path"]
    ).read_bytes()
    assert (tmp_path / "regenerated/report.md").read_bytes() == Path(
        record.artifacts["report"]["path"]
    ).read_bytes()
    Path(record.artifacts["game-0"]["path"]).write_text("corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        report_selection_run(tmp_path / "run/run.json", "selection", tmp_path / "bad")


def test_capped_game_retains_attempt_cost_and_partial_receipt(tmp_path: Path) -> None:
    recipe = diagnostic_recipe()
    stage = recipe.stages[-1]
    assert isinstance(stage, CollectSelection)
    stage.max_steps = 1
    with VerifyStore(tmp_path / "training.sqlite") as store:
        with pytest.raises(RuntimeError, match="step cap"):
            execute_regime(recipe, 693, tmp_path / "failed", store)
        exported = json.loads((tmp_path / "failed/run.json").read_text())
        run = store.training_run(exported["id"])
    assert run.status == "failed"
    record = run.stages[-1]
    assert record.games == 0 and record.collection_seconds > 0
    assert len(record.rejected_artifacts) == 1
    assert "population" in record.artifacts
    assert "dataset" not in record.artifacts
    partial = Path(next(iter(record.rejected_artifacts.values()))["path"]).read_text()
    assert '"prediction"' in partial and '"winner"' not in partial
    with pytest.raises(ValueError, match="completed"):
        report_selection_run(
            tmp_path / "failed/run.json", "selection", tmp_path / "bad"
        )


def test_changed_source_policy_is_retained_as_failed_attempt(tmp_path: Path) -> None:
    recipe = diagnostic_recipe()
    recipe.stages = recipe.stages[:1]
    with VerifyStore(tmp_path / "training.sqlite") as store:
        source = execute_regime(recipe, 693, tmp_path / "source", store)
        mismatch = diagnostic_recipe(recipe, source_run=source.id)
        mismatch.agent.hidden_dim = 16
        with pytest.raises(ValueError, match="configuration differs"):
            execute_regime(mismatch, 693, tmp_path / "mismatch", store)
        frozen = Path(source.stages[0].artifacts["raw"]["path"])
        frozen.write_bytes(b"changed checkpoint")
        followup = diagnostic_recipe(recipe, source_run=source.id)
        with pytest.raises(ValueError, match="artifact changed"):
            execute_regime(followup, 693, tmp_path / "failed", store)
        exported = json.loads((tmp_path / "failed/run.json").read_text())
        failed = store.training_run(exported["id"])
        assert failed.status == "failed" and failed.seconds > 0
        assert failed.stages[0].inputs["policy"] == source.stages[0].artifacts["raw"]
        assert not failed.stages[0].artifacts
