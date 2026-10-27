"""Dashboard replay and tiny real training preserve metric and learning meaning."""

import json
from pathlib import Path

from etude.train.read import read
import pytest
import torch

from experiments.direct_self_play import direct_self_play
from manabot.experiment.evaluation_queue import checkpoints
from manabot.model.checkpoint import load_checkpoint
from manabot.reports import training as monitoring
from manabot.reports.training import (
    Dashboard,
    training_dashboard,
)
from manabot.training.execution import execute_regime
from manabot.training.records import StageRecord, TrainingRun
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.training.update_metrics import read_updates, stage_rows
from manabot.verify.store import VerifyStore
import managym


def _recipe() -> TrainingRegime:
    value = direct_self_play()
    value.world = managym.WORLD_VERSION
    value.id = f"test-{value.world}-{value.id}"
    value.stages = value.stages[:1]
    stage = value.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.updates = 2
    stage.transitions = 16
    stage.streams = 2
    stage.learning.epochs = 1
    value.agent.hidden_dim = 8
    value.agent.num_attention_heads = 2
    value.schedule_clock = "iteration_fraction"
    return value


def test_retained_rows_without_coordinates_stay_unavailable() -> None:
    run = TrainingRun(
        id="saved-fixture",
        regime=_recipe(),
        regime_digest="fixture",
        seed=197,
        seed_streams={},
        identities={},
        stages=[StageRecord(id="policy-0", diagnostics=[{"policy_loss": -0.4}])],
    )
    dashboard = training_dashboard(run, stage_rows(run, {}))
    assert dashboard.summary["history_incomplete"] is True
    assert "progress/training_seconds" not in dashboard.rows[0]
    assert dashboard.rows[0]["rl/policy_loss"] == -0.4
    assert dashboard.rows[0]["availability/rl/value_loss"] is False


def _history(directory: Path, run_id: str) -> list[dict[str, object]]:

    return [
        table.row(index)["values"]
        for table in read(directory / "etude", "manabot-monitoring", run_id).tables
        for index in range(table.rows)
        if "projection/row_sha256" in table.row(index)["values"]
    ]


def test_monitor_exports_do_not_change_learning_and_survive_reload(
    tmp_path: Path,
) -> None:
    recipe = _recipe()
    with VerifyStore(tmp_path / "runs.sqlite") as store:
        baseline = execute_regime(recipe, 197, tmp_path / "plain", store)
        monitored = execute_regime(
            recipe, 197, tmp_path / "monitored", store, checkpoint_seconds=1e-9
        )
        assert store.training_run(monitored.id) == monitored
    # The stage's raw export owns the endpoint; monitoring must not duplicate it.
    assert [c.updates for c in monitored.monitoring_checkpoints] == [1]
    evaluated = checkpoints(monitored)
    assert [c.coordinates.updates for c in evaluated] == [1, 2]
    assert evaluated[-1].artifact == monitored.selected_artifact
    for checkpoint in monitored.monitoring_checkpoints:
        assert checkpoint.error is None and checkpoint.artifact is not None
        load_checkpoint(checkpoint.artifact["path"])
    plain, _ = load_checkpoint(baseline.selected_artifact["path"])
    tracked, _ = load_checkpoint(monitored.selected_artifact["path"])
    for key, weights in plain.state_dict().items():
        torch.testing.assert_close(weights, tracked.state_dict()[key], rtol=0, atol=0)
    dashboard = training_dashboard(
        monitored, stage_rows(monitored, read_updates(tmp_path, [monitored]))
    )
    assert dashboard.summary["history_incomplete"] is False
    assert len(dashboard.rows) == 2
    assert (
        dashboard.rows[1]["progress/learner_transitions"]
        > dashboard.rows[0]["progress/learner_transitions"]
    )
    assert dashboard.rows[0]["availability/rl/policy_loss"] is True


def test_dashboard_offline_roundtrip() -> None:
    dashboard = Dashboard(
        run_id="offline-fixture",
        config={},
        summary={},
        rows=[
            {
                "progress/observation": 0,
                "rl/loss": 0.3,
                "rl/policy_loss": 0.4,
            }
        ],
    )
    reloaded = Dashboard.model_validate_json(dashboard.model_dump_json())
    assert reloaded == dashboard


def test_publisher_reads_etude_prefix_and_rejects_rewritten_history(
    tmp_path: Path,
) -> None:
    dashboard = Dashboard(
        run_id="fixture",
        config={},
        summary={},
        rows=[{"progress/observation": 0, "rl/entropy": 0.5}],
    )
    destination = tmp_path / "destination"
    monitoring.publish_dashboard(dashboard, tmp_path, destination=str(destination))
    monitoring.publish_dashboard(dashboard, tmp_path, destination=str(destination))
    assert len(_history(tmp_path, dashboard.run_id)) == 1
    dashboard.rows.append({"progress/observation": 1, "rl/entropy": 0.4})
    monitoring.publish_dashboard(dashboard, tmp_path, destination=str(destination))
    assert len(_history(tmp_path, dashboard.run_id)) == 2

    copied = read(destination, "manabot-monitoring", dashboard.run_id)
    assert list(copied.points("rl/entropy", "observation").y) == [0.5, 0.4]
    dashboard.rows[0]["rl/entropy"] = 0.6
    with pytest.raises(ValueError, match="published prefix"):
        monitoring.publish_dashboard(dashboard, tmp_path)


def test_publisher_waits_for_evaluation_completion_without_losing_local_progress(
    tmp_path: Path,
) -> None:
    dashboard = Dashboard(
        run_id="monitor-fixture",
        config={},
        summary={"attempted_games": 4},
        rows=[
            {
                "progress/observation": 0,
                "monitor/status": "running",
                "monitor/attempted_games": 4,
            }
        ],
    )
    monitoring.publish_dashboard(dashboard, tmp_path)
    dashboard.rows[0]["monitor/attempted_games"] = 8
    dashboard.summary["attempted_games"] = 8
    monitoring.publish_dashboard(dashboard, tmp_path)
    assert not _history(tmp_path, dashboard.run_id)
    assert dashboard.rows[0]["monitor/attempted_games"] == 8

    summaries = [
        table.row(index)["values"]["projection/summary_json"]
        for table in read(
            tmp_path / "etude", "manabot-monitoring", dashboard.run_id
        ).tables
        for index in range(table.rows)
    ]
    assert any(json.loads(summary)["attempted_games"] == 8 for summary in summaries)
    dashboard.rows[0]["monitor/status"] = "completed"
    monitoring.publish_dashboard(dashboard, tmp_path)
    dashboard.rows.append(
        {
            "progress/observation": 1,
            "monitor/status": "running",
            "monitor/attempted_games": 0,
        }
    )
    monitoring.publish_dashboard(dashboard, tmp_path)
    assert len(_history(tmp_path, dashboard.run_id)) == 1
    dashboard.rows[1]["monitor/status"] = "incomplete"
    monitoring.publish_dashboard(dashboard, tmp_path)
    assert sorted(
        row["monitor/status"] for row in _history(tmp_path, dashboard.run_id)
    ) == ["completed", "incomplete"]
    dashboard.rows[0]["monitor/attempted_games"] = 7
    with pytest.raises(ValueError, match="published prefix"):
        monitoring.publish_dashboard(dashboard, tmp_path)


def test_numerical_projection_distinguishes_unsampled_and_rejected_updates(
    tmp_path: Path,
) -> None:
    from manabot.reports.experiment import ReportEvidence, write_dashboard
    from manabot.training.records import NumericalFailure

    run = TrainingRun(
        id="numerical-fixture",
        regime=_recipe(),
        regime_digest="fixture",
        seed=1,
        seed_streams={},
        identities={},
        status="failed",
        stages=[
            StageRecord(
                id="policy-0",
                diagnostics=[
                    {
                        "numerical": {
                            "legal_logit_gap_max": 104.0,
                            "legal_underflow_count": 2,
                            "parameter_delta_l2": 0.01,
                            "sampled_steps": 1,
                        },
                        "coordinates": {"updates": 1},
                    },
                    {
                        "numerical": {
                            "legal_logit_gap_max": 105.0,
                            "legal_underflow_count": 3,
                            "sampled_steps": 0,
                        },
                        "coordinates": {"updates": 2},
                    },
                ],
                numerical_failure=NumericalFailure(
                    invariant="nonfinite_gradients",
                    iteration=3,
                    health={"rejected_steps": 1, "rejected_updates": 1},
                ),
            )
        ],
    )
    dashboard = training_dashboard(run, stage_rows(run, {}))
    assert len(dashboard.rows) == 3
    assert dashboard.rows[0]["numerical/parameter_delta_l2"] == 0.01
    assert "numerical/parameter_delta_l2" not in dashboard.rows[1]
    assert dashboard.rows[-1]["progress/updates"] == 2
    assert dashboard.rows[-1]["numerical/rejected_steps"] == 1
    assert dashboard.rows[-1]["failure/invariant"] == "nonfinite_gradients"
    evidence = ReportEvidence(tmp_path, (run,), (), (), (), ())
    path = write_dashboard(
        evidence,
        tmp_path / "health.html",
        question="Numerical controls",
        docs="metrics.md",
        sections=[],
    )
    html = path.read_text()
    assert "Numerical health" in html and "numerical-health.md" in html
    assert "Measured parameter movement" in html
    assert (
        "nonfinite_gradients" in path.with_name("health-diagnostics.json").read_text()
    )
