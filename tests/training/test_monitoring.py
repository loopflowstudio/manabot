"""Dashboard replay and tiny real training preserve metric and learning meaning."""

import json
from pathlib import Path

from etude.train.read import read
import pytest

from manabot.reports import training as monitoring
from manabot.reports.training import (
    Dashboard,
)


def _history(directory: Path, run_id: str) -> list[dict[str, object]]:

    return [
        table.row(index).values
        for table in read(directory / "etude", "manabot-monitoring", run_id).columns
        for index in range(table.rows)
        if "projection/row_sha256" in table.row(index).values
    ]


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
        table.row(index).values["projection/summary_json"]
        for table in read(
            tmp_path / "etude", "manabot-monitoring", dashboard.run_id
        ).columns
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
