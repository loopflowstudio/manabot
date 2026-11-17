"""Monitoring retains failed attempts and resamples complete deal blocks."""

import json
from pathlib import Path
from typing import Any, Literal

import pytest

from manabot.eval import monitor as monitor
from manabot.eval.models import EvaluationKey
from manabot.infra.identity import canonical_sha256
from manabot.reports.training import evaluation_dashboard
from manabot.sim.registration import PlayerRegistration


def failure_summary(result: monitor.MonitorResult) -> str:
    """Name why a cohort is incomplete; its full record exceeds pytest's limit."""
    lines = [
        f"{result.status} with {len(result.rows)}/{result.expected_games} games: "
        f"{result.error}"
    ]
    lines += [
        f"row {i}: {row.termination_reason} after {row.decisions} decisions "
        f"in {row.game_seconds:.1f}s: {row.failure}"
        for i, row in enumerate(result.rows)
        if row.failure is not None
    ]
    return "\n".join(lines)


def _manifest() -> monitor.MonitorResult:
    common = dict(
        world="w4",
        content_suite="w4-allies-lessons-v1",
        information_boundary="acting-viewer",
        observation_abi_sha256="a" * 64,
        action_abi_sha256="b" * 64,
        matchup_sha256="c" * 64,
        player_seed_derivation_id="arena-pair-deal-player-v1",
    )
    candidate = PlayerRegistration(
        **common,
        player_id="candidate",
        display_name="candidate",
        role="challenger",
        runner_kind="checkpoint",
        player_spec={
            "kind": "checkpoint",
            "deterministic": False,
            "device": "cpu",
            "batch_size": 1,
        },
        compute_class_id="cpu",
        checkpoint_sha256="d" * 64,
        checkpoint_bytes=1,
        parameter_count=1,
        training_seed=7,
        artifact_id="fixed-checkpoint",
    )
    opponent = PlayerRegistration(
        **common,
        player_id="scripted-greedy",
        display_name="scripted",
        role="anchor",
        runner_kind="code",
        player_spec={"kind": "scripted_greedy"},
        compute_class_id="cpu",
        source_sha256="e" * 64,
    )
    key = EvaluationKey(
        world="w4",
        content_suite="w4-allies-lessons-v1",
        viewer_boundary="acting-viewer",
        arena_version="training-monitor-v1",
        rating_model_version="unrated",
        rating_prior_sha256=canonical_sha256({}),
        anchor_cohort_sha256=canonical_sha256(opponent.model_dump()),
        evaluation_compute_envelope_id="cpu",
    )
    return monitor.MonitorResult(
        run_id="original-run",
        regime_digest="f" * 64,
        training_seed=7,
        artifact={"path": "absent.pt", "sha256": "d" * 64, "bytes": 1},
        coordinates=monitor.TrainingCoordinates(
            stage_id="stage", updates=50, training_seconds=3600
        ),
        protocol=monitor.MonitorProtocol(deal_seeds=(101, 102)),
        key=key,
        candidate=candidate,
        opponent=opponent,
        expected_games=8,
    )


def _rows(
    result: monitor.MonitorResult, seeds: tuple[int, ...] = (101, 102)
) -> list[dict[str, Any]]:
    return [
        dict(
            arena_key=result.key.model_dump(),
            deal_seed=seed,
            leg=leg,
            player_a=result.candidate.player_id,
            player_b=result.opponent.player_id,
            player_a_registration_sha256=result.candidate.identity_sha256,
            player_b_registration_sha256=result.opponent.identity_sha256,
            score_a=float(seed == 102),
            failure=None,
            terminated=True,
            truncated=False,
            replay_passed=True,
            trace_path="traces/commands.jsonl.gz",
            game_seconds=0.01,
            integrity={"illegal_actions": 0},
            game_trace_sha256="retained-original",
        )
        for seed in seeds
        for leg in range(4)
    ]


def test_deal_cluster_bootstrap_and_original_coordinates(tmp_path: Path) -> None:
    manifest = _manifest()
    source = tmp_path / "rows.json"
    source.write_text(json.dumps(_rows(manifest)))
    result = monitor.import_saved_rows(manifest, source, tmp_path / "import")
    assert result.status == "completed"
    assert result.score is not None
    assert result.score.mean == 0.5
    # Eight independent game draws would produce narrower intervals; four legs
    # share a deal and must all move together.
    assert (result.score.lower, result.score.upper) == (0, 1)
    assert result.coordinates == manifest.coordinates
    assert result.evaluation_seconds is None
    assert result.rows[0].model_extra == {"game_trace_sha256": "retained-original"}
    assert result.source_rows_sha256 == monitor.file_sha256(source)


@pytest.mark.parametrize("defect", ["missing", "timeout", "replay", "integrity"])
def test_incomplete_cohort_never_emits_strength(tmp_path: Path, defect: str) -> None:
    manifest = _manifest()
    rows = _rows(manifest)
    if defect == "missing":
        rows.pop()
    elif defect == "timeout":
        rows[0].update(failure="timeout", terminated=False, score_a=0.0)
    elif defect == "replay":
        rows[0]["replay_passed"] = False
    else:
        rows[0]["integrity"] = {"private_exposures": 1}
    source = tmp_path / "rows.json"
    source.write_text(json.dumps(rows))
    result = monitor.import_saved_rows(manifest, source, tmp_path / "import")
    assert result.status == "incomplete"
    assert result.win is result.draw is result.score is None
    assert len(result.rows) == len(rows)


@pytest.mark.parametrize("defect", ["duplicate", "identity", "foreign-deal", "score"])
def test_saved_rows_admission(tmp_path: Path, defect: str) -> None:
    manifest = _manifest()
    rows = _rows(manifest)
    if defect == "duplicate":
        rows.append(rows[0])
    elif defect == "identity":
        rows[0]["player_a_registration_sha256"] = "0" * 64
    elif defect == "score":
        rows[0]["score_a"] = 0.3
    else:
        rows[0]["deal_seed"] = 999
    source = tmp_path / "rows.json"
    source.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        monitor.import_saved_rows(manifest, source, tmp_path / "import")


def test_default_monitoring_cohort() -> None:
    protocol = monitor.MonitorProtocol()
    assert len(protocol.deal_seeds) * 4 == 100
    with pytest.raises(ValueError):
        monitor.MonitorProtocol(deal_seeds=(1, 1))


@pytest.mark.parametrize("binding", ["monitoring", "study", "comparison"])
def test_result_purpose_preserves_protocol_binding(
    binding: Literal["monitoring", "study", "comparison"],
) -> None:
    protocol = (
        monitor.MonitorProtocol(
            purpose="frozen-study-evaluation", study_protocol_sha256="a" * 64
        )
        if binding == "study"
        else monitor.MonitorProtocol(
            comparison_sha256="a" * 64 if binding == "comparison" else None
        )
    )
    expected = {
        "monitoring": "monitoring-not-scientific-evaluation",
        "study": "frozen-study-evaluation",
        "comparison": "predeclared-comparison-not-admission",
    }[binding]
    data = _manifest().model_dump(mode="json")
    data.update(protocol=protocol.model_dump(mode="json"), purpose=expected)
    result = monitor.MonitorResult.model_validate(data)
    assert monitor.MonitorResult.model_validate_json(result.model_dump_json()) == result
    data["purpose"] = (
        "frozen-study-evaluation"
        if binding == "monitoring"
        else "monitoring-not-scientific-evaluation"
    )
    with pytest.raises(ValueError, match="purpose differs"):
        monitor.MonitorResult.model_validate(data)


def test_saved_monitor_dashboard_preserves_intervals_and_failed_attempts(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    source = tmp_path / "rows.json"
    source.write_text(json.dumps(_rows(manifest)))
    complete = monitor.import_saved_rows(manifest, source, tmp_path / "complete")
    failed = manifest.model_copy(update={"status": "incomplete", "error": "timeout"})
    dashboard = evaluation_dashboard([complete, failed])
    assert dashboard.rows[0]["monitor/score/mean"] == 0.5
    assert dashboard.rows[0]["monitor/score/lower"] == 0
    assert dashboard.rows[1]["availability/monitor_rates"] is False
    assert "monitor/score/mean" not in dashboard.rows[1]
    changed = complete.model_copy(deep=True)
    changed.protocol.deal_seeds = (200,)
    with pytest.raises(ValueError, match="same run"):
        evaluation_dashboard([complete, changed])
