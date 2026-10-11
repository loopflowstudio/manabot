"""Monitoring retains failed attempts and resamples complete deal blocks."""

import json
from pathlib import Path
from typing import Any, Literal

import pytest
import torch

from manabot.arena.match import selected_match
from manabot.arena.models import ArenaKey, PlayerRegistration, canonical_sha256
from manabot.env import Match, ObservationSpace
from manabot.infra.hypers import AgentSpec
from manabot.model.agent import Agent
from manabot.sim.distill import save_bc_checkpoint
from manabot.sim.teacher1_evidence import runtime_fingerprints
from manabot.training import monitor_evaluation as monitor
from manabot.training.models import TrainingRegime, TrainingRun, TrainSelfPlay
from manabot.training.monitoring import default_panels, evaluation_dashboard


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
    key = ArenaKey(
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


def test_block_failure_retains_prior_rows_and_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    monkeypatch.setattr(
        monitor, "_manifest", lambda *args: manifest.model_copy(deep=True)
    )

    def play(
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        if kwargs["deal_seeds"] == (102,):
            raise TimeoutError("retained block failure")
        return _rows(manifest, (101,)), {"sha256": "trace"}, {"passed": True}

    monkeypatch.setattr(monitor, "play_cell", play)
    # The injected manifest isolates arena orchestration from checkpoint loading.
    result = monitor.evaluate_checkpoint(
        TrainingRun.model_construct(),
        manifest.artifact,
        manifest.coordinates,
        tmp_path / "attempt",
    )
    assert result.status == "incomplete"
    assert len(result.rows) == 4
    assert result.error == "TimeoutError: retained block failure"
    assert result.evaluation_seconds is not None and result.evaluation_seconds > 0
    assert result.score is None
    assert result.rows[0].trace_path.startswith("deal-000/")
    saved = monitor.MonitorResult.model_validate_json(
        (tmp_path / "attempt/monitor.json").read_text()
    )
    assert saved == result
    with pytest.raises(FileExistsError):
        monitor.evaluate_checkpoint(
            TrainingRun.model_construct(),
            manifest.artifact,
            manifest.coordinates,
            tmp_path / "attempt",
        )


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


@pytest.mark.parametrize(
    "max_commands,opponent",
    [(1, "scripted_greedy"), (10000, "scripted_greedy"), (10000, "random")],
)
def test_real_arena_checkpoint_outcomes(
    tmp_path: Path, max_commands: int, opponent: Literal["scripted_greedy", "random"]
) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        match = selected_match()
        space = ObservationSpace()
        hypers = AgentSpec(
            hidden_dim=8, num_attention_heads=2, semantic_pack="ur-lessons-vs-gw-allies"
        )
        torch.manual_seed(101)
        checkpoint = tmp_path / "untrained.pt"
        save_bc_checkpoint(
            Agent(space, hypers),
            space,
            checkpoint,
            player_configs=Match(match).to_rust(),
        )
        regime = TrainingRegime(
            id="monitor-fixture",
            world="w4",
            match=match,
            agent=hypers,
            wall_seconds=10,
            stages=[
                TrainSelfPlay(
                    id="unused",
                    operation="train_self_play",
                    updates=1,
                    transitions=8,
                    streams=2,
                )
            ],
        )
        run = TrainingRun(
            id="fixture",
            regime_digest=canonical_sha256(regime.model_dump(mode="json")),
            regime=regime,
            seed=17,
            seed_streams={},
            identities=runtime_fingerprints(
                17, match_hypers=match, observation_space=space
            ),
        )
        result = monitor.evaluate_checkpoint(
            run,
            {
                "path": str(checkpoint),
                "sha256": monitor.file_sha256(checkpoint),
                "bytes": checkpoint.stat().st_size,
            },
            monitor.TrainingCoordinates(
                stage_id="unused", updates=0, training_seconds=0
            ),
            tmp_path / "arena",
            protocol=monitor.MonitorProtocol(
                deal_seeds=(1910101000,),
                max_commands=max_commands,
                game_seconds=30,
                opponent=opponent,
                comparison_sha256="a" * 64 if opponent == "random" else None,
            ),
        )
        assert (
            result.started_unix is not None
            and result.finished_unix >= result.started_unix
        )
        assert result.opponent.player_spec["kind"] == opponent
        assert result.purpose == (
            "predeclared-comparison-not-admission"
            if opponent == "random"
            else "monitoring-not-scientific-evaluation"
        )
        assert len(result.rows) == 4
        if max_commands == 1:
            assert result.status == "incomplete"
            assert all(row.failure is not None for row in result.rows)
            assert result.score is None
        else:
            assert result.status == "completed", result.model_dump_json()
            assert all(row.valid for row in result.rows)
            assert result.score is not None
        assert all(
            (tmp_path / "arena" / row.trace_path).is_file() for row in result.rows
        )
    finally:
        torch.set_num_threads(previous_threads)


def test_checkpoint_admission_failure_is_saved(tmp_path: Path) -> None:
    manifest = _manifest()
    with pytest.raises(FileNotFoundError):
        monitor.evaluate_checkpoint(
            TrainingRun.model_construct(id="failed-admission"),
            manifest.artifact,
            manifest.coordinates,
            tmp_path / "attempt",
        )
    failure = json.loads((tmp_path / "attempt/admission-failure.json").read_text())
    assert failure["run_id"] == "failed-admission"
    assert failure["artifact"] == manifest.artifact
    assert failure["error"].startswith("FileNotFoundError:")


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
    assert len(default_panels(dashboard)) == 6
    changed = complete.model_copy(deep=True)
    changed.protocol.deal_seeds = (200,)
    with pytest.raises(ValueError, match="same run"):
        evaluation_dashboard([complete, changed])
