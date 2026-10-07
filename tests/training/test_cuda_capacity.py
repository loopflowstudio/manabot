"""Capacity admission and evaluation schedule contracts; no rentals or training."""

from pathlib import Path

import pytest

from experiments.runners.cuda_capacity import CapacityPlan, Timing, protocols
from experiments.runners.training_protocol import EvaluationProtocol
from manabot.arena.models import canonical_sha256
from manabot.remote.plan import Source
from manabot.training import checkpoint_queue as queue
from manabot.training.models import TrainingCoordinates
from manabot.training.monitor_evaluation import Checkpoint, MonitorProtocol
from tests.training.test_checkpoint_queue import (
    Process,
    run_fixture,
)

pytest_plugins = ["tests.training.test_checkpoint_queue"]


def protocol() -> EvaluationProtocol:
    return EvaluationProtocol(
        study="cuda-capacity",
        purpose="scientific",
        regime_digests=("a" * 64, "b" * 64),
        training_seeds=(10351, 10352, 10353),
        paired_deals=(),
        anchor_deals=tuple(range(1910103510, 1910103535)),
        endpoint_anchor_deals=tuple(range(1910103610, 1910103635)),
        random_diagnostic_deals=tuple(range(1910103710, 1910103735)),
        anchors=("scripted-greedy", "random"),
        checkpoint_count=3,
        cost_cutoffs_seconds=(300, 600),
        early_progress_seconds=600,
        progress_score=0.5,
        process_seconds=41700,
        uncertainty="paired-seed-descriptive",
    )


def test_reserved_cohorts_and_original_protocol_rules() -> None:
    p = protocol()
    with pytest.raises(ValueError, match="disjoint"):
        EvaluationProtocol.model_validate(
            p.model_dump() | {"random_diagnostic_deals": p.anchor_deals}
        )
    with pytest.raises(ValueError):
        EvaluationProtocol.model_validate(
            p.model_dump() | {"regime_digests": ("a" * 64,)}
        )
    with pytest.raises(ValueError, match="three seeds"):
        EvaluationProtocol.model_validate(p.model_dump() | {"training_seeds": (10351,)})
    with pytest.raises(ValueError, match="frozen study"):
        MonitorProtocol(purpose="frozen-study-evaluation")
    with pytest.raises(ValueError, match="frozen study"):
        MonitorProtocol(study_protocol_sha256="a" * 64)


def test_initial_midpoint_endpoint_and_random_stay_distinct(tmp_path: Path) -> None:
    run = run_fixture(tmp_path / "run.json")
    run.stages[0].id = "policy-0"
    run.stages[1].id = "policy-1"
    plan = CapacityPlan.model_construct(
        source=Source(commit="a" * 40, tree="b" * 40, lock_sha256="c" * 64),
        calibration_root=str(tmp_path),
        calibration_receipts={},
        timings=[
            Timing(
                capacity="w64-d2",
                path="receipt",
                sha256="d" * 64,
                seconds_per_update=1,
                updates=20,
            )
        ],
        experiment_receipt_sha256="e" * 64,
        protocol=protocol(),
        deployments=[],
        created_unix=1,
    )

    def checkpoint(stage: str, updates: int) -> Checkpoint:
        return Checkpoint(
            artifact={"path": "raw.pt", "sha256": "f" * 64, "bytes": 1},
            coordinates=TrainingCoordinates(
                stage_id=stage, updates=updates, training_seconds=updates
            ),
        )

    first = protocols(plan, run, checkpoint("policy-0", 0))
    mid = protocols(plan, run, checkpoint("policy-0", 10))
    assert first == mid and first[0].deal_seeds == plan.protocol.anchor_deals
    assert protocols(plan, run, checkpoint("policy-1", 20)) == []
    run.stages[1].status = "completed"
    final = protocols(plan, run, checkpoint("policy-1", 20))
    assert [p.opponent for p in final] == ["scripted_greedy", "random"]
    assert final[0].deal_seeds == plan.protocol.endpoint_anchor_deals
    assert final[0].study_protocol_sha256 == canonical_sha256(
        plan.protocol.model_dump(mode="json")
    )
    assert final[1].deal_seeds == plan.protocol.random_diagnostic_deals
    assert final[1].purpose == "monitoring-not-scientific-evaluation"


def test_queue_binds_two_protocols_without_reusing_the_cohort(
    tmp_path: Path, fake_process: list[Process]
) -> None:
    source = tmp_path / "run.json"
    run_fixture(source)
    variants = [
        MonitorProtocol(
            deal_seeds=(1910103610,),
            purpose="frozen-study-evaluation",
            study_protocol_sha256="a" * 64,
        ),
        MonitorProtocol(deal_seeds=(1910103710,), opponent="random"),
    ]
    monitor = queue.CheckpointQueue(
        tmp_path / "monitor",
        queue.MonitoringBudget(seconds=10, attempt_seconds=5),
        lease=tmp_path / "lease",
        protocols_for=lambda r, c: variants,
    )
    try:
        monitor.tick([source, source])
        assert len(monitor.attempts) == 1 and monitor.pending == 1
        job = queue.EvaluationJob.model_validate_json(
            next((tmp_path / "monitor").glob("attempt-*/job.json")).read_text()
        )
        assert job.protocol == variants[0]
        fake_process[0].code = 1
        monitor.tick([source])
        assert len(monitor.attempts) == 2 and monitor.pending == 0
        assert monitor.attempts[0].identity != monitor.attempts[1].identity
    finally:
        monitor.close()


def test_declarative_terminal_schedule_stays_off_monitoring(tmp_path: Path) -> None:
    run = run_fixture(tmp_path / "run.json")
    config = queue.MonitoringBudget(
        seconds=100,
        attempt_seconds=10,
        include_initial=True,
        protocol=MonitorProtocol(deal_seeds=(1,)),
        terminal_protocols=(
            MonitorProtocol(
                deal_seeds=(2,),
                purpose="frozen-study-evaluation",
                study_protocol_sha256="a" * 64,
            ),
            MonitorProtocol(deal_seeds=(3,), opponent="random"),
        ),
    )
    first = queue.checkpoints(run)[0]
    assert config.protocols_for(run, first) == [config.protocol]
    last = run.stages[-1]
    last.status = "completed"
    last.cumulative_seconds = 2
    last.artifacts["raw"] = {"path": "final.pt", "sha256": "b" * 64, "bytes": 1}
    final = queue.checkpoints(run)[-1]
    assert config.protocols_for(run, final) == list(config.terminal_protocols)
    with pytest.raises(ValueError, match="disjoint"):
        queue.MonitoringBudget.model_validate(
            config.model_dump() | {"terminal_protocols": [config.protocol.model_dump()]}
        )
