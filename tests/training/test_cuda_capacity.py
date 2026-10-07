"""Capacity admission and evaluation schedule contracts; no rentals or training."""

from pathlib import Path

import pytest

import experiments.runners.cuda_capacity as capacity
from experiments.runners.cuda_capacity import CapacityPlan, Timing, protocols
from experiments.runners.training_protocol import EvaluationProtocol
from manabot.arena.models import canonical_sha256
from manabot.remote.deploy import Receipt
from manabot.remote.plan import Source
from manabot.training import checkpoint_queue as queue
from manabot.training.execution import atomic_json
from manabot.training.models import TrainingCoordinates, TrainSelfPlay
from manabot.training.monitor_evaluation import Checkpoint, MonitorProtocol
from manabot.training.presets import ataraxos_mtg_v1
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


def test_freeze_binds_real_authoring_and_rejects_control_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "calibration"
    (root / "inputs").mkdir(parents=True)
    base = ataraxos_mtg_v1().regime()
    base.agent.value_kind = "scalar"
    base.agent.value_aggregation = "value_token"
    base.wall_seconds = 1200
    for stage in base.stages:
        assert isinstance(stage, TrainSelfPlay)
        stage.execution.wall_seconds = 590
    (root / "inputs/cuda-calibration-small.json").write_text(base.model_dump_json())
    atomic_json(
        root / "calibration-amendment.json",
        dict(
            calibration_seconds=8700, comparison_seconds=41700, combined_seconds=50400
        ),
    )
    (root / "inputs/mix.json").write_bytes(
        Path("ops/mixes/runpod-small.json").read_bytes()
    )
    for name in ("small", "large", "l4-sweep", "a40-sweep"):
        directory = root / f"calibration-{name}"
        directory.mkdir()
        atomic_json(
            directory / "deployment.json",
            Receipt(
                plan_sha256="a" * 64,
                started=1000,
                deadline=2000,
                phase="deleted",
                estimated_dollars=0.1,
                complete=True,
            ).model_dump(mode="json"),
        )
    for name in ("w64-d2", "w384-d8"):
        directory = (
            root / f"calibration-l4-sweep/evidence/performance/{name}-loop-b512-s16/run"
        )
        directory.mkdir(parents=True)
        run = run_fixture(directory / "run.json")
        run.status = "completed"
        run.seconds = 3
        run.stages = run.stages[:1]
        run.stages[0].seconds = 3
        run.stages[0].diagnostics = [
            {"coordinates": {"training_seconds": i}} for i in (1, 2, 3)
        ]
        atomic_json(directory / "run.json", run.model_dump(mode="json"))
    (root / "live-workflow-control").mkdir()
    atomic_json(
        root / "live-workflow-control/receipt.json",
        dict(
            status="completed",
            seconds=100,
            evaluator_seconds=20,
            rental_estimated_dollars=0.1,
            observations=[dict(at=1010, run_status="running", updates=1, attempts=1)],
            purpose="fixture",
        ),
    )
    source = Source(commit="a" * 40, tree="b" * 40, lock_sha256="c" * 64)
    monkeypatch.setattr(capacity, "current_source", lambda root: source)
    monkeypatch.setattr(capacity.time, "time", lambda: 1001)
    plan = capacity.freeze(root, tmp_path / "science", "NVIDIA L4", 16, 512)
    assert len(plan.deployments) == 6
    assert len(set(p.regime.agent.hidden_dim for p in plan.deployments)) == 2
    assert [t.updates for t in plan.timings] == [600, 600]
    altered = plan.model_dump(mode="json")
    altered["deployments"][0]["regime"]["stages"][0]["learning"]["min_advantage"] = 0
    with pytest.raises(ValueError):
        CapacityPlan.model_validate(altered)


def test_four_hour_preparation_preserves_controls_and_shared_cost(
    tmp_path: Path,
) -> None:
    root = tmp_path / "calibration"
    (root / "inputs").mkdir(parents=True)
    base = ataraxos_mtg_v1().regime()
    base.agent.value_kind = "scalar"
    (root / "inputs/cuda-calibration-small.json").write_text(base.model_dump_json())
    (root / "inputs/mix.json").write_bytes(
        Path("ops/mixes/runpod-small.json").read_bytes()
    )
    source = Source(commit="a" * 40, tree="b" * 40, lock_sha256="c" * 64)
    plan = capacity.prepare_four_hour(
        root, tmp_path / "prepared", source, quote=0.49, quote_unix=1
    )
    assert plan.status == "prepared-not-admitted"
    assert len(plan.deployments) == 6
    assert plan.projected_dollars == pytest.approx(26.55597661771377)
    assert plan.monitoring.include_initial
    assert len(plan.monitoring.protocol.deal_seeds) * 4 == 100
    assert [p.opponent for p in plan.monitoring.terminal_protocols] == [
        "scripted_greedy",
        "random",
    ]
    controls: list[dict[str, object]] = []
    total = 0.0
    for deployment in plan.deployments:
        stage = deployment.regime.stages[0]
        assert isinstance(stage, TrainSelfPlay)
        assert stage.active_seconds == 14400
        total += stage.active_seconds
        payload = deployment.regime.model_dump()
        payload.pop("id")
        for field in ("hidden_dim", "attention_layers", "attention_feedforward_dim"):
            payload["agent"].pop(field, None)
        controls.append(payload)
    assert total == 24 * 3600 and all(c == controls[0] for c in controls)
    with pytest.raises(ValueError, match="at most"):
        capacity.prepare_four_hour(
            root, tmp_path / "expensive", source, quote=0.60, quote_unix=1
        )


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
