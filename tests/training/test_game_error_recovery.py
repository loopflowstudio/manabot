"""Repaired CPU continuation under the existing provider and evaluation owners.

Only provider effects, evaluator processes and placement are fake. Real training
snapshots, allocation receipts and exported checkpoints connect the owners; no
paid execution or portable CUDA recovery is exercised here.
"""

from collections.abc import Mapping
from contextlib import closing
import inspect
import json
from pathlib import Path
import time

import pytest
import torch

from manabot.arena.models import canonical_sha256, file_sha256
from manabot.env import ObservationSpace
from manabot.experiment import evaluation_queue as checkpoint_queue
from manabot.infra.config import ModelConfig
from manabot.remote import plan as remote_plan
from manabot.remote.cohort import CohortEntry
from manabot.remote.jobs import JobPhase, JobRecord, Resource
from manabot.remote.plan import Allocation, JobSpec
from manabot.remote.provider import ProviderError
from manabot.training import execution
from manabot.training.collection import SeatRoutedCollector
from manabot.training.game_errors import (
    GameAttempt,
    GameErrorLedger,
    GameErrorPolicy,
    GameErrorState,
    replay_attempt,
)
from manabot.training.models import (
    Learning,
    RecoveryPolicy,
    TrainingRegime,
    TrainingRun,
    TrainSelfPlay,
)
from manabot.verify.store import VerifyStore
import managym
from tests.remote.test_cohort import Harness
from tests.remote.test_compile import SOURCE
from tests.training.game_error_fixtures import (
    PaymentMismatchCollector,
    PaymentMismatchEnv,
    _bounded_fixture_replay,
    _payment_match,
    _repaired_payment,
)
from tests.training.test_checkpoint_queue import Process


@pytest.mark.parametrize("provider_phase", ["completed", "failed"])
def test_repaired_restart_preserves_post_snapshot_failures_and_costs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_phase: JobPhase,
) -> None:
    torch.set_num_threads(1)
    recipe = TrainingRegime(
        id="game-recovery",
        world=managym.WORLD_VERSION,
        match=_payment_match(),
        agent=ModelConfig(attention_on=False, hidden_dim=8),
        schedule_clock="iteration_fraction",
        wall_seconds=120,
        recovery=RecoveryPolicy(checkpoint_updates=1),
        stages=[
            TrainSelfPlay(
                id="train",
                operation="train_self_play",
                updates=3,
                streams=2,
                transitions=1,
                game_errors=GameErrorPolicy(),
                learning=Learning(epochs=1, minibatches=1, retained_fraction=1),
            )
        ],
    )
    remote = Harness(tmp_path, monkeypatch)
    remote.clock.now = time.time()
    machine_spec = remote.cohort.entries[0].plan.spec

    def local_placement(raw: str, spec: JobSpec) -> TrainingRegime:
        # Fake provider runs this exact CPU recipe. Production RunPod placement
        # requires CUDA and still rejects process recovery; do not bypass that
        # admission outside this fixture or claim portable GPU recovery.
        assert spec == machine_spec
        return TrainingRegime.model_validate_json(raw)

    monkeypatch.setattr(remote_plan, "resolve", local_placement)
    plan = remote_plan.compile_plan(recipe.model_dump_json(), machine_spec, SOURCE, 197)
    monitoring = checkpoint_queue.MonitoringBudget(seconds=10, attempt_seconds=5)
    remote.cohort = remote.cohort.model_copy(
        update={
            "deadline": remote.clock.now + 3600,
            "entries": (
                CohortEntry(
                    job_id="repaired-training",
                    plan=plan,
                    monitoring=monitoring,
                    checkpoint_updates=1,
                ),
            ),
        }
    )
    admitted = remote.restart().tick()
    spec = admitted.attempts[0].spec
    allocation = spec.allocation
    assert allocation is not None and spec.plan.regime == recipe
    reserved = admitted.charged_dollars(remote.cohort)
    assert reserved == remote.cohort.reserved_dollars
    remote.provider.failure = "ambiguous"
    with pytest.raises(ProviderError, match="create timed out"):
        remote.restart().tick()
    remote.provider.failure = ""
    with pytest.raises(TimeoutError, match="acceptance"):
        remote.restart().tick()
    assert remote.provider.created == 1
    job_store = remote.jobs(spec)
    frozen_remote = {
        key: job_store.read(key)
        for key in ("spec.json", "training-claim.json", "training.json")
    }
    resource_bytes = frozen_remote["training.json"]
    assert resource_bytes is not None
    resource = Resource.model_validate_json(resource_bytes.data)
    assert resource.claim.deadline == allocation.deadline
    running = JobRecord(
        spec_sha256=spec.identity,
        pod_id=resource.pod.id,
        phase="running",
        accepted_at=remote.clock.now,
        heartbeat_at=remote.clock.now,
    )
    assert job_store.create("runtime/record.json", running.model_dump_json().encode())
    assert remote.restart().tick().charged_dollars(remote.cohort) == reserved
    original_fingerprints = execution._runtime_identities

    def fixture_identities(
        seed: int, regime: TrainingRegime, space: ObservationSpace
    ) -> dict[str, object]:
        identities = original_fingerprints(seed, regime, space)
        return {
            **identities,
            "fixture_source_sha256": canonical_sha256(
                {
                    "driver": file_sha256(Path(__file__)),
                    "payment": file_sha256(Path(inspect.getfile(PaymentMismatchEnv))),
                }
            ),
            "training_source_sha256": canonical_sha256(
                {
                    "training": identities["training_source_sha256"],
                    "fixture_payment_source": inspect.getsource(
                        PaymentMismatchEnv.payment
                    ),
                }
            ),
        }

    monkeypatch.setattr(execution, "_runtime_identities", fixture_identities)
    record = GameErrorLedger.record

    def interrupt_after_second_crash(
        self: GameErrorLedger, row: GameAttempt, context: Mapping[str, object]
    ) -> GameErrorState:
        record(self, row, context)
        if self.state.crashed == 2:
            raise KeyboardInterrupt("after durable failure, before next update")
        return self.state

    with VerifyStore(tmp_path / "training.sqlite") as store:

        def run(
            name: str,
            *,
            resume_from: str | None = None,
            source_repair_reason: str | None = None,
            admitted_allocation: Allocation = allocation,
        ) -> TrainingRun:
            return execution.execute_regime(
                spec.plan.regime,
                spec.plan.seed,
                tmp_path / name,
                store,
                checkpoint_updates=spec.checkpoint_updates,
                allocation=admitted_allocation,
                resume_from=resume_from,
                source_repair_reason=source_repair_reason,
            )

        with monkeypatch.context() as patch:
            patch.setattr(execution, "SeatRoutedCollector", PaymentMismatchCollector)
            patch.setattr(GameErrorLedger, "record", interrupt_after_second_crash)
            with pytest.raises(KeyboardInterrupt):
                run("first")
        parent_id = store.con.execute(
            "SELECT id FROM training_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
        parent = store.training_run(parent_id)
        assert 1 <= len(parent.stages[0].diagnostics) < 3
        assert parent.game_errors is not None and parent.game_errors.crashed == 2
        original = Path(parent.game_error_ledgers[0]).read_bytes()
        entries = [json.loads(line) for line in original.splitlines()]
        line_index, failed_entry = next(
            (index, entry)
            for index, entry in reversed(list(enumerate(entries, 1)))
            if entry["attempt"]["outcome"] == "engine_error"
        )
        retained = GameAttempt.model_validate(failed_entry["attempt"])
        assert retained.frame_json is not None
        offer = json.loads(retained.frame_json)["offers"][retained.actions[-1]]
        assert offer["verb"] == "cast"
        assert offer["public_commitment"]["card"] == "Gray Ogre"
        assert retained.discarded_samples > 0
        with monkeypatch.context() as patch:
            patch.setattr(managym, "Env", PaymentMismatchEnv)
            replay = replay_attempt(retained)
        assert replay["status"] == "failed" and replay["at_retained_failure"]
        assert replay["error"] == retained.error
        assert replay["frame"] == retained.frame_json
        with monkeypatch.context() as patch:
            patch.setattr(
                PaymentMismatchEnv, "payment", staticmethod(_repaired_payment)
            )
            patch.setattr(managym, "Env", PaymentMismatchEnv)
            assert replay_attempt(retained)["status"] == "advanced"
        bounded = _bounded_fixture_replay(
            "manabot.training.game_errors",
            Path(parent.game_error_ledgers[0]),
            "--attempt",
            line_index,
        )
        assert bounded["at_retained_failure"] and bounded["error"] == retained.error
        assert bounded["matches_binding"] and bounded["matches_error"]
        allowance = parent.game_errors.model_dump()
        repaired = _bounded_fixture_replay(
            "manabot.training.game_errors",
            Path(parent.game_error_ledgers[0]),
            "--attempt",
            line_index,
            repaired=True,
        )
        assert repaired["status"] == "advanced"
        assert repaired["matches_binding"] and repaired["at_retained_failure"]
        assert repaired["matches_error"] is None
        assert repaired["original_outcome"] == "engine_error"
        assert Path(parent.game_error_ledgers[0]).read_bytes() == original
        assert parent.game_errors.model_dump() == allowance
        assert failed_entry["context"]["coordinates"]["updates"] == len(
            parent.stages[0].diagnostics
        )
        assert parent.recovery_artifact is not None
        frozen_snapshot = Path(parent.recovery_artifact["path"]).read_bytes()
        processes: list[Process] = []

        def evaluator(*args: object, **kwargs: object) -> Process:
            process = Process()
            processes.append(process)
            return process

        # Only the worker is fake; the queue persists real checkpoint bindings,
        # reservation, deadline and attempt receipts from this interrupted run.
        monkeypatch.setattr(checkpoint_queue.subprocess, "Popen", evaluator)
        monkeypatch.setattr(checkpoint_queue, "stop_process_group", lambda p: p.wait())
        monitor_out, lease = tmp_path / "monitor", tmp_path / "evaluator.lock"

        def restart_monitor() -> checkpoint_queue.CheckpointQueue:
            return checkpoint_queue.CheckpointQueue(
                monitor_out, monitoring, clock=remote.clock.time, lease=lease
            )

        monitor = restart_monitor()
        parent_source = tmp_path / "first/run.json"
        try:
            monitor.tick([parent_source])
            assert len(processes) == 1 and monitor.active is not None
            interrupted_identity = monitor.active.identity
            interrupted_job = monitor_out / f"attempt-{interrupted_identity}/job.json"
            frozen_evaluation = interrupted_job.read_bytes()
        finally:
            # Simulate loss of the owner and its fake worker without finalizing
            # the running receipt. The next owner must charge the full reserve.
            monitor.lock.close()
        assert remote.restart().tick().charged_dollars(remote.cohort) == reserved
        monkeypatch.setattr(
            PaymentMismatchEnv, "payment", staticmethod(_repaired_payment)
        )
        assert (
            fixture_identities(197, recipe, ObservationSpace(recipe.observation))[
                "training_source_sha256"
            ]
            != parent.identities["training_source_sha256"]
        )
        with pytest.raises(ValueError, match="identity mismatch"):
            run("unapproved", resume_from=parent.id)

        with monkeypatch.context() as patch:
            patch.setattr(
                execution,
                "_runtime_identities",
                lambda *args: {**fixture_identities(*args), "world": "incompatible"},
            )
            with pytest.raises(ValueError, match="identity mismatch"):
                run(
                    "wrong-world",
                    resume_from=parent.id,
                    source_repair_reason="A source repair cannot admit another world",
                )

        def unavailable(*args: object, **kwargs: object) -> SeatRoutedCollector:
            raise RuntimeError("repair attempt failed during collector setup")

        with monkeypatch.context() as patch:
            patch.setattr(execution, "SeatRoutedCollector", unavailable)
            with pytest.raises(RuntimeError, match="collector setup"):
                run(
                    "setup-failed",
                    resume_from=parent.id,
                    source_repair_reason="Deterministic payment fixture repaired; unchanged world and inputs",
                )
        failed = store.training_run(
            store.con.execute(
                "SELECT id FROM training_runs ORDER BY rowid DESC LIMIT 1"
            ).fetchone()[0]
        )
        assert failed.recovery_artifact == parent.recovery_artifact
        with pytest.raises(ValueError, match="original allocation deadline"):
            run(
                "renewed",
                admitted_allocation=allocation.model_copy(
                    update={"admitted_at": allocation.admitted_at + 60}
                ),
                resume_from=failed.id,
            )
        monkeypatch.setattr(execution, "SeatRoutedCollector", PaymentMismatchCollector)
        child = run("second", resume_from=failed.id)
        assert child.status == "completed"
        assert child.game_errors is not None and child.game_errors.crashed == 2
        assert child.game_errors.attempted > parent.game_errors.attempted
        assert len(child.stages[0].diagnostics) == 3
        assert child.prior_seconds == failed.prior_seconds + failed.seconds
        assert (
            child.prior_watchdog_seconds
            == failed.prior_watchdog_seconds + failed.watchdog_seconds
        )
        assert child.regime.wall_seconds == parent.regime.wall_seconds
        assert child.parent_run_id == failed.id and child.source_repair_reason
        assert Path(parent.game_error_ledgers[0]).read_bytes() == original
        assert (
            child.stages[0].diagnostics[: len(parent.stages[0].diagnostics)]
            == parent.stages[0].diagnostics
        )
        assert Path(parent.recovery_artifact["path"]).read_bytes() == frozen_snapshot
        assert child.allocation_deadline == parent.allocation_deadline
        assert child.allocation_deadline == allocation.deadline
        assert child.optimizer_updates_through() == 3
        exported = TrainingRun.model_validate_json(
            (tmp_path / "second/run.json").read_text()
        )
        assert exported.status == "completed"
        assert exported.game_errors == child.game_errors
        assert exported.game_errors.attempted == exported.game_errors.completed + 2
        assert exported.game_errors.discarded_samples > 0
        assert exported.game_errors.replacement_samples > 0
        assert store.training_run(parent.id) == parent
        assert child.monitoring_checkpoints[: len(parent.monitoring_checkpoints)] == (
            parent.monitoring_checkpoints
        )
        with pytest.raises(ValueError, match="already has a recovery child"):
            run("duplicate-recovery", resume_from=failed.id)

        sources = [
            parent_source,
            tmp_path / "setup-failed/run.json",
            tmp_path / "second/run.json",
        ]
        with closing(restart_monitor()) as monitor:
            assert monitor.attempts[0].identity == interrupted_identity
            assert monitor.attempts[0].status == "interrupted"
            assert monitor.charged_seconds == monitoring.attempt_seconds
            monitor.tick(sources + sources)
            assert len(processes) == len(monitor.attempts) == 2
            assert monitor.active is not None
            assert monitor.active.identity != interrupted_identity
            job = checkpoint_queue.EvaluationJob.model_validate_json(
                (
                    monitor_out / f"attempt-{monitor.active.identity}/job.json"
                ).read_text()
            )
            assert job.checkpoint.coordinates.updates == 2
            assert job.checkpoint.artifact == child.monitoring_checkpoints[1].artifact
            remote.clock.now += monitoring.attempt_seconds
            processes[-1].code = 1
            monitor.tick(sources)
            assert monitor.charged_seconds == monitoring.seconds
            assert monitor.process is None and monitor.pending == 1
            assert [a.status for a in monitor.attempts] == ["interrupted", "failed"]
        with closing(restart_monitor()) as monitor:
            monitor.tick(sources)
            assert len(processes) == len(monitor.attempts) == 2
            assert monitor.charged_seconds == monitoring.seconds
            assert monitor.pending == 1 and monitor.process is None
            assert interrupted_job.read_bytes() == frozen_evaluation

        state = remote.restart().tick()
        assert len(state.attempts) == remote.provider.created == 1
        assert state.attempts[0].spec == spec
        assert state.charged_dollars(remote.cohort) == reserved
        assert {key: job_store.read(key) for key in frozen_remote} == frozen_remote
        remote.finish(spec, provider_phase)
        state = remote.restart().tick()
        assert state.phase == provider_phase
        expected_charge = (
            remote.cohort.prior_dollars
            + remote.cohort.controller_dollars
            + (remote.clock.now - resource.claim.intent_time)
            * (resource.pod.rate + machine_spec.machine.storage_hourly_allowance)
            / 3600
        )
        assert state.charged_dollars(remote.cohort) == pytest.approx(expected_charge)
        remote.clock.now += 60
        for _ in range(2):
            state = remote.restart().tick()
            assert state.phase == provider_phase
            assert len(state.attempts) == remote.provider.created == 1
            assert state.attempts[0].spec.deadline == allocation.deadline
            assert state.charged_dollars(remote.cohort) == pytest.approx(
                expected_charge
            )
        assert remote.verified == (
            [spec.job_id] if provider_phase == "completed" else []
        )
