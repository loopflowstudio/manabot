"""Bounded fixtures retain complete games, failed attempts and exact input replay."""

from collections.abc import Callable
import json
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import psutil
from pydantic import JsonValue
import pytest
from test_batched_eval import fixture_policy

from experiments.inference_baseline import (
    Cell,
    PolicyWorkload,
    Protocol,
    Recipe,
    _compare_outputs,
    freeze,
    run,
    validate,
)
from experiments.report_prototype.build import write_report
from manabot._process import bounded_call
from manabot.eval.evaluation import EvaluationRequest
from manabot.reports.inference import EvidenceFile
from manabot.sim.execution import EvalCompute, EvalExecution

if TYPE_CHECKING:
    from manabot.llm.schema import (
        EncodedPrompt,
        GeneratedResponse,
        InferenceConfig,
        Message,
    )


def _protocol(tmp_path: Path, *, trace: Path | None = None, roots: int = 2) -> Protocol:
    key, player, anchor, checkpoint = fixture_policy(tmp_path)
    request = EvaluationRequest(
        key=key,
        candidate=player,
        opponent=anchor,
        checkpoint_paths={player.player_id: str(checkpoint)},
        deal_seeds=(18801,),
        game_seconds=120,
        max_commands=10000,
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json())
    compute = tmp_path / "compute.json"
    compute.write_text(
        EvalCompute(
            max_active_games=2, max_batch_rows=2, max_delay_seconds=0
        ).model_dump_json()
    )
    recipe = Recipe(
        workloads=(
            PolicyWorkload(
                id="fixture",
                kind="policy",
                request=request_path,
                legs=(0, 1),
                trace=trace,
                roots=roots,
            ),
        ),
        cells=(Cell(id="shared", workload="fixture", execution=compute),),
        repetitions=1,
        cell_seconds=120,
        total_seconds=180,
    )
    return freeze(recipe)


def _fail(
    sender: Connection,
    protocol: Protocol,
    cell: Cell,
    out: Path,
    replay: bool,
    deadline: float,
) -> None:
    (out / "received.json").write_text('{"progress":"before injected failure"}')
    sender.send(
        {"status": "failed", "reason": "injected runtime failure", "completed_games": 0}
    )


def _wait() -> None:
    # A sleeping child is a lifecycle fixture, not artificial resource pressure.
    time.sleep(30)


def _orphan(sender: Connection, path: Path) -> None:
    child = mp.get_context("spawn").Process(target=_wait)
    child.start()
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps({"pid": child.pid}))
    temporary.replace(path)
    time.sleep(30)


def test_failure_is_pinned_and_reported_without_a_fake_game(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    manifest = run(protocol, tmp_path / "run", worker=_fail)
    attempt = manifest.attempts[0]
    assert attempt.status == "failed" and attempt.files
    receipt = json.loads((tmp_path / "run/0-shared/attempt.json").read_text())
    assert receipt["whole_attempt_seconds"] >= receipt["supervision"]["seconds"]
    assert receipt["supervision"]["cleanup_complete"]
    report = write_report([], [], tmp_path / "report", tmp_path / "run/manifest.json")
    data = json.loads(report.with_name("data.json").read_text())["experiments"][0]
    projected = data["inference_performance"]["attempts"][0]
    assert projected["observed_status"] == "failed"
    assert projected["run"] is None
    assert projected["execution"]["supervision"]["payload"]["completed_games"] == 0
    assert (report.parent / projected["execution_evidence"]).is_file()
    original = (tmp_path / "run/0-shared/received.json").read_bytes()
    (tmp_path / "run/0-shared/received.json").write_text("{}")
    with pytest.raises(ValueError, match="mismatch"):
        write_report([], [], tmp_path / "changed", tmp_path / "run/manifest.json")
    (tmp_path / "run/0-shared/received.json").write_bytes(original)
    (tmp_path / "run/protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="protocol digest"):
        write_report(
            [], [], tmp_path / "changed-protocol", tmp_path / "run/manifest.json"
        )


def test_timeout_stops_owned_descendants_and_retains_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = tmp_path / "child.json"
    real_time = time.time
    deadline = real_time() + 30
    # Advance the parent deadline only after the child's durable readiness
    # witness. Slow interpreter startup is not the behavior under test.
    monkeypatch.setattr(
        time, "time", lambda: deadline if progress.exists() else real_time()
    )
    outcome = bounded_call(
        _orphan,
        (progress,),
        deadline_unix=deadline,
        process_tree_bytes=2 * 2**30,
    )
    assert outcome.error and "deadline" in outcome.error
    assert outcome.cleanup_complete
    pid = json.loads(progress.read_text())["pid"]
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )


def test_frozen_mutation_is_failed_evidence_not_an_unrecorded_preflight(
    tmp_path: Path,
) -> None:
    protocol = _protocol(tmp_path)
    protocol.recipe.workloads[0].request.write_text("{}")
    with pytest.raises(ValueError):
        validate(protocol)
    manifest = run(protocol, tmp_path / "run")
    assert manifest.attempts[0].status == "failed"
    assert manifest.attempts[0].files
    assert (tmp_path / "run/0-shared/attempt.json").is_file()


def test_complete_overlapping_games_and_retained_inputs(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    manifest = run(protocol, tmp_path / "games")
    assert manifest.attempts[0].status == "completed", manifest.attempts[0].reason
    directory = tmp_path / "games/0-shared"
    outcome = json.loads((directory / "result.json").read_text())
    assert outcome["completed_games"] == 2
    rows = json.loads((directory / "rows.json").read_text())
    assert len(rows) == 2 and all(
        row["terminated"] and row["replay_passed"] for row in rows
    )
    assert sum(row["decisions"] for row in rows) == outcome["committed_decisions"]
    assert json.loads((directory / "attempt.json").read_text())["supervision"][
        "cleanup_complete"
    ]
    report = write_report([], [], tmp_path / "report", tmp_path / "games/manifest.json")
    data = json.loads(report.with_name("data.json").read_text())["experiments"][0]
    assert (
        data["inference_performance"]["attempts"][0]["execution"]["supervision"][
            "payload"
        ]["completed_games"]
        == 2
    )
    trace = (
        directory / json.loads((directory / "trace.json").read_text())["artifact_path"]
    )
    workload = protocol.recipe.workloads[0].model_copy(update={"trace": trace})
    recipe = protocol.recipe.model_copy(
        update={
            "workloads": (workload,),
            "cells": (
                Cell(id="scalar", workload="fixture"),
                protocol.recipe.cells[0],
                protocol.recipe.cells[0].model_copy(
                    update={"id": "uninstrumented", "instrumentation": False}
                ),
            ),
        }
    )
    replay = run(freeze(recipe), tmp_path / "inputs", replay=True)
    assert all(a.status == "completed" for a in replay.attempts), [
        a.reason for a in replay.attempts
    ]
    assert all(
        a.kind == "retained-input" and any(f.path == "inputs.npz" for f in a.files)
        for a in replay.attempts
    )
    comparison = json.loads((tmp_path / "inputs/0-shared/comparison.json").read_text())
    assert comparison["legal_logits_close"]
    assert comparison["action_agreement"] == 1
    overhead = json.loads(
        (tmp_path / "inputs/0-uninstrumented/comparison.json").read_text()
    )
    assert overhead["instrumentation_outputs_identical"] is True
    assert overhead["instrumentation_host_overhead"] is not None
    proof = write_report(
        [], [], tmp_path / "input-report", tmp_path / "inputs/manifest.json"
    )
    assert proof.is_file()


@pytest.mark.parametrize("enabled_first", [True, False])
@pytest.mark.parametrize("changed_outputs", [True, False])
def test_instrumentation_pairing_survives_scalar_first_and_repeat_order(
    tmp_path: Path, enabled_first: bool, changed_outputs: bool
) -> None:
    protocol = _protocol(tmp_path)

    def witness(
        name: str,
        *,
        enabled: bool,
        seconds: float,
        rows: int = 2,
        diagnostic_batches: int = 0,
        changed: bool = False,
    ) -> Path:
        root = tmp_path / name
        root.mkdir()
        settings = EvalExecution(
            deadline_unix=1000,
            max_active_games=2,
            max_batch_rows=rows,
            max_delay_seconds=0,
            instrumentation=enabled,
            diagnostic_batches=diagnostic_batches,
        )
        (root / "preparation.json").write_text(
            json.dumps({"execution": settings.model_dump(mode="json")})
        )
        np.savez(root / "inputs.npz", inputs=np.array([[1, 2]]), seeds=np.array([7]))
        np.savez(
            root / "pass-2-batch-0.npz",
            logits=np.array([[1.0, 3.0 if changed else 2.0]]),
            masks=np.array([[1, 1]]),
            actions=np.array([1]),
        )
        (root / "pass-2.json").write_text(json.dumps({"seconds": seconds}))
        return root

    scalar = witness("scalar", enabled=enabled_first, seconds=3, rows=1)
    old = witness("old", enabled=enabled_first, seconds=9)
    paired = witness("paired", enabled=enabled_first, seconds=5 if enabled_first else 4)
    diagnostic = witness(
        "diagnostic", enabled=enabled_first, seconds=12, diagnostic_batches=1
    )
    current = witness(
        "current",
        enabled=not enabled_first,
        seconds=4 if enabled_first else 5,
        changed=changed_outputs,
    )
    references = (scalar, old, paired, diagnostic)
    if changed_outputs:
        with pytest.raises(ValueError, match="instrumentation changed"):
            _compare_outputs(references, current, protocol)
    else:
        _compare_outputs(references, current, protocol)
    comparison = json.loads((current / "comparison.json").read_text())
    assert comparison["reference"] == str(paired.resolve())
    assert comparison["instrumentation_outputs_identical"] is not changed_outputs
    assert comparison["numerically_equivalent"] is not changed_outputs
    assert comparison["instrumentation_host_overhead"] == pytest.approx(0.25)


class _RecoveryRuntime:
    """A finite scripted response sequence; no model loading or synthetic load."""

    def __init__(self, config: "InferenceConfig") -> None:
        from manabot.llm.schema import RuntimeIdentity

        self.responses = iter(("A0", "invalid", "A0", "invalid", "invalid"))
        self.identity = RuntimeIdentity(
            runtime="fixture",
            runtime_version="1",
            torch_version="unused",
            python_version="fixture",
            device_name="cpu",
            chat_template="fixture",
            generation_config={},
        )

    def encode(self, messages: "tuple[Message, ...]") -> "EncodedPrompt":
        from manabot.llm.schema import EncodedPrompt

        return EncodedPrompt(text=messages[0].content, token_ids=(1, 2))

    def generate(self, prompt: "EncodedPrompt", deadline: float) -> "GeneratedResponse":
        from manabot.llm.schema import GeneratedResponse

        try:
            text = next(self.responses)
        except StopIteration:
            raise RuntimeError("injected generation failure") from None
        return GeneratedResponse(text=text, token_ids=(3,), finish_reason="eos")

    def peak_gpu_bytes(self) -> int | None:
        return None


def _llm_failure(
    sender: Connection,
    protocol: Protocol,
    cell: Cell,
    out: Path,
    replay: bool,
    deadline: float,
) -> None:
    from manabot.llm.schema import InferenceConfig
    from manabot.llm.session import InferenceSession
    from manabot.verify.llm_game import GameConfig, play

    config = InferenceConfig(
        model_id="fixture",
        model_revision="a" * 40,
        tokenizer_id="fixture",
        tokenizer_revision="b" * 40,
        device="cpu",
        precision="float32",
        context_tokens=100,
        max_new_tokens=4,
        deadline_seconds=10,
        fallback_seed=7,
    )
    with InferenceSession(config, factory=_RecoveryRuntime, load_seconds=20) as session:
        result = play(
            GameConfig(max_decisions=100, max_seconds=30), out / "game", session
        )
    sender.send(
        {
            "status": "failed",
            "reason": result.failure,
            "completed_games": 0,
            "committed_decisions": result.decisions,
        }
    )


def test_llm_recovery_and_failure_keep_separate_timing_populations(
    tmp_path: Path,
) -> None:
    from experiments.inference_baseline import LLMWorkload

    protocol = _protocol(tmp_path)
    # Controlled runtime injection needs no downloadable model cache. The test
    # invokes the real session, selector, native game writer and parent ledger.
    workload = LLMWorkload(
        id="fixture",
        kind="llm",
        config=tmp_path / "request.json",
        games=(tmp_path / "request.json",),
        cache_manifest=tmp_path / "request.json",
    )
    protocol = protocol.model_copy(
        update={"recipe": protocol.recipe.model_copy(update={"workloads": (workload,)})}
    )
    manifest = run(protocol, tmp_path / "llm", worker=_llm_failure)
    assert manifest.attempts[0].status == "failed"
    report = write_report([], [], tmp_path / "report", tmp_path / "llm/manifest.json")
    data = json.loads(report.with_name("data.json").read_text())["experiments"][0]
    llm = data["inference_performance"]["runs"][0]
    assert (
        llm["coverage"]["initial"],
        llm["coverage"]["repaired"],
        llm["coverage"]["fallback"],
        llm["coverage"]["failed_selections"],
    ) == (1, 1, 1, 1)
    assert llm["selection_latency"]["n"] == 3
    assert llm["failed_selection_latency"]["n"] == 1
    assert llm["generation_host_latency"]["n"] == 5
    assert llm["failed_generation_host_latency"]["n"] == 1
    assert llm["metrics"]["tokenization"]["status"] == "recorded"
    assert llm["metrics"]["whole_attempt"]["status"] == "recorded"
    assert llm["metrics"]["prefill"]["value"] is None
    assert data["inference_performance"]["attempts"][0]["execution"]["supervision"][
        "cleanup_complete"
    ]


def test_controlled_wall_clock_retains_planned_and_failed_denominators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.inference_baseline as baseline
    from manabot._process import ProcessOutcome

    protocol = _protocol(tmp_path)
    protocol = protocol.model_copy(
        update={
            "recipe": protocol.recipe.model_copy(
                update={
                    "cells": (
                        Cell(id="a", workload="fixture"),
                        Cell(id="b", workload="fixture"),
                    ),
                    "repetitions": 3,
                    "total_seconds": 30,
                    "cell_seconds": 20,
                }
            )
        }
    )
    now = 100.0
    monkeypatch.setattr(baseline.time, "monotonic", lambda: now)
    monkeypatch.setattr(baseline.time, "time", lambda: now)
    calls = 0
    worker_deadline = 0.0

    def fixture_call(
        target: object,
        args: tuple[object, ...],
        *,
        deadline_unix: float,
        process_tree_bytes: int,
        on_progress: Callable[[dict[str, JsonValue]], None] | None = None,
    ) -> ProcessOutcome:
        nonlocal now, calls, worker_deadline
        if on_progress is not None:
            assert deadline_unix == worker_deadline
            assert process_tree_bytes == protocol.recipe.process_tree_bytes
            now += 1
            return ProcessOutcome(
                {"status": "completed", "inventory_complete": True},
                None,
                1,
                0.25,
                512,
                0,
                True,
            )
        ledger = json.loads((tmp_path / "run/manifest.json").read_text())["attempts"]
        assert len(ledger) == 6
        assert [a["id"] for a in ledger] == ["0-a", "0-b", "1-b", "1-a", "2-a", "2-b"]
        assert ledger[calls]["status"] == "attempted"
        assert all(a["status"] == "planned" for a in ledger[calls + 1 :])
        calls += 1
        worker_deadline = deadline_unix
        assert deadline_unix == min(130, now + 20)
        now += 13
        return ProcessOutcome(
            {
                "status": "failed" if calls == 1 else "completed",
                "reason": "injected measurements",
                "completed_games": 2,
                "summed_workers_seconds": 20,
            },
            None,
            13,
            2,
            1024,
            0,
            True,
        )

    monkeypatch.setattr(baseline, "bounded_call", fixture_call)
    manifest = run(protocol, tmp_path / "run")
    assert [a.status for a in manifest.attempts] == [
        "failed",
        "completed",
        "unattempted",
        "unattempted",
        "unattempted",
        "unattempted",
    ]
    for attempt in manifest.attempts[:2]:
        receipt = json.loads(
            (tmp_path / "run" / attempt.id / "attempt.json").read_text()
        )
        assert receipt["whole_attempt_seconds"] == 14
        assert receipt["supervision"]["cleanup_seconds"] == 2
        assert receipt["collection_seconds"] == 1
        assert receipt["supervision"]["collection"]["seconds"] == 1
    assert json.loads((tmp_path / "run/invocation.json").read_text())["seconds"] == 28


def _received(
    sender: Connection,
    protocol: Protocol,
    cell: Cell,
    out: Path,
    replay: bool,
    deadline: float,
) -> None:
    (out / "a.json").write_text('{"progress":"durable worker evidence"}')
    # Collection fixtures never execute or parse these arrays.
    (out / "inputs.npz").write_bytes(b"retained witness")
    sender.send({"status": "completed", "completed_games": 2})


@pytest.mark.parametrize("cleanup_complete", [True, False])
def test_exhausted_execution_budget_does_not_start_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_complete: bool
) -> None:
    import experiments.inference_baseline as baseline
    from manabot._process import ProcessOutcome

    protocol = _protocol(tmp_path)
    protocol = protocol.model_copy(
        update={
            "recipe": protocol.recipe.model_copy(
                update={
                    "cells": (
                        Cell(id="a", workload="fixture"),
                        Cell(id="b", workload="fixture"),
                    ),
                    "cell_seconds": 20,
                    "total_seconds": 30,
                }
            )
        }
    )
    now = 100.0
    monkeypatch.setattr(baseline.time, "time", lambda: now)
    monkeypatch.setattr(baseline.time, "monotonic", lambda: now)
    calls = 0

    def supervised(
        target: Callable[..., None],
        args: tuple[object, ...],
        *,
        deadline_unix: float,
        process_tree_bytes: int,
        on_progress: Callable[[dict[str, JsonValue]], None] | None = None,
    ) -> ProcessOutcome:
        nonlocal now, calls
        calls += 1
        assert deadline_unix == 120
        if on_progress is not None:
            return bounded_call(
                target,
                args,
                deadline_unix=deadline_unix,
                process_tree_bytes=process_tree_bytes,
                on_progress=on_progress,
            )
        directory = args[2]
        assert isinstance(directory, Path)
        (directory / "received.json").write_text('{"progress":1}')
        now = 128 if cleanup_complete else 110
        return ProcessOutcome(
            {"status": "completed"}, None, now - 100, 2, 1024, 0, cleanup_complete
        )

    monkeypatch.setattr(baseline, "bounded_call", supervised)
    out = tmp_path / "run"
    manifest = run(protocol, out)
    assert calls == (2 if cleanup_complete else 1)
    assert [a.status for a in manifest.attempts] == ["failed", "unattempted"]
    attempt = manifest.attempts[0]
    assert not attempt.inventory_complete
    assert [f.path for f in attempt.files] == ["attempt.json"]
    receipt = json.loads((out / "0-a/attempt.json").read_text())
    if cleanup_complete:
        assert "no execution budget" in attempt.reason
        assert receipt["supervision"]["collection"]["exitcode"] is None
    else:
        assert "collection not started" in attempt.reason
        assert receipt["supervision"]["collection"] is None
        assert "cleanup incomplete" in manifest.attempts[1].reason
    assert (out / "0-a/received.json").is_file()
    assert write_report([], [], tmp_path / "report", out / "manifest.json").is_file()


def _collect_blocked_hash(
    sender: Connection,
    protocol: Protocol,
    directory: Path,
    references: tuple[Path, ...],
) -> None:
    import experiments.inference_baseline as baseline
    import manabot.reports.inference as receipts

    pin = receipts.pin_file

    def blocked(path: Path, *, relative_to: Path | None = None) -> EvidenceFile:
        if path.name == "inputs.npz":
            _orphan(sender, directory.parent / "collection-ready.json")
        return pin(path, relative_to=relative_to)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(receipts, "pin_file", blocked)
        baseline._collect(sender, protocol, directory, references)


def _collect_blocked_comparison(
    sender: Connection,
    protocol: Protocol,
    directory: Path,
    references: tuple[Path, ...],
) -> None:
    import experiments.inference_baseline as baseline

    def blocked(
        references: tuple[Path, ...], directory: Path, protocol: Protocol
    ) -> None:
        (directory / "comparison.json.pending").write_text("unfinished comparison")
        _orphan(sender, directory.parent / "collection-ready.json")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(baseline, "_compare_outputs", blocked)
        baseline._collect(sender, protocol, directory, references)


@pytest.mark.parametrize("phase", ["hash", "comparison"])
@pytest.mark.parametrize("budget", ["invocation", "cell", "rss"])
def test_collection_exhaustion_retains_pins_and_reaps_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, budget: str
) -> None:
    import experiments.inference_baseline as baseline
    from manabot._process import ProcessOutcome

    protocol = _protocol(tmp_path)
    protocol = protocol.model_copy(
        update={
            "recipe": protocol.recipe.model_copy(
                update={
                    "cells": tuple(
                        Cell(id=name, workload="fixture") for name in ("a", "b", "c")
                    ),
                    "cell_seconds": 60,
                    "total_seconds": 60 if budget == "invocation" else 180,
                }
            )
        }
    )
    out = tmp_path / "run"
    ready = out / "collection-ready.json"
    exhausted = False
    clock_offset = 0.0
    real_time = time.time
    memory_info = psutil.Process.memory_info

    def measured_memory(process: psutil.Process) -> object:
        if budget == "rss" and exhausted:
            return SimpleNamespace(rss=protocol.recipe.process_tree_bytes + 1)
        return memory_info(process)

    # Advance on the next parent clock read once both the pin and child are
    # available, including when readiness follows the last progress message.
    received_pin = False

    def clock() -> float:
        nonlocal exhausted, clock_offset
        if ready.exists() and received_pin and not exhausted:
            exhausted = True
            if budget != "rss":
                clock_offset = active_deadline - real_time()
        return real_time() + clock_offset

    active_deadline = 0.0
    children: list[int] = []

    def observed_call(
        target: Callable[..., None],
        args: tuple[object, ...],
        *,
        deadline_unix: float,
        process_tree_bytes: int,
        on_progress: Callable[[dict[str, JsonValue]], None] | None = None,
    ) -> ProcessOutcome:
        nonlocal active_deadline, received_pin, exhausted
        active_deadline = deadline_unix
        received_pin = False

        def receive(progress: dict[str, JsonValue]) -> None:
            nonlocal received_pin
            assert on_progress is not None
            on_progress(progress)
            received_pin = True

        outcome = bounded_call(
            target,
            args,
            deadline_unix=deadline_unix,
            process_tree_bytes=process_tree_bytes,
            on_progress=receive if on_progress is not None else None,
        )
        if ready.exists():
            children.append(json.loads(ready.read_text())["pid"])
            ready.unlink()
            exhausted = False
        return outcome

    monkeypatch.setattr(baseline.time, "time", clock)
    monkeypatch.setattr(psutil.Process, "memory_info", measured_memory)
    monkeypatch.setattr(baseline, "bounded_call", observed_call)
    collector = (
        _collect_blocked_hash if phase == "hash" else _collect_blocked_comparison
    )
    manifest = run(protocol, out, replay=True, worker=_received, collector=collector)
    failed_index = 0 if phase == "hash" else 1
    attempt = manifest.attempts[failed_index]
    assert attempt.status == "failed"
    assert not attempt.inventory_complete
    assert "collection:" in attempt.reason
    assert ("RSS" if budget == "rss" else "deadline") in attempt.reason
    if budget == "invocation":
        assert all(
            a.status == "unattempted" for a in manifest.attempts[failed_index + 1 :]
        )
        assert all(
            not (out / a.id).exists() for a in manifest.attempts[failed_index + 1 :]
        )
    directory = out / attempt.id
    assert (directory / "inputs.npz").read_bytes() == b"retained witness"
    assert any(f.path == "a.json" for f in attempt.files)
    receipt = json.loads((directory / "attempt.json").read_text())
    assert receipt["supervision"]["payload"]["completed_games"] == 2
    assert receipt["supervision"]["collection"]["cleanup_complete"]
    assert children
    assert all(
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        for pid in children
    )
    assert receipt["whole_attempt_seconds"] >= receipt["collection_seconds"]
    report = write_report([], [], tmp_path / "report", out / "manifest.json")
    data = json.loads(report.with_name("data.json").read_text())["experiments"][0]
    projected = data["inference_performance"]["attempts"][failed_index]
    assert projected["observed_status"] == "failed" and projected["run"] is None
    assert (report.parent / projected["execution_evidence"]).is_file()
    (directory / "a.json").write_text("{}")
    with pytest.raises(ValueError, match="mismatch"):
        write_report([], [], tmp_path / "changed", out / "manifest.json")
