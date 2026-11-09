"""The production process supervisor must stop stalled inference and keep evidence."""

import multiprocessing
import subprocess
import sys
import time

import pytest

from manabot.llm import runner
from manabot.llm.schema import (
    DecisionInput,
    InferenceConfig,
    SelectionResult,
)
from manabot.sim.child import ChildProcess


@pytest.fixture
def decision() -> DecisionInput:
    return DecisionInput(
        text="test",
        legal_references=("A0",),
        representation_id="fixture",
        source_identity={},
    )


@pytest.fixture
def config() -> InferenceConfig:
    return InferenceConfig(
        model_id="fixture/model",
        model_revision="a" * 40,
        tokenizer_id="fixture/model",
        tokenizer_revision="a" * 40,
        device="cpu",
        precision="float32",
        context_tokens=100,
        max_new_tokens=8,
        deadline_seconds=2,
        fallback_seed=1,
    )


@pytest.mark.parametrize(
    ("target", "failure"),
    [("_stall", "deadline"), ("_exit", "runtime"), ("_complete", None)],
)
def test_one_shot_lifecycle_retains_evidence_and_reaps_child(
    monkeypatch: pytest.MonkeyPatch,
    decision: DecisionInput,
    config: InferenceConfig,
    target: str,
    failure: str | None,
) -> None:
    children: list[ChildProcess] = []

    def child(worker: str, arguments: tuple[object, ...]) -> ChildProcess:
        process = ChildProcess(f"tests.fixtures.llm_process:{target}", arguments)
        children.append(process)
        return process

    monkeypatch.setattr(runner, "ChildProcess", child)
    # Only the stalled worker should race the startup-inclusive deadline.
    config = config.model_copy(
        update={"deadline_seconds": 2 if failure == "deadline" else 30}
    )
    result = runner.run_decision(decision, config)
    if failure is None:
        assert result.failure is None
        assert result.origin == "initial" and result.selected_reference == "A0"
    else:
        assert result.failure is not None and result.failure.kind == failure
        assert result.selected_reference is result.origin is result.fallback is None
    if failure == "deadline":
        assert result.seconds < 6
        assert result.attempts[0].response.text == "bad"
    assert len(children) == 1
    assert children[0].exitcode is not None
    assert not children[0].is_alive()
    assert children[0].connection.closed


def test_importing_harness_does_not_load_optional_runtime() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import manabot.llm.runner; assert 'transformers' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        check=True,
        timeout=10,
    )


def test_missing_runtime_returns_failure_and_closes_worker_pipe(
    monkeypatch: pytest.MonkeyPatch,
    decision: DecisionInput,
    config: InferenceConfig,
) -> None:
    monkeypatch.setitem(sys.modules, "manabot.llm.transformers_runtime", None)
    receiver, sender = multiprocessing.Pipe(duplex=False)
    with receiver, sender:
        runner._worker(sender, decision, config, time.monotonic())
        assert receiver.poll(1)
        result = receiver.recv()
        assert isinstance(result, SelectionResult)
        assert result.failure.kind == "runtime"
        assert "ModuleNotFoundError" in result.failure.message
        assert result.decision == decision and result.config == config
        assert result.origin is None and result.fallback is None
        assert receiver.poll(1)
        with pytest.raises(EOFError):
            receiver.recv()
