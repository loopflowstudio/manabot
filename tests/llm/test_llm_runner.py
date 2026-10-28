"""The production process supervisor must stop stalled inference and keep evidence."""

import multiprocessing
from multiprocessing.connection import Connection
import os
import subprocess
import sys
import time

import pytest

from manabot.llm import runner
from manabot.llm.schema import (
    Attempt,
    DecisionInput,
    GeneratedResponse,
    InferenceConfig,
    Message,
    SelectionResult,
)


def _stall(
    connection: Connection,
    decision: DecisionInput,
    config: InferenceConfig,
    started: float,
) -> None:
    connection.send(
        SelectionResult(
            decision=decision,
            config=config,
            attempts=(
                Attempt(
                    messages=(Message(role="user", content=decision.text),),
                    response=GeneratedResponse(
                        text="bad", token_ids=(42,), finish_reason="eos"
                    ),
                    validation_error="expected a legal reference",
                ),
            ),
        ).model_dump_json()
    )
    time.sleep(30)


def _exit(
    connection: Connection,
    decision: DecisionInput,
    config: InferenceConfig,
    started: float,
) -> None:
    os._exit(1)


def _complete(
    connection: Connection,
    decision: DecisionInput,
    config: InferenceConfig,
    started: float,
) -> None:
    connection.send(
        SelectionResult(
            decision=decision,
            config=config,
            selected_reference=decision.legal_references[0],
            origin="initial",
        ).model_dump_json()
    )
    connection.close()


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


def test_process_deadline_retains_completed_attempt(
    monkeypatch: pytest.MonkeyPatch,
    decision: DecisionInput,
    config: InferenceConfig,
) -> None:
    monkeypatch.setattr(runner, "_worker", _stall)
    result = runner.run_decision(decision, config)
    assert result.failure.kind == "deadline"
    assert result.seconds < 6
    assert result.attempts[0].response.text == "bad"
    assert result.origin is None and result.fallback is None


def test_worker_exit_is_an_explicit_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
    decision: DecisionInput,
    config: InferenceConfig,
) -> None:
    monkeypatch.setattr(runner, "_worker", _exit)
    # The deadline covers spawned-interpreter startup; exit must not race it.
    result = runner.run_decision(
        decision, config.model_copy(update={"deadline_seconds": 30})
    )
    assert result.failure.kind == "runtime"
    assert result.origin is None and result.fallback is None


def test_completed_worker_retains_queued_selection(
    monkeypatch: pytest.MonkeyPatch,
    decision: DecisionInput,
    config: InferenceConfig,
) -> None:
    monkeypatch.setattr(runner, "_worker", _complete)
    # The deadline covers spawned-interpreter startup; success must not race it.
    result = runner.run_decision(
        decision, config.model_copy(update={"deadline_seconds": 30})
    )
    assert result.failure is None
    assert result.origin == "initial" and result.selected_reference == "A0"


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
        result = SelectionResult.model_validate_json(receiver.recv())
        assert result.failure.kind == "runtime"
        assert "ModuleNotFoundError" in result.failure.message
        assert result.decision == decision and result.config == config
        assert result.origin is None and result.fallback is None
        assert receiver.poll(1)
        with pytest.raises(EOFError):
            receiver.recv()
