"""A reused runtime stays isolated and a stalled decision cannot select an action."""

from pathlib import Path
import subprocess
import sys
from typing import Callable

import pytest

from manabot.llm import session as session_module
from manabot.llm.schema import (
    DecisionInput,
    InferenceConfig,
    SelectionResult,
)
from manabot.llm.selection import Runtime
from manabot.llm.session import InferenceSession
from manabot.sim.child import ChildProcess
from tests.fixtures.llm_process import (
    CountingRuntime,
    ReplyMismatch,
    load_exit,
    load_failure,
    load_stall,
)


def _config() -> InferenceConfig:
    return InferenceConfig(
        model_id="test",
        model_revision="a" * 40,
        tokenizer_id="test",
        tokenizer_revision="a" * 40,
        device="cpu",
        precision="float32",
        context_tokens=64,
        max_new_tokens=4,
        deadline_seconds=1,
        fallback_seed=1,
    )


def _decision(text: str = "choose") -> DecisionInput:
    return DecisionInput(
        text=text,
        legal_references=("A0", "A1"),
        representation_id="test",
        source_identity={},
    )


def test_session_loads_once_and_keeps_decision_evidence_separate() -> None:
    with InferenceSession(
        _config(), factory=CountingRuntime, load_seconds=10
    ) as session:
        first = session.decide(_decision("first"))
        second = session.decide(_decision("second"))
        assert first.selected_reference == "A0"
        assert second.selected_reference == "A1"
        assert first.origin == second.origin == "initial"
        assert len(first.attempts) == len(second.attempts) == 1
        assert first.decision.text == "first" and second.decision.text == "second"
        assert (
            first.behavior_version
            == second.behavior_version
            == session.behavior_version
        )
        assert first.behavior_version == session.identity.behavior_version(_config())
    with pytest.raises(RuntimeError, match="closed"):
        session.decide(_decision())


def test_session_deadline_retains_pending_prompt_and_invalidates_worker() -> None:
    with InferenceSession(
        _config(), factory=CountingRuntime, load_seconds=10
    ) as session:
        result = session.decide(_decision("stall"))
        assert result.failure is not None and result.failure.kind == "deadline"
        assert result.attempts[0].prompt is not None
        assert result.attempts[0].response is None
        assert result.selected_reference is result.origin is result.fallback is None
        assert session.closed
        assert session.process.exitcode is not None
        assert session.connection.closed
        with pytest.raises(RuntimeError, match="closed"):
            session.decide(_decision())


@pytest.mark.parametrize(
    "mismatch", ["request", "version", "decision", "configuration", "runtime"]
)
def test_session_rejects_mismatched_replies_before_publishing_actions(
    monkeypatch: pytest.MonkeyPatch, mismatch: ReplyMismatch
) -> None:
    def child(target: str, arguments: tuple[object, ...]) -> ChildProcess:
        return ChildProcess(
            "tests.fixtures.llm_process:wrong_reply", (mismatch, *arguments)
        )

    monkeypatch.setattr(session_module, "ChildProcess", child)
    received: list[SelectionResult] = []
    with InferenceSession(
        _config(), factory=CountingRuntime, load_seconds=10
    ) as session:
        result = session.decide(_decision(), progress=received.append)
        assert result.failure is not None
        assert "stale or mismatched" in result.failure.message
        assert result.selected_reference is result.origin is result.fallback is None
        assert session.closed
        assert session.process.exitcode is not None
        assert session.connection.closed
        assert received == []


def test_behavior_identity_binds_model_tokenizer_and_effective_runtime() -> None:
    config = _config()
    identity = CountingRuntime(config).identity
    version = identity.behavior_version(config)
    assert (
        identity.behavior_version(
            InferenceConfig.model_validate_json(config.model_dump_json())
        )
        == version
    )
    for changed in (
        {"model_revision": "b" * 40},
        {"tokenizer_revision": "b" * 40},
        {"max_new_tokens": 5},
        {"enable_thinking": True},
    ):
        assert identity.behavior_version(config.model_copy(update=changed)) != version
    assert (
        identity.model_copy(update={"chat_template": "other"}).behavior_version(config)
        != version
    )


@pytest.mark.parametrize(
    "factory,error",
    [(load_failure, RuntimeError), (load_stall, TimeoutError), (load_exit, EOFError)],
)
def test_failed_session_admission_reaps_child(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[InferenceConfig], Runtime],
    error: type[Exception],
) -> None:
    children: list[ChildProcess] = []

    def child(target: str, arguments: tuple[object, ...]) -> ChildProcess:
        process = ChildProcess(target, arguments)
        children.append(process)
        return process

    monkeypatch.setattr(session_module, "ChildProcess", child)
    with pytest.raises(error):
        InferenceSession(_config(), factory=factory, load_seconds=1)
    assert len(children) == 1
    assert children[0].exitcode is not None
    assert not children[0].is_alive()
    assert children[0].connection.closed


def test_session_child_does_not_reexecute_parent_main(tmp_path: Path) -> None:
    marker = tmp_path / "main-imports"
    parent = tmp_path / "parent.py"
    parent.write_text(
        "from pathlib import Path\n"
        f"with Path({str(marker)!r}).open('a') as f: f.write('main\\n')\n"
        "from manabot.llm.session import InferenceSession\n"
        "from manabot.llm.schema import InferenceConfig\n"
        "from tests.fixtures.llm_process import CountingRuntime\n"
        "if __name__ == '__main__':\n"
        f"    config = InferenceConfig.model_validate_json({_config().model_dump_json()!r})\n"
        "    with InferenceSession(config, factory=CountingRuntime, load_seconds=10): pass\n"
    )
    subprocess.run([sys.executable, str(parent)], check=True, timeout=20)
    assert marker.read_text().splitlines() == ["main"]


def test_session_ipc_exit_keeps_pending_evidence_and_reaps_worker() -> None:
    with InferenceSession(
        _config(), factory=CountingRuntime, load_seconds=10
    ) as session:
        result = session.decide(_decision("exit"))
        assert result.failure is not None and result.failure.kind == "runtime"
        assert result.attempts[0].prompt is not None
        assert result.attempts[0].response is None
        assert result.selected_reference is result.origin is result.fallback is None
        assert session.closed and session.connection.closed
        assert session.process.exitcode == 7
