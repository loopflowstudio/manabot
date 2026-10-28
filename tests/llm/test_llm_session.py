"""A reused runtime stays isolated and a stalled decision cannot select an action."""

from functools import partial
from multiprocessing.connection import Connection
import time
from typing import Callable

import pytest

from manabot.llm import session as session_module
from manabot.llm.schema import (
    DecisionInput,
    EncodedPrompt,
    GeneratedResponse,
    InferenceConfig,
    Message,
    RuntimeIdentity,
    SelectionResult,
)
from manabot.llm.selection import Runtime
from manabot.llm.session import InferenceSession


class CountingRuntime:
    def __init__(self, config: InferenceConfig) -> None:
        self.calls = 0
        self.identity = RuntimeIdentity(
            runtime="test",
            runtime_version="1",
            torch_version="unused",
            python_version="test",
            device_name="cpu",
            chat_template="test",
            generation_config={},
        )

    def encode(self, messages: tuple[Message, ...]) -> EncodedPrompt:
        return EncodedPrompt(text=messages[0].content, token_ids=(1, 2))

    def generate(self, prompt: EncodedPrompt, deadline: float) -> GeneratedResponse:
        if "stall" in prompt.text:
            time.sleep(20)
        response = GeneratedResponse(
            text=f"A{self.calls}", token_ids=(3,), finish_reason="eos"
        )
        self.calls += 1
        return response

    def peak_gpu_bytes(self) -> int | None:
        return None


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
        with pytest.raises(RuntimeError, match="closed"):
            session.decide(_decision())


def _wrong_reply(
    mismatch: str,
    connection: Connection,
    config: InferenceConfig,
    factory: Callable[[InferenceConfig], Runtime],
) -> None:
    """Controlled peer sends a well-formed action belonging to another request."""
    with connection:
        identity = factory(config).identity
        version = identity.behavior_version(config)
        if mismatch == "admission":
            connection.send(session_module._Ready(identity, "0" * 64))
            return
        connection.send(session_module._Ready(identity, version))
        request = connection.recv()
        result = SelectionResult(
            decision=request.decision,
            config=config,
            runtime=identity,
            behavior_version=version,
            selected_reference="A0",
            origin="initial",
        )
        request_id = request.request_id
        if mismatch == "request":
            request_id = "previous-request"
        elif mismatch == "version":
            result = result.model_copy(update={"behavior_version": "0" * 64})
        elif mismatch == "decision":
            result = result.model_copy(update={"decision": _decision("previous")})
        elif mismatch == "configuration":
            result = result.model_copy(
                update={"config": config.model_copy(update={"max_new_tokens": 5})}
            )
        elif mismatch == "runtime":
            result = result.model_copy(
                update={
                    "runtime": identity.model_copy(update={"runtime_version": "other"})
                }
            )
        connection.send(session_module._Progress(request_id, result))


@pytest.mark.parametrize(
    "mismatch", ["request", "version", "decision", "configuration", "runtime"]
)
def test_session_rejects_mismatched_replies_before_publishing_actions(
    monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    monkeypatch.setattr(session_module, "_serve", partial(_wrong_reply, mismatch))
    received: list[SelectionResult] = []
    with InferenceSession(
        _config(), factory=CountingRuntime, load_seconds=10
    ) as session:
        result = session.decide(_decision(), progress=received.append)
        assert result.failure is not None
        assert "stale or mismatched" in result.failure.message
        assert result.selected_reference is result.origin is result.fallback is None
        assert session.closed
        assert received == []


def test_session_rejects_wrong_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_module, "_serve", partial(_wrong_reply, "admission"))
    with pytest.raises(RuntimeError, match="different behavior version"):
        InferenceSession(_config(), factory=CountingRuntime, load_seconds=10)


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
