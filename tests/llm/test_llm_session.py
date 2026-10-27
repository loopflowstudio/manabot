"""A reused runtime stays isolated and a stalled decision cannot select an action."""

import time

import pytest

from manabot.llm.schema import (
    DecisionInput,
    EncodedPrompt,
    GeneratedResponse,
    InferenceConfig,
    Message,
    RuntimeIdentity,
)
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
