"""Behavioral checks for text selection, evidence and protocol-only recovery."""

from dataclasses import dataclass, field
import random
import time

from pydantic import ValidationError
import pytest

from manabot.llm.schema import (
    DecisionInput,
    EncodedPrompt,
    GeneratedResponse,
    InferenceConfig,
    Message,
    RuntimeIdentity,
    SelectionResult,
)
from manabot.llm.selection import REPAIR, select


def decision() -> DecisionInput:
    return DecisionInput(
        text="Synthetic decision. A0 — wait. A1 — continue.",
        legal_references=("A0", "A1"),
        representation_id="handwritten-test-v1",
        source_identity={"fixture": "protocol-only; no native observation"},
    )


def config(**overrides: object) -> InferenceConfig:
    values: dict[str, object] = dict(
        model_id="fixture/model",
        model_revision="a" * 40,
        tokenizer_id="fixture/tokenizer",
        tokenizer_revision="b" * 40,
        device="cpu",
        precision="float32",
        context_tokens=100,
        max_new_tokens=8,
        deadline_seconds=30,
        fallback_seed=37,
    )
    return InferenceConfig.model_validate(values | overrides)


def response(text: str, *, complete: bool = True) -> GeneratedResponse:
    return GeneratedResponse(
        text=text,
        token_ids=(10, 11, 12),
        finish_reason="eos" if complete else "length",
    )


@dataclass
class ScriptedRuntime:
    replies: list[GeneratedResponse | Exception]
    prompt_lengths: list[int] = field(default_factory=lambda: [5, 9])
    identity: RuntimeIdentity = field(
        default_factory=lambda: RuntimeIdentity(
            runtime="scripted-test",
            runtime_version="1",
            torch_version="not-loaded",
            python_version="test",
            device_name="cpu",
            chat_template="fixture template",
            generation_config={"do_sample": False},
        )
    )

    def encode(self, messages: tuple[Message, ...]) -> EncodedPrompt:
        return EncodedPrompt(
            text="\n".join(message.content for message in messages),
            token_ids=tuple(range(self.prompt_lengths.pop(0))),
        )

    def generate(self, prompt: EncodedPrompt, deadline: float) -> GeneratedResponse:
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def peak_gpu_bytes(self) -> int | None:
        return None


def test_initial_selection_retains_exact_input_output_and_identity() -> None:
    result = select(decision(), config(), ScriptedRuntime([response("  A1\n")]))
    assert result.selected_reference == "A1"
    assert result.origin == "initial"
    assert result.failure is None and result.fallback is None
    assert result.decision == decision()
    assert result.runtime.runtime == "scripted-test"
    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert attempt.response.text == "  A1\n"
    assert attempt.prompt_tokens == len(attempt.prompt.token_ids) == 5
    assert attempt.output_tokens == len(attempt.response.token_ids) == 3
    assert attempt.seconds >= 0 and result.seconds >= attempt.seconds
    assert result.peak_gpu_bytes is None
    assert SelectionResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("invalid", ["", "A9", "A0 A1", "I choose A1", "A1."])
def test_repair_preserves_input_and_failed_answer(invalid: str) -> None:
    snapshots: list[SelectionResult] = []
    result = select(
        decision(),
        config(),
        ScriptedRuntime([response(invalid), response("A0")]),
        progress=snapshots.append,
    )
    assert result.origin == "retry" and result.selected_reference == "A0"
    first, second = result.attempts
    assert first.validation_error is not None and second.validation_error is None
    assert second.messages[0] == first.messages[0]
    assert second.messages[1] == Message(role="assistant", content=invalid)
    assert second.messages[2] == Message(role="user", content=REPAIR)
    assert second.prompt_tokens == 9
    # Earlier pending evidence stays pending after the final response arrives.
    pending = next(snapshot for snapshot in snapshots if snapshot.attempts)
    assert len(pending.attempts) == 1
    assert pending.attempts[0].prompt is None
    assert pending.attempts[0].response is None
    assert pending.attempts[0].seconds is None
    assert snapshots[-1] == result


def test_two_protocol_failures_produce_reproducible_legal_fallback() -> None:
    def run() -> SelectionResult:
        return select(
            decision(),
            config(),
            ScriptedRuntime([response("no"), response("still no")]),
        )

    first, second = run(), run()
    expected = random.Random(37).randrange(2)
    assert first.origin == second.origin == "fallback"
    assert (
        first.selected_reference
        == second.selected_reference
        == decision().legal_references[expected]
    )
    assert first.fallback == second.fallback
    assert first.fallback.selected_index == expected
    assert first.fallback.state_before == repr(random.Random(37).getstate())
    assert len(first.attempts) == 2 and all(a.validation_error for a in first.attempts)
    assert first.failure is None


def test_length_limited_legal_looking_answer_is_incomplete() -> None:
    result = select(
        decision(),
        config(),
        ScriptedRuntime(
            [
                response("A1", complete=False),
                response("A0"),
            ]
        ),
    )
    assert result.origin == "retry"
    assert (
        result.attempts[0].validation_error
        == "incomplete response: generation limit reached"
    )


@pytest.mark.parametrize("after_invalid", [False, True])
@pytest.mark.parametrize(
    "error", [RuntimeError("OOM"), ValueError("engine rejection"), TimeoutError("late")]
)
def test_runtime_failures_never_trigger_fallback(
    error: Exception, after_invalid: bool
) -> None:
    replies: list[GeneratedResponse | Exception] = [error]
    if after_invalid:
        replies.insert(0, response("bad"))
    result = select(decision(), config(), ScriptedRuntime(replies))
    assert (
        result.origin is None
        and result.selected_reference is None
        and result.fallback is None
    )
    assert result.failure.kind == (
        "deadline" if isinstance(error, TimeoutError) else "runtime"
    )
    assert str(error) in result.failure.message
    assert len(result.attempts) == (2 if after_invalid else 1)


@pytest.mark.parametrize("lengths, attempts", [([93], 1), ([5, 93], 2)])
def test_overflow_preserves_full_tokens_and_never_falls_back(
    lengths: list[int], attempts: int
) -> None:
    result = select(decision(), config(), ScriptedRuntime([response("bad")], lengths))
    assert result.failure.kind == "overflow"
    assert result.origin is None and result.fallback is None
    assert len(result.attempts) == attempts
    assert len(result.attempts[-1].prompt.token_ids) == 93
    assert result.attempts[-1].response is None


def test_exact_context_boundary_is_admitted() -> None:
    result = select(decision(), config(), ScriptedRuntime([response("A1")], [92]))
    assert result.origin == "initial"


def test_expired_deadline_cannot_accept_a_response() -> None:
    result = select(
        decision(), config(), ScriptedRuntime([]), started=time.monotonic() - 31
    )
    assert result.failure.kind == "deadline" and not result.attempts
    assert result.origin is None


def test_late_runtime_response_is_retained_but_not_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: 0.0)

    class LateRuntime(ScriptedRuntime):
        def generate(self, prompt: EncodedPrompt, deadline: float) -> GeneratedResponse:
            monkeypatch.setattr(time, "monotonic", lambda: deadline)
            return response("A1")

    result = select(decision(), config(), LateRuntime([]))
    assert result.failure.kind == "deadline"
    assert result.attempts[0].response.text == "A1"
    assert result.selected_reference is None


@pytest.mark.parametrize("references", [(), ("A0", "A0"), ("A 0",), ("",)])
def test_invalid_reference_lists_fail_before_inference(
    references: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError):
        DecisionInput(
            text="fixture",
            legal_references=references,
            representation_id="test",
            source_identity={},
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"model_revision": "main"},
        {"tokenizer_revision": "latest"},
        {"max_new_tokens": 100},
        {"deadline_seconds": 0},
        {"deadline_seconds": float("inf")},
        {"enable_thinking": True},
    ],
)
def test_unpinned_or_unbounded_configuration_is_rejected(
    settings: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        config(**settings)
