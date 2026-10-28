"""Select a reference with one protocol-only repair and a marked uniform fallback.

Runtime is a narrow inference boundary. Selection never resolves Commands or
advances a game. The supervised runner enforces the wall deadline even when a
runtime hangs; this function also rejects late results before accepting them.
"""

from collections.abc import Callable
import random
import time
from typing import Protocol

from manabot.llm.schema import (
    Attempt,
    DecisionInput,
    DecisionObjective,
    EncodedPrompt,
    FallbackEvidence,
    GeneratedResponse,
    InferenceConfig,
    Message,
    RuntimeIdentity,
    SelectionResult,
)

REPAIR = (
    "Invalid response. Return exactly one action reference from the "
    "legal-actions list, with no other text."
)


class Runtime(Protocol):
    identity: RuntimeIdentity

    def encode(self, messages: tuple[Message, ...]) -> EncodedPrompt: ...

    def generate(self, prompt: EncodedPrompt, deadline: float) -> GeneratedResponse: ...

    def peak_gpu_bytes(self) -> int | None: ...


class ContextOverflow(ValueError):
    """The complete prompt plus its generation allowance exceeds the budget."""


def _answer(text: str, thinking: bool | None) -> str:
    """Qwen opens thinking in its prompt; require exactly one closing boundary.

    Never search reasoning for a legal reference or accept an unfinished thought.
    Raw text and tokens remain unchanged in the response receipt.
    """
    if thinking is not True:
        return text.strip()
    if text.startswith("<think>"):
        text = text[len("<think>") :]
    if text.count("</think>") != 1 or "<think>" in text:
        return ""
    return text.split("</think>", 1)[1].strip()


def decision_messages(
    decision: DecisionInput,
    *,
    objective: DecisionObjective = "legal-action",
    thinking: bool | None = None,
) -> tuple[Message, ...]:
    """The shared protocol prompt, independent of runtime or chat template."""
    instruction = (
        "You are playing Magic: The Gathering as the player labeled You. "
        "Choose the legal action that maximizes your probability of winning "
        "this game.\n\n"
        if objective == "win-game"
        else ""
    )
    return (
        Message(
            role="user",
            content=(
                instruction
                + decision.text
                + "\n\nLegal action references:\n"
                + "\n".join(decision.legal_references)
                + (
                    "\n\nReason about the decision, then end your thinking and return "
                    "exactly one listed action reference as the final answer, with no other text."
                    if thinking is True
                    else "\n\nReturn exactly one listed action reference, with no other text."
                )
            ),
        ),
    )


def uniform_fallback(decision: DecisionInput, seed: int) -> FallbackEvidence:
    rng = random.Random(seed)
    state = repr(rng.getstate())
    return FallbackEvidence(
        seed=seed,
        state_before=state,
        selected_index=rng.randrange(len(decision.legal_references)),
    )


def select(
    decision: DecisionInput,
    config: InferenceConfig,
    runtime: Runtime,
    *,
    started: float | None = None,
    progress: Callable[[SelectionResult], None] | None = None,
    expected_prompt: EncodedPrompt | None = None,
) -> SelectionResult:
    """Return evidence for success or failure; only invalid answers get recovery.

    ``started`` is a monotonic timestamp including runtime loading, when supplied.
    Progress snapshots include the pending prompt before generation begins.
    """
    start = time.monotonic() if started is None else started
    deadline = start + config.deadline_seconds
    result = SelectionResult(
        decision=decision,
        config=config,
        runtime=runtime.identity,
        behavior_version=runtime.identity.behavior_version(config),
    )
    attempts: list[Attempt] = []
    messages = decision_messages(decision, thinking=config.enable_thinking)

    def publish() -> None:
        nonlocal result
        # Copy the working list so later progress cannot change an earlier snapshot.
        result = result.model_copy(
            update={"attempts": tuple(attempts), "seconds": time.monotonic() - start}
        )
        if progress is not None:
            progress(result)

    def check_deadline() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("decision deadline exhausted")

    attempt_start = start
    try:
        publish()
        for origin in ("initial", "retry"):
            check_deadline()
            attempt_start = time.monotonic()
            attempt = Attempt(messages=messages)
            attempts.append(attempt)
            publish()
            phase_start = time.monotonic()
            try:
                prompt = runtime.encode(messages)
            finally:
                attempt = attempt.model_copy(
                    update={"encode_seconds": time.monotonic() - phase_start}
                )
                attempts[-1] = attempt
            attempt = attempt.model_copy(
                update={"prompt": prompt, "prompt_tokens": len(prompt.token_ids)}
            )
            attempts[-1] = attempt
            publish()
            check_deadline()
            if (
                origin == "initial"
                and expected_prompt is not None
                and prompt != expected_prompt
            ):
                raise ValueError(
                    "retained prompt text/token IDs differ from current encoding"
                )
            if len(prompt.token_ids) + config.max_new_tokens > config.context_tokens:
                raise ContextOverflow(
                    "complete prompt plus generation exceeds context_tokens"
                )
            phase_start = time.monotonic()
            try:
                response = runtime.generate(prompt, deadline)
            finally:
                attempt = attempt.model_copy(
                    update={"generation_seconds": time.monotonic() - phase_start}
                )
                attempts[-1] = attempt
            validation_start = time.monotonic()
            reference = _answer(response.text, config.enable_thinking)
            error = None
            if response.finish_reason != "eos":
                error = "incomplete response: generation limit reached"
            elif reference not in decision.legal_references:
                error = "expected exactly one listed action reference"
            validation_seconds = time.monotonic() - validation_start
            peak = runtime.peak_gpu_bytes()
            attempt = attempt.model_copy(
                update={
                    "response": response,
                    "output_tokens": len(response.token_ids),
                    "validation_error": error,
                    "validation_seconds": validation_seconds,
                    "seconds": time.monotonic() - attempt_start,
                    "peak_gpu_bytes": peak,
                }
            )
            attempts[-1] = attempt
            result = result.model_copy(update={"peak_gpu_bytes": peak})
            publish()
            check_deadline()
            if error is None:
                result = result.model_copy(
                    update={
                        "selected_reference": reference,
                        "origin": origin,
                    }
                )
                break
            messages = (
                *messages,
                Message(role="assistant", content=response.text),
                Message(role="user", content=REPAIR),
            )
        else:
            check_deadline()
            fallback = uniform_fallback(decision, config.fallback_seed)
            result = result.model_copy(
                update={
                    "selected_reference": decision.legal_references[
                        fallback.selected_index
                    ],
                    "origin": "fallback",
                    "fallback": fallback,
                }
            )
        # Every action, including fallback, requires successful evidence retention.
        publish()
    except Exception as error:
        if attempts:
            attempts[-1] = attempts[-1].model_copy(
                update={"seconds": time.monotonic() - attempt_start}
            )
        result = result.invalidate(
            error, kind="overflow" if isinstance(error, ContextOverflow) else None
        )
        publish()
    return result
