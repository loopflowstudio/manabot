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
    EncodedPrompt,
    Failure,
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


def select(
    decision: DecisionInput,
    config: InferenceConfig,
    runtime: Runtime,
    *,
    started: float | None = None,
    progress: Callable[[SelectionResult], None] | None = None,
) -> SelectionResult:
    """Return evidence for success or failure; only invalid answers get recovery.

    ``started`` is a monotonic timestamp including runtime loading, when supplied.
    Progress snapshots include the pending prompt before generation begins.
    """
    start = time.monotonic() if started is None else started
    deadline = start + config.deadline_seconds
    result = SelectionResult(decision=decision, config=config, runtime=runtime.identity)
    attempts: list[Attempt] = []
    messages = (
        Message(
            role="user",
            content=(
                decision.text
                + "\n\nLegal action references:\n"
                + "\n".join(decision.legal_references)
                + "\n\nReturn exactly one listed action reference, with no other text."
            ),
        ),
    )

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
            prompt = runtime.encode(messages)
            attempt = attempt.model_copy(
                update={"prompt": prompt, "prompt_tokens": len(prompt.token_ids)}
            )
            attempts[-1] = attempt
            publish()
            check_deadline()
            if len(prompt.token_ids) + config.max_new_tokens > config.context_tokens:
                raise ContextOverflow(
                    "complete prompt plus generation exceeds context_tokens"
                )
            response = runtime.generate(prompt, deadline)
            reference = response.text.strip()
            error = None
            if response.finish_reason != "eos":
                error = "incomplete response: generation limit reached"
            elif reference not in decision.legal_references:
                error = "expected exactly one listed action reference"
            peak = runtime.peak_gpu_bytes()
            attempt = attempt.model_copy(
                update={
                    "response": response,
                    "output_tokens": len(response.token_ids),
                    "validation_error": error,
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
                publish()
                return result
            messages = (
                *messages,
                Message(role="assistant", content=response.text),
                Message(role="user", content=REPAIR),
            )
        check_deadline()
        rng = random.Random(config.fallback_seed)
        state = repr(rng.getstate())
        selected_index = rng.randrange(len(decision.legal_references))
        result = result.model_copy(
            update={
                "selected_reference": decision.legal_references[selected_index],
                "origin": "fallback",
                "fallback": FallbackEvidence(
                    seed=config.fallback_seed,
                    state_before=state,
                    selected_index=selected_index,
                ),
            }
        )
    except Exception as error:
        kind = (
            "deadline"
            if isinstance(error, TimeoutError)
            else "overflow"
            if isinstance(error, ContextOverflow)
            else "runtime"
        )
        if attempts:
            attempts[-1] = attempts[-1].model_copy(
                update={"seconds": time.monotonic() - attempt_start}
            )
        result = result.model_copy(
            update={
                "failure": Failure(
                    kind=kind, message=f"{type(error).__name__}: {error}"
                ),
            }
        )
    publish()
    return result
