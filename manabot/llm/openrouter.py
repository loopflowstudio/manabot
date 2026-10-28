"""Hosted text decisions with OpenRouter usage receipts and bounded recovery.

OpenRouterSession implements the game harness's synchronous decision boundary;
async HTTP permits one total deadline across both attempts. The service owns
tokenization and context admission: no local token IDs or template are invented.
Chat selects only complete final content; Decisions selects the typed choice.
Provider errors never fall back. Both transports share dispatch and accounting.
"""

import asyncio
from collections.abc import Callable
import hashlib
import json
import os
import time
from typing import Annotated, Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from manabot.llm.budget import BudgetLedger
from manabot.llm.schema import (
    DecisionInput,
    DecisionObjective,
    Failure,
    FallbackEvidence,
    Message,
    Record,
)
from manabot.llm.selection import REPAIR, decision_messages, uniform_fallback

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DECISIONS_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
WIN_QUESTION = (
    "You are playing Magic: The Gathering as the player labeled You. "
    "Which legal action maximizes your probability of winning this game?"
)


class OpenRouterConfig(Record):
    transport: Literal["chat", "decisions"] = "chat"
    model_id: str = Field(min_length=1, pattern=r"^[^/\s]+/[^\s]+$")
    # Explicit treatment: old receipts retain their protocol-only instruction.
    objective: DecisionObjective = "legal-action"
    max_output_tokens: int = Field(default=4096, gt=0, strict=True)
    deadline_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    fallback_seed: int = Field(default=37, strict=True)
    reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh", "max"
    ] = "low"
    # Explicit opt-in: the current renderer has little reusable prefix between
    # decisions, and cache writes can cost more than ordinary input tokens.
    cache_prompt: bool = False
    provider: str | None = Field(default=None, min_length=1)
    max_prompt_bytes: int = Field(default=524288, gt=0, strict=True)
    # Resend only after connection establishment failed, when no request bytes
    # reached the provider. Each send has its own reservation; within the deadline.
    connect_retries: int = Field(default=0, ge=0, le=5, strict=True)

    @model_validator(mode="after")
    def decisions_treatment(self) -> "OpenRouterConfig":
        if self.transport == "decisions" and (
            self.objective != "win-game"
            or self.cache_prompt
            or self.provider is not None
        ):
            raise ValueError(
                "Decisions requires win-game and no chat cache/provider options"
            )
        return self


class TokenDetails(BaseModel):
    cached_tokens: int | None = Field(default=None, ge=0, strict=True)
    cache_write_tokens: int | None = Field(default=None, ge=0, strict=True)
    reasoning_tokens: int | None = Field(default=None, ge=0, strict=True)


class Usage(BaseModel):
    """Provider-native counts; completion_tokens already includes reasoning."""

    prompt_tokens: int = Field(ge=0, strict=True)
    completion_tokens: int = Field(ge=0, strict=True)
    total_tokens: int = Field(ge=0, strict=True)
    cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    prompt_tokens_details: TokenDetails | None = None
    completion_tokens_details: TokenDetails | None = None


class _ReplyMessage(BaseModel):
    content: str | None
    refusal: str | None = None


class _Choice(BaseModel):
    message: _ReplyMessage
    finish_reason: str


class Completion(BaseModel):
    # Ignore future service fields at this boundary, but validate every field
    # used for selection/accounting. Never persist reasoning text or credentials.
    model_config = ConfigDict(extra="ignore")
    id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider: str | None = None
    choices: tuple[_Choice, ...] = Field(min_length=1, max_length=1)
    usage: Usage | None = None


class DecisionUsage(BaseModel):
    input_tokens: int = Field(ge=0, strict=True)
    output_tokens: int = Field(ge=0, strict=True)
    cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)


Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability


class DecisionReply(BaseModel):
    id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider: str | None = None
    answers: dict[str, ChoiceAnswer]
    usage: DecisionUsage | None = None

    def select_reference(self, legal_references: tuple[str, ...]) -> str:
        """Validate the captured choice set after retaining and settling usage."""
        if set(self.answers) != {"action"}:
            raise RuntimeError("Decisions reply has unexpected questions")
        answer = self.answers["action"]
        if answer.choice not in legal_references or set(answer.probabilities) != set(
            legal_references
        ):
            raise RuntimeError("Decisions reply does not cover the captured choices")
        return answer.choice


class ChoiceQuestion(Record):
    type: Literal["choice"] = "choice"
    instructions: str = WIN_QUESTION
    criteria: dict[str, str] = Field(min_length=1)


class DecisionRequest(Record):
    model: str
    state: str
    questions: dict[str, ChoiceQuestion]


class HostedAttempt(Record):
    messages: tuple[Message, ...] = ()
    request: DecisionRequest | None = None
    decision_reply: DecisionReply | None = None
    reservation_id: str | None = None
    # Historical attempts were dispatched; new preflight failures are explicit.
    dispatched: bool = True
    # Connection never established; its reservation stays counted, abandoned.
    connect_failed: bool = False
    completion: Completion | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    validation_error: str | None = None
    seconds: float | None = None

    @property
    def usage(self) -> Usage | DecisionUsage | None:
        if self.decision_reply is not None:
            return self.decision_reply.usage
        return self.completion.usage if self.completion is not None else None

    def with_response(
        self, content: bytes, legal_references: tuple[str, ...], started: float
    ) -> "HostedAttempt":
        """Retain the native envelope and normalized counters before settlement.

        Invalid chat answers remain eligible for protocol repair. Typed answer
        validation happens after retention so its reported charge is not lost.
        """
        update: dict[str, object] = {"seconds": time.monotonic() - started}
        try:
            if self.request is not None:
                reply = DecisionReply.model_validate_json(content)
                update["decision_reply"] = reply
                if reply.usage is not None:
                    update["prompt_tokens"] = reply.usage.input_tokens
                    update["output_tokens"] = reply.usage.output_tokens
            else:
                completion = Completion.model_validate_json(content)
                update["completion"] = completion
                if completion.usage is not None:
                    update["prompt_tokens"] = completion.usage.prompt_tokens
                    update["output_tokens"] = completion.usage.completion_tokens
                choice = completion.choices[0]
                if choice.finish_reason != "stop":
                    update["validation_error"] = (
                        f"incomplete response: {choice.finish_reason}"
                    )
                elif (choice.message.content or "").strip() not in legal_references:
                    update["validation_error"] = (
                        "expected exactly one listed action reference"
                    )
        except ValidationError:
            name = "Decisions" if self.request is not None else "completion"
            raise RuntimeError(f"invalid OpenRouter {name} envelope") from None
        return self.model_copy(update=update)


class OpenRouterIdentity(Record):
    runtime: Literal["openrouter-chat-v1", "openrouter-decisions-v1"] = (
        "openrouter-chat-v1"
    )
    endpoint: Literal[
        "https://openrouter.ai/api/v1/chat/completions",
        "https://openrouter.ai/api/alpha/decisions",
    ] = ENDPOINT
    session_id: str
    config: OpenRouterConfig
    # Hosted model slugs aren't immutable checkpoint revisions.
    model_revision: None = None
    tokenizer: Literal["provider-owned"] = "provider-owned"


class OpenRouterResult(Record):
    version: Literal["openrouter-decision-v1"] = "openrouter-decision-v1"
    decision: DecisionInput
    config: OpenRouterConfig
    runtime: OpenRouterIdentity
    attempts: tuple[HostedAttempt, ...] = ()
    selected_reference: str | None = None
    origin: Literal["initial", "retry", "fallback"] | None = None
    fallback: FallbackEvidence | None = None
    failure: Failure | None = None
    seconds: float = 0
    peak_gpu_bytes: None = None


_UNSENT = {
    "dispatched": False,
    "reservation_id": None,
    "connect_failed": False,
    "seconds": None,
}


class OpenRouterSession:
    """Independent observations; only a repair carries the previous answer.

    A session ID is an opaque routing hint, not model memory. No HTTP retries or
    provider fallback are enabled. A timed-out request may still incur charges;
    its missing usage remains unknown and the session closes after any failure.
    """

    def __init__(
        self,
        config: OpenRouterConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        budget: BudgetLedger | None = None,
        budget_owner: str = "standalone",
    ) -> None:
        self.config = config
        typed = config.transport == "decisions"
        self.identity = OpenRouterIdentity(
            session_id=str(uuid4()),
            config=config,
            runtime="openrouter-decisions-v1" if typed else "openrouter-chat-v1",
            endpoint=DECISIONS_ENDPOINT if typed else ENDPOINT,
        )
        self.load_seconds = 0.0
        self.closed = False
        self._transport = transport
        self._budget = budget
        self._budget_owner = budget_owner

    def decide(
        self,
        decision: DecisionInput,
        *,
        progress: Callable[[OpenRouterResult], None] | None = None,
    ) -> OpenRouterResult:
        if self.closed:
            raise RuntimeError("inference session is closed")
        return asyncio.run(self._decide(decision, progress))

    async def _decide(
        self,
        decision: DecisionInput,
        progress: Callable[[OpenRouterResult], None] | None,
    ) -> OpenRouterResult:
        started = time.monotonic()
        result = OpenRouterResult(
            decision=decision, config=self.config, runtime=self.identity
        )
        attempts: list[HostedAttempt] = []
        typed = self.config.transport == "decisions"
        messages = (
            ()
            if typed
            else decision_messages(decision, objective=self.config.objective)
        )

        def publish() -> None:
            nonlocal result
            result = result.model_copy(
                update={
                    "attempts": tuple(attempts),
                    "seconds": time.monotonic() - started,
                }
            )
            if progress is not None:
                progress(result)

        attempt_start = started
        try:
            key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is required")
            typed_request = self.decision_request(decision) if typed else None
            async with (
                asyncio.timeout(self.config.deadline_seconds),
                httpx.AsyncClient(
                    transport=self._transport,
                    follow_redirects=False,
                    timeout=self.config.deadline_seconds,
                ) as client,
            ):
                for origin in ("initial",) if typed else ("initial", "retry"):
                    attempt_start = time.monotonic()
                    attempts.append(
                        HostedAttempt(
                            messages=messages,
                            request=typed_request,
                            dispatched=False,
                        )
                    )
                    publish()
                    body = (
                        typed_request.model_dump()
                        if typed_request
                        else self.request(messages)
                    )
                    # Check, identify and send the same bytes.
                    payload = json.dumps(
                        body,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode()
                    if len(payload) > self.config.max_prompt_bytes:
                        raise OverflowError("complete request exceeds max_prompt_bytes")
                    for connect_retry in range(self.config.connect_retries + 1):
                        if connect_retry:
                            attempt_start = time.monotonic()
                            attempts.append(attempts[-1].model_copy(update=_UNSENT))
                        reservation = None
                        if self._budget is not None:
                            digest = hashlib.sha256(payload).hexdigest()
                            reservation = self._budget.reserve(
                                self._budget_owner, digest
                            )
                        attempts[-1] = attempts[-1].model_copy(
                            update={
                                "dispatched": True,
                                "reservation_id": reservation,
                            }
                        )
                        publish()
                        try:
                            response = await client.post(
                                self.identity.endpoint,
                                content=payload,
                                headers={
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {key}",
                                    "X-OpenRouter-Title": "Manabot",
                                },
                            )
                            break
                        except (httpx.ConnectError, httpx.ConnectTimeout):
                            if reservation is not None:
                                self._budget.abandon(reservation)
                            attempts[-1] = attempts[-1].model_copy(
                                update={
                                    "connect_failed": True,
                                    "seconds": time.monotonic() - attempt_start,
                                }
                            )
                            publish()
                            if connect_retry == self.config.connect_retries:
                                raise
                            await asyncio.sleep(2**connect_retry)
                    # Error bodies can echo request/auth data. Retain status,
                    # not the service's untrusted diagnostic text.
                    if response.status_code != 200:
                        raise RuntimeError(f"OpenRouter HTTP {response.status_code}")
                    attempt = attempts[-1].with_response(
                        response.content,
                        decision.legal_references,
                        attempt_start,
                    )
                    attempts[-1] = attempt
                    publish()
                    # Settle the retained response before accepting any action.
                    if reservation is not None:
                        usage = attempt.usage
                        self._budget.settle(
                            reservation, usage.cost if usage is not None else None
                        )
                    if time.monotonic() - started >= self.config.deadline_seconds:
                        raise TimeoutError("decision deadline exhausted")
                    if attempt.decision_reply is not None:
                        reference = attempt.decision_reply.select_reference(
                            decision.legal_references
                        )
                    else:
                        assert attempt.completion is not None
                        choice = attempt.completion.choices[0]
                        text = choice.message.content or ""
                        if (
                            choice.finish_reason not in ("stop", "length")
                            or choice.message.refusal
                        ):
                            raise RuntimeError(
                                "provider did not produce an ordinary completion"
                            )
                        if attempt.validation_error is not None:
                            messages = (
                                *messages,
                                Message(role="assistant", content=text),
                                Message(role="user", content=REPAIR),
                            )
                            continue
                        reference = text.strip()
                    result = result.model_copy(
                        update={"selected_reference": reference, "origin": origin}
                    )
                    break
                else:
                    fallback = uniform_fallback(decision, self.config.fallback_seed)
                    result = result.model_copy(
                        update={
                            "selected_reference": decision.legal_references[
                                fallback.selected_index
                            ],
                            "origin": "fallback",
                            "fallback": fallback,
                        }
                    )
        except Exception as error:
            self.close()
            # The failed request stays counted at its full bound.
            if self._budget is not None and attempts and attempts[-1].reservation_id:
                self._budget.abandon(attempts[-1].reservation_id)
            if attempts:
                attempts[-1] = attempts[-1].model_copy(
                    update={"seconds": time.monotonic() - attempt_start}
                )
            kind = (
                "deadline"
                if isinstance(error, (TimeoutError, httpx.TimeoutException))
                else "overflow"
                if isinstance(error, OverflowError)
                else "runtime"
            )
            # Transport exceptions may include proxy credentials or request data.
            message = (
                f"OpenRouter transport failure: {type(error).__name__}"
                if isinstance(error, httpx.HTTPError)
                else str(error)
            )
            if kind == "deadline":
                message = "OpenRouter decision deadline exhausted"
            result = result.model_copy(
                update={
                    "failure": Failure(kind=kind, message=message),
                    "origin": None,
                    "selected_reference": None,
                    "fallback": None,
                }
            )
        publish()
        return result

    def decision_request(self, decision: DecisionInput) -> DecisionRequest:
        """Send every captured choice, with its native renderer's description.

        No shortlist, hierarchy, invented option or probability-based override.
        Provider-owned tokenization rejects context overflow; the byte guard is
        only a local payload limit and is never used as a billing token bound.
        """
        if len(decision.legal_references) > 255:
            raise OverflowError(
                "Decisions supports at most 255 choices; cannot truncate"
            )
        if decision.action_descriptions is None:
            raise ValueError("Decisions requires captured action descriptions")
        return DecisionRequest(
            model=self.config.model_id,
            state=decision.text,
            questions={
                "action": ChoiceQuestion(
                    criteria={
                        ref: f"{ref}: {decision.action_descriptions[ref]}"
                        for ref in decision.legal_references
                    }
                )
            },
        )

    def request(self, messages: tuple[Message, ...]) -> dict[str, object]:
        """Serialize the exact full messages, optionally caching the initial one.

        The first block is unchanged during repair. No automatic truncation or
        accumulated game transcript is permitted. Cache support is provider-owned.
        """
        wire_messages: list[dict[str, object]] = []
        for index, message in enumerate(messages):
            if index == 0 and self.config.cache_prompt:
                content: object = [
                    {
                        "type": "text",
                        "text": message.content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                content = message.content
            wire_messages.append({"role": message.role, "content": content})
        provider: dict[str, object] = {
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        if self.config.provider:
            provider["only"] = [self.config.provider]
        body: dict[str, object] = {
            "model": self.config.model_id,
            "messages": wire_messages,
            "stream": False,
            "max_tokens": self.config.max_output_tokens,
            "reasoning": {"effort": self.config.reasoning_effort, "exclude": True},
            "provider": provider,
            "session_id": self.identity.session_id,
            "transforms": [],
        }
        if self.config.model_id.startswith("openai/"):
            # Explicit mode without a breakpoint disables automatic cache writes.
            body["prompt_cache_options"] = {"mode": "explicit"}
        return body

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "OpenRouterSession":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
