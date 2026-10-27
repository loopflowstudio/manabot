"""Hosted text decisions with OpenRouter usage receipts and bounded recovery.

OpenRouterSession implements the game harness's synchronous decision boundary;
async HTTP permits one total deadline across both attempts. The service owns
tokenization and context admission: no local token IDs or template are invented.
Only complete final content can select an action. Provider errors never fall back.
"""

import asyncio
from collections.abc import Callable
import os
import time
from typing import Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


class OpenRouterConfig(Record):
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


class HostedAttempt(Record):
    messages: tuple[Message, ...]
    completion: Completion | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    validation_error: str | None = None
    seconds: float | None = None


class OpenRouterIdentity(Record):
    runtime: Literal["openrouter-chat-v1"] = "openrouter-chat-v1"
    endpoint: Literal["https://openrouter.ai/api/v1/chat/completions"] = ENDPOINT
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
    ) -> None:
        self.config = config
        self.identity = OpenRouterIdentity(session_id=str(uuid4()), config=config)
        self.load_seconds = 0.0
        self.closed = False
        self._transport = transport

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
        messages = decision_messages(decision, objective=self.config.objective)

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
            async with asyncio.timeout(self.config.deadline_seconds):
                async with httpx.AsyncClient(
                    transport=self._transport,
                    follow_redirects=False,
                    timeout=self.config.deadline_seconds,
                ) as client:
                    for origin in ("initial", "retry"):
                        attempt_start = time.monotonic()
                        attempts.append(HostedAttempt(messages=messages))
                        publish()
                        if (
                            sum(len(m.content.encode()) for m in messages)
                            > self.config.max_prompt_bytes
                        ):
                            raise OverflowError(
                                "complete messages exceed max_prompt_bytes"
                            )
                        response = await client.post(
                            ENDPOINT,
                            json=self.request(messages),
                            headers={
                                "Authorization": f"Bearer {key}",
                                "X-OpenRouter-Title": "Manabot",
                            },
                        )
                        # Error bodies can echo request/auth data. Retain status,
                        # not the service's untrusted diagnostic text.
                        if response.status_code != 200:
                            raise RuntimeError(
                                f"OpenRouter HTTP {response.status_code}"
                            )
                        try:
                            completion = Completion.model_validate_json(
                                response.content
                            )
                        except ValidationError:
                            raise RuntimeError(
                                "invalid OpenRouter completion envelope"
                            ) from None
                        choice = completion.choices[0]
                        text = choice.message.content or ""
                        error = None
                        if choice.finish_reason != "stop":
                            error = f"incomplete response: {choice.finish_reason}"
                        elif text.strip() not in decision.legal_references:
                            error = "expected exactly one listed action reference"
                        usage = completion.usage
                        attempts[-1] = attempts[-1].model_copy(
                            update={
                                "completion": completion,
                                "prompt_tokens": usage.prompt_tokens if usage else None,
                                "output_tokens": usage.completion_tokens
                                if usage
                                else None,
                                "validation_error": error,
                                "seconds": time.monotonic() - attempt_start,
                            }
                        )
                        publish()
                        if time.monotonic() - started >= self.config.deadline_seconds:
                            raise TimeoutError("decision deadline exhausted")
                        if (
                            choice.finish_reason not in ("stop", "length")
                            or choice.message.refusal
                        ):
                            raise RuntimeError(
                                "provider did not produce an ordinary completion"
                            )
                        if error is None:
                            result = result.model_copy(
                                update={
                                    "selected_reference": text.strip(),
                                    "origin": origin,
                                }
                            )
                            break
                        messages = (
                            *messages,
                            Message(role="assistant", content=text),
                            Message(role="user", content=REPAIR),
                        )
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
