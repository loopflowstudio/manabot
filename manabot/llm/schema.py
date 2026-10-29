"""Versioned inputs and evidence for one text-only LLM decision.

The caller owns observation meaning and legal-reference binding. These records
retain that input verbatim; no type here can execute a game action.
"""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

DecisionObjective = Literal["legal-action", "win-game"]


class Record(BaseModel):
    # Importing infra's shared base also loads its training runtime and Torch.
    model_config = ConfigDict(extra="forbid", frozen=True)


class DecisionInput(Record):
    text: str = Field(min_length=1)
    legal_references: tuple[str, ...] = Field(min_length=1)
    representation_id: str = Field(min_length=1)
    # Opaque provenance from the renderer, or an explicitly named synthetic fixture.
    source_identity: dict[str, str]
    # Exact renderer descriptions for typed-choice transports. Old text-only
    # receipts omit this field and retain their original meaning.
    action_descriptions: dict[str, str] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_references(self) -> "DecisionInput":
        refs = self.legal_references
        if len(set(refs)) != len(refs):
            raise ValueError("legal references must be unique")
        if any(not ref or any(c.isspace() for c in ref) for ref in refs):
            raise ValueError("legal references must be nonempty single tokens")
        if self.action_descriptions is not None and (
            set(self.action_descriptions) != set(refs)
            or any(not value.strip() for value in self.action_descriptions.values())
        ):
            raise ValueError("descriptions must cover exactly every legal reference")
        return self


class Sampling(Record):
    """Explicit stochastic decoding; absent fields never inherit model defaults.

    Each request derives its seed from `seed` and the exact prompt token IDs, so
    a repeated prompt repeats its draw. Batched GPU numerics can still differ;
    the retained tokens, not the seed, are the authority for what was played.
    """

    temperature: float = Field(gt=0, allow_inf_nan=False)
    top_p: float = Field(gt=0, le=1, allow_inf_nan=False)
    top_k: int = Field(ge=1, strict=True)
    min_p: float = Field(default=0, ge=0, lt=1, allow_inf_nan=False)
    presence_penalty: float = Field(default=0, ge=-2, le=2, allow_inf_nan=False)
    seed: int = Field(ge=0, strict=True)

    def request_seed(self, prompt_token_ids: tuple[int, ...]) -> int:
        payload = json.dumps([self.seed, list(prompt_token_ids)]).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8]) >> 1


class InferenceConfig(Record):
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_id: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    device: Literal["cpu", "cuda"]
    precision: Literal["float32", "float16", "bfloat16"]
    context_tokens: int = Field(gt=0, strict=True)
    max_new_tokens: int = Field(gt=0, strict=True)
    deadline_seconds: float = Field(gt=0, allow_inf_nan=False)
    fallback_seed: int = Field(strict=True)
    decoding: Literal["greedy", "sampled"] = "greedy"
    sampling: Sampling | None = None
    # Explicit thinking uses Qwen's closing </think> delimiter; the final answer
    # alone must be one legal reference. None preserves the historical protocol.
    enable_thinking: bool | None = None
    # Budget forcing: after this many generated tokens without a closing
    # delimiter, the harness inserts one and the model writes its answer within
    # the rest of max_new_tokens. Each response records the forced close.
    thinking_budget: int | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode="after")
    def validate_budget(self) -> "InferenceConfig":
        if self.max_new_tokens >= self.context_tokens:
            raise ValueError("context must leave room for the prompt")
        if (self.decoding == "sampled") != (self.sampling is not None):
            raise ValueError("sampled decoding requires exactly its sampling settings")
        if self.thinking_budget is not None and (
            self.enable_thinking is not True
            or self.thinking_budget >= self.max_new_tokens
        ):
            raise ValueError(
                "a thinking budget requires enabled thinking and answer tokens"
            )
        return self


class Message(Record):
    role: Literal["user", "assistant"]
    content: str


class EncodedPrompt(Record):
    text: str
    token_ids: tuple[int, ...]


class GeneratedResponse(Record):
    # Decode without hiding special tokens, except the final EOS terminator.
    text: str
    token_ids: tuple[int, ...]
    finish_reason: Literal["eos", "length"]
    reasoning_tokens: int | None = Field(default=None, ge=0)
    answer_tokens: int | None = Field(default=None, ge=0)
    # Budget forcing: token_ids[forced_close_at:][:len(inserted_token_ids)] were
    # inserted by the harness, not generated; every other token was generated.
    forced_close_at: int | None = Field(default=None, ge=0)
    inserted_token_ids: tuple[int, ...] = ()


class RuntimeIdentity(Record):
    runtime: str
    runtime_version: str
    torch_version: str
    python_version: str
    device_name: str
    chat_template: str
    generation_config: dict[str, JsonValue]
    model_class: str | None = None

    def behavior_version(self, config: InferenceConfig) -> str:
        """Bind model, tokenizer and treatment to the actual runtime and template."""
        # Keep receipt identities usable without infra's training/runtime imports.
        payload = json.dumps(
            {
                "inference": config.model_dump(mode="json"),
                "runtime": self.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class Attempt(Record):
    messages: tuple[Message, ...]
    prompt: EncodedPrompt | None = None
    response: GeneratedResponse | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    validation_error: str | None = None
    # Null in a pending snapshot; never report unfinished inference as zero latency.
    seconds: float | None = None
    peak_gpu_bytes: int | None = None
    # Worker monotonic host intervals, nested within seconds. Generation includes
    # input/output transfer and decoding; it is not device-only prefill/decode.
    encode_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    generation_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    validation_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)


FailureKind = Literal["overflow", "runtime", "deadline"]


class Failure(Record):
    kind: FailureKind
    message: str


class FallbackEvidence(Record):
    algorithm: Literal["python-random-randrange-v1"] = "python-random-randrange-v1"
    seed: int
    state_before: str
    selected_index: int


class SelectionResult(Record):
    version: Literal["llm-decision-v1"] = "llm-decision-v1"
    decision: DecisionInput
    config: InferenceConfig
    runtime: RuntimeIdentity | None = None
    # Historical receipts predate versioned admission; absence is not a version.
    behavior_version: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    attempts: tuple[Attempt, ...] = ()
    selected_reference: str | None = None
    origin: Literal["initial", "retry", "fallback"] | None = None
    fallback: FallbackEvidence | None = None
    failure: Failure | None = None
    seconds: float = 0
    peak_gpu_bytes: int | None = None

    def invalidate(
        self, error: Exception, *, kind: FailureKind | None = None
    ) -> "SelectionResult":
        """Retain evidence but revoke the action when selection or supervision fails.

        Without an explicit kind, timeouts are deadline failures and all other
        errors are runtime failures.
        """
        if kind is None:
            kind = "deadline" if isinstance(error, TimeoutError) else "runtime"
        return self.model_copy(
            update={
                "selected_reference": None,
                "origin": None,
                "fallback": None,
                "failure": Failure(
                    kind=kind, message=f"{type(error).__name__}: {error}"
                ),
            }
        )
