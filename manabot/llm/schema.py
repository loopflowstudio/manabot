"""Versioned inputs and evidence for one text-only LLM decision.

The caller owns observation meaning and legal-reference binding. These records
retain that input verbatim; no type here can execute a game action.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class Record(BaseModel):
    # Importing infra's shared base also loads its training runtime and Torch.
    model_config = ConfigDict(extra="forbid", frozen=True)


class DecisionInput(Record):
    text: str = Field(min_length=1)
    legal_references: tuple[str, ...] = Field(min_length=1)
    representation_id: str = Field(min_length=1)
    # Opaque provenance from the renderer, or an explicitly named synthetic fixture.
    source_identity: dict[str, str]

    @model_validator(mode="after")
    def validate_references(self) -> "DecisionInput":
        refs = self.legal_references
        if len(set(refs)) != len(refs):
            raise ValueError("legal references must be unique")
        if any(not ref or any(c.isspace() for c in ref) for ref in refs):
            raise ValueError("legal references must be nonempty single tokens")
        return self


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
    decoding: Literal["greedy"] = "greedy"
    # None preserves the tokenizer's default; reasoning output is not supported.
    enable_thinking: Literal[False] | None = None

    @model_validator(mode="after")
    def validate_budget(self) -> "InferenceConfig":
        if self.max_new_tokens >= self.context_tokens:
            raise ValueError("context must leave room for the prompt")
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


class RuntimeIdentity(Record):
    runtime: str
    runtime_version: str
    torch_version: str
    python_version: str
    device_name: str
    chat_template: str
    generation_config: dict[str, JsonValue]
    model_class: str | None = None


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


class Failure(Record):
    kind: Literal["overflow", "runtime", "deadline"]
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
    attempts: tuple[Attempt, ...] = ()
    selected_reference: str | None = None
    origin: Literal["initial", "retry", "fallback"] | None = None
    fallback: FallbackEvidence | None = None
    failure: Failure | None = None
    seconds: float = 0
    peak_gpu_bytes: int | None = None
