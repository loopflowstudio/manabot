"""A loopback vLLM completion client preserving the ordinary selection contract.

The campaign owns the server and its pinned environment. Requests send the exact
locally encoded token IDs, never ask the server to apply another chat template,
and require returned prompt/output IDs. HTTP failure is a runtime failure, not
an invalid answer. The outer campaign supervisor owns process-tree cleanup.
"""

import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoConfig, GenerationConfig

from manabot.llm.schema import (
    EncodedPrompt,
    GeneratedResponse,
    InferenceConfig,
    Message,
    RuntimeIdentity,
)
from manabot.llm.tokenization import (
    encode_prompt,
    eos_token_ids,
    generate_response,
    load_tokenizer,
)

SERVICE_PORT = 8123
SERVICE_URL = f"http://127.0.0.1:{SERVICE_PORT}"


class _Choice(BaseModel):
    # Decode retained IDs locally; server-rendered text and stop_reason are not
    # authorities for the answer or EOS boundary.
    model_config = ConfigDict(extra="ignore", strict=True)
    index: int
    finish_reason: str
    token_ids: list[int] = Field(min_length=1)
    prompt_token_ids: list[int]


class _Completion(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    model: str
    choices: list[_Choice] = Field(min_length=1, max_length=1)


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    id: str


class _Models(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    data: list[_Model] = Field(min_length=1, max_length=1)


def verify_service(version: str, *, timeout: float) -> None:
    """Require the single frozen service to acknowledge its behavior route."""
    with urlopen(f"{SERVICE_URL}/v1/models", timeout=timeout) as response:
        models = _Models.model_validate_json(response.read())
    if models.data[0].id != version:
        raise ValueError("vLLM service does not acknowledge the behavior version")


class VLLMRuntime:
    def __init__(self, config: InferenceConfig, *, identity_path: Path) -> None:
        self.config = config
        self.identity = RuntimeIdentity.model_validate_json(identity_path.read_text())
        if self.identity.runtime != "vllm":
            raise ValueError("expected a vLLM runtime identity")
        self.behavior_version = self.identity.behavior_version(config)
        self.tokenizer = load_tokenizer(config, offline=True)
        model_snapshot = snapshot_download(
            config.model_id,
            revision=config.model_revision,
            allow_patterns=["*.json"],
            local_files_only=True,
        )
        # Generation config is optional on the Hub. Match the causal model's
        # config-derived defaults when the publisher supplies no separate file.
        if (Path(model_snapshot) / "generation_config.json").exists():
            generation = GenerationConfig.from_pretrained(
                model_snapshot, local_files_only=True
            )
        else:
            model_config = AutoConfig.from_pretrained(
                model_snapshot, local_files_only=True, trust_remote_code=False
            )
            generation = GenerationConfig.from_model_config(
                model_config.get_text_config()
            )
        self.eos_ids = eos_token_ids(
            generation.eos_token_id, self.tokenizer.eos_token_id
        )
        if self.identity.chat_template != self.tokenizer.get_chat_template():
            raise ValueError(
                "server receipt and pinned tokenizer chat template disagree"
            )
        verify_service(self.behavior_version, timeout=config.deadline_seconds)

    def encode(self, messages: tuple[Message, ...]) -> EncodedPrompt:
        return encode_prompt(self.tokenizer, self.config, messages)

    def generate(self, prompt: EncodedPrompt, deadline: float) -> GeneratedResponse:
        return generate_response(
            lambda tokens, limit: self._complete(tokens, limit, deadline),
            self.tokenizer,
            self.config,
            prompt,
            self.eos_ids,
        )

    def _complete(
        self, token_ids: tuple[int, ...], limit: int, deadline: float
    ) -> tuple[int, ...]:
        """One exact-token request; the reply must bind this prompt and route."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("decision deadline exhausted before server request")
        sampling = self.config.sampling
        decoding: dict[str, float | int] = (
            {"temperature": 0, "top_p": 1, "top_k": -1, "min_p": 0}
            if sampling is None
            else {
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
                "min_p": sampling.min_p,
                "seed": sampling.request_seed(token_ids),
            }
        )
        body = {
            "model": self.behavior_version,
            "prompt": list(token_ids),
            "max_tokens": limit,
            **decoding,
            "repetition_penalty": 1,
            "presence_penalty": 0 if sampling is None else sampling.presence_penalty,
            "frequency_penalty": 0,
            "stop_token_ids": sorted(self.eos_ids),
            "ignore_eos": False,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "add_special_tokens": False,
            "return_token_ids": True,
            "stream": False,
            "n": 1,
        }
        request = Request(
            f"{SERVICE_URL}/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=remaining) as response:
            reply = _Completion.model_validate_json(response.read())
        if time.monotonic() >= deadline:
            raise TimeoutError("decision deadline exhausted during server request")
        if reply.model != self.behavior_version:
            raise ValueError("vLLM response has a different behavior version")
        choice = reply.choices[0]
        if choice.index != 0 or tuple(choice.prompt_token_ids) != token_ids:
            raise ValueError("vLLM response does not bind the exact submitted prompt")
        tokens = tuple(choice.token_ids)
        if len(tokens) > limit or any(token < 0 for token in tokens):
            raise ValueError(
                "vLLM returned invalid token IDs or exceeded the output bound"
            )
        if choice.finish_reason not in ("stop", "length"):
            raise RuntimeError(f"vLLM generation ended with {choice.finish_reason}")
        if (choice.finish_reason == "stop") != (tokens[-1] in self.eos_ids):
            raise ValueError("vLLM finish reason and retained EOS token disagree")
        return tokens

    def peak_gpu_bytes(self) -> None:
        # The client cannot observe the server allocator. Device sampling is separate.
        return None
