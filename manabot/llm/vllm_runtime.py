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
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from manabot.llm.cache import TOKENIZER_PATTERNS
from manabot.llm.schema import (
    EncodedPrompt,
    GeneratedResponse,
    InferenceConfig,
    Message,
    RuntimeIdentity,
)
from manabot.llm.tokenization import decode_response, encode_prompt


class _Choice(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    index: int
    text: str
    finish_reason: str
    stop_reason: int | str | None = None
    token_ids: list[int] = Field(min_length=1)
    prompt_token_ids: list[int]


class _Completion(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    choices: list[_Choice] = Field(min_length=1, max_length=1)


class VLLMRuntime:
    def __init__(
        self, config: InferenceConfig, *, port: int = 8123, identity_path: Path
    ) -> None:
        self.config = config
        self.url = f"http://127.0.0.1:{port}/v1/completions"
        self.identity = RuntimeIdentity.model_validate_json(identity_path.read_text())
        snapshot = snapshot_download(
            config.tokenizer_id,
            revision=config.tokenizer_revision,
            allow_patterns=TOKENIZER_PATTERNS,
            local_files_only=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
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
        eos = generation.eos_token_id
        self.eos_ids = (
            set() if eos is None else {eos} if isinstance(eos, int) else set(eos)
        )
        if self.tokenizer.eos_token_id is not None:
            self.eos_ids.add(self.tokenizer.eos_token_id)
        if not self.eos_ids:
            raise ValueError("model/tokenizer must define an EOS token")
        if self.identity.chat_template != self.tokenizer.get_chat_template():
            raise ValueError(
                "server receipt and pinned tokenizer chat template disagree"
            )

    def encode(self, messages: tuple[Message, ...]) -> EncodedPrompt:
        return encode_prompt(self.tokenizer, self.config, messages)

    def generate(self, prompt: EncodedPrompt, deadline: float) -> GeneratedResponse:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("decision deadline exhausted before server request")
        body = {
            "model": self.config.model_id,
            "prompt": list(prompt.token_ids),
            "max_tokens": self.config.max_new_tokens,
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "min_p": 0,
            "repetition_penalty": 1,
            "presence_penalty": 0,
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
            self.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=remaining) as response:
            reply = _Completion.model_validate_json(response.read())
        choice = reply.choices[0]
        if choice.index != 0 or tuple(choice.prompt_token_ids) != prompt.token_ids:
            raise ValueError("vLLM response does not bind the exact submitted prompt")
        tokens = tuple(choice.token_ids)
        if len(tokens) > self.config.max_new_tokens or any(
            token < 0 for token in tokens
        ):
            raise ValueError(
                "vLLM returned invalid token IDs or exceeded the output bound"
            )
        decoded = decode_response(
            self.tokenizer, tokens, self.eos_ids, self.config.enable_thinking
        )
        if choice.finish_reason not in ("stop", "length"):
            raise RuntimeError(f"vLLM generation ended with {choice.finish_reason}")
        if (choice.finish_reason == "stop") != (decoded.finish_reason == "eos"):
            raise ValueError("vLLM finish reason and retained EOS token disagree")
        return decoded

    def peak_gpu_bytes(self) -> None:
        # The client cannot observe the server allocator. Device sampling is separate.
        return None
