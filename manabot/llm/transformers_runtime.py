"""Pinned, untouched Transformers causal-LM inference for the text harness.

Only the worker imports this module. Hub snapshots bind both sets of bytes to
explicit commit IDs; remote Python code, adapters and training are not enabled.
"""

import platform
import time

from huggingface_hub import snapshot_download
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from manabot.llm.schema import (
    EncodedPrompt,
    GeneratedResponse,
    InferenceConfig,
    Message,
    RuntimeIdentity,
)


class TransformersRuntime:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        if config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.reset_peak_memory_stats()
        # Resolve the Hub repository explicitly, even if a same-named local path exists.
        model_path = snapshot_download(
            config.model_id,
            revision=config.model_revision,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
        )
        tokenizer_path = snapshot_download(
            config.tokenizer_id,
            revision=config.tokenizer_revision,
            allow_patterns=["*.json", "*.jinja", "*.txt", "*.model"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.template = self.tokenizer.get_chat_template()
        model_config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        text_config = model_config.get_text_config()
        if any(
            getattr(candidate, "quantization_config", None) is not None
            for candidate in (model_config, text_config)
        ):
            raise ValueError(
                "quantized checkpoints require a separately declared treatment"
            )
        if text_config.is_encoder_decoder:
            raise ValueError("the harness requires a decoder-only instruction model")
        model_limit = getattr(text_config, "max_position_embeddings", None)
        if model_limit is not None and config.context_tokens > model_limit:
            raise ValueError("context_tokens exceeds the model's declared context")
        # Transformers dispatches composite checkpoints such as Qwen3.5 to their
        # text-only causal class and maps the original language-model weights.
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                config=model_config,
                local_files_only=True,
                trust_remote_code=False,
                dtype=getattr(torch, config.precision),
                use_safetensors=True,
            )
            .to(config.device)
            .eval()
        )
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = (
            set() if eos is None else {eos} if isinstance(eos, int) else set(eos)
        )
        # Chat tokenizers can declare a message terminator distinct from the
        # decoder's end-of-text token (Qwen3.5 does). Both end the answer;
        # retain other special tokens so they still fail the exact-ref protocol.
        if self.tokenizer.eos_token_id is not None:
            self.eos_ids.add(self.tokenizer.eos_token_id)
        if not self.eos_ids:
            raise ValueError("model/tokenizer must define an EOS token")
        pad = self.tokenizer.pad_token_id
        # Use explicit greedy settings instead of inheriting a model's sampling,
        # repetition penalty or beam-search defaults. Retain all effective values.
        self.generation = GenerationConfig(
            do_sample=False,
            num_beams=1,
            max_new_tokens=config.max_new_tokens,
            eos_token_id=sorted(self.eos_ids),
            pad_token_id=pad if pad is not None else min(self.eos_ids),
            bos_token_id=self.tokenizer.bos_token_id,
            use_cache=True,
        )
        self.identity = RuntimeIdentity(
            runtime="transformers",
            runtime_version=transformers.__version__,
            torch_version=torch.__version__,
            python_version=platform.python_version(),
            device_name=(
                torch.cuda.get_device_name() if config.device == "cuda" else "cpu"
            ),
            chat_template=self.template,
            generation_config=self.generation.to_dict(),
            model_class=type(self.model).__name__,
        )

    def encode(self, messages: tuple[Message, ...]) -> EncodedPrompt:
        text = self.tokenizer.apply_chat_template(
            [message.model_dump() for message in messages],
            chat_template=self.template,
            tokenize=False,
            add_generation_prompt=True,
            **(
                {"enable_thinking": self.config.enable_thinking}
                if self.config.enable_thinking is not None
                else {}
            ),
        )
        # The chat template already supplies its special tokens. Never truncate.
        tokens = self.tokenizer(text, add_special_tokens=False, truncation=False)[
            "input_ids"
        ]
        return EncodedPrompt(text=text, token_ids=tuple(tokens))

    def generate(self, prompt: EncodedPrompt, deadline: float) -> GeneratedResponse:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("decision deadline exhausted before generation")
        inputs = torch.tensor(
            [prompt.token_ids], dtype=torch.long, device=self.config.device
        )
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=self.generation,
                max_time=remaining,
            )
        tokens = tuple(int(token) for token in output[0, inputs.shape[1] :].tolist())
        ended = bool(tokens) and tokens[-1] in self.eos_ids
        text = self.tokenizer.decode(
            tokens[:-1] if ended else tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        reasoning_tokens = answer_tokens = None
        if self.config.enable_thinking is True:
            boundary = self.tokenizer.encode("</think>", add_special_tokens=False)
            if len(boundary) != 1:
                raise ValueError(
                    "thinking requires a single-token Qwen closing delimiter"
                )
            content = tokens[:-1] if ended else tokens
            if content.count(boundary[0]) == 1:
                split = content.index(boundary[0])
                reasoning_tokens = split
                answer_tokens = len(content) - split - 1
            elif boundary[0] not in content:
                reasoning_tokens, answer_tokens = len(content), 0
        return GeneratedResponse(
            text=text,
            token_ids=tokens,
            finish_reason="eos" if ended else "length",
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
        )

    def peak_gpu_bytes(self) -> int | None:
        # CPU has no GPU measurement; zero would incorrectly claim one.
        return (
            torch.cuda.max_memory_allocated() if self.config.device == "cuda" else None
        )
