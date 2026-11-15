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
    GenerationConfig,
)

from manabot.llm.cache import MODEL_PATTERNS
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


class TransformersRuntime:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        if config.sampling is not None and config.sampling.presence_penalty != 0:
            # Transformers has no presence penalty; never run a nearby substitute.
            raise ValueError(
                "presence_penalty is unsupported by the Transformers runtime"
            )
        if config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.reset_peak_memory_stats()
        # Resolve the Hub repository explicitly, even if a same-named local path exists.
        model_path = snapshot_download(
            config.model_id,
            revision=config.model_revision,
            allow_patterns=MODEL_PATTERNS,
        )
        self.tokenizer = load_tokenizer(config, offline=False)
        template = self.tokenizer.get_chat_template()
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
        self.eos_ids = eos_token_ids(
            self.model.generation_config.eos_token_id, self.tokenizer.eos_token_id
        )
        pad = self.tokenizer.pad_token_id
        # Use explicit decoding settings instead of inheriting a model's sampling,
        # repetition penalty or beam-search defaults. Retain all effective values.
        sampling = config.sampling
        self.generation = GenerationConfig(
            do_sample=sampling is not None,
            **(
                {}
                if sampling is None
                else dict(
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    top_k=sampling.top_k,
                    min_p=sampling.min_p,
                )
            ),
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
            chat_template=template,
            generation_config=self.generation.to_dict(),
            model_class=type(self.model).__name__,
        )

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
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("decision deadline exhausted before generation")
        inputs = torch.tensor([token_ids], dtype=torch.long, device=self.config.device)
        if self.config.sampling is not None:
            # The worker process owns this runtime, so the global generator is
            # reseeded from the exact prompt immediately before each draw.
            torch.manual_seed(self.config.sampling.request_seed(token_ids))
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=self.generation,
                max_new_tokens=limit,
                max_time=remaining,
            )
        return tuple(int(token) for token in output[0, inputs.shape[1] :].tolist())

    def peak_gpu_bytes(self) -> int | None:
        # CPU has no GPU measurement; zero would incorrectly claim one.
        return (
            torch.cuda.max_memory_allocated() if self.config.device == "cuda" else None
        )
