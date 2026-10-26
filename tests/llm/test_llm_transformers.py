"""Exercise the real optional runtime offline with a tiny local causal model."""

from pathlib import Path
import time
from typing import Literal

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

from manabot.llm import transformers_runtime
from manabot.llm.schema import DecisionInput, InferenceConfig, Message
from manabot.llm.selection import select


@pytest.mark.parametrize("architecture", ["gpt2", "qwen3_5"])
def test_transformers_template_tokens_and_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    architecture: Literal["gpt2", "qwen3_5"],
) -> None:
    tokenizer_impl = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[EOS]": 1,
                "[BOS]": 2,
                "A0": 3,
                "A1": 4,
                "user": 5,
                "assistant": 6,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        unk_token="[UNK]",
        eos_token="[EOS]",
        bos_token="[BOS]",
        chat_template="{{ bos_token }}{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {{ eos_token }}{% endfor %}{% if add_generation_prompt %}assistant {% if enable_thinking is defined and enable_thinking is false %}<think></think>{% endif %}{% endif %}",
    )
    tokenizer.save_pretrained(tmp_path)
    # Random weights test runtime plumbing only, never pretrained behavior.
    with torch.random.fork_rng():
        torch.manual_seed(7)
        if architecture == "gpt2":
            model = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=7,
                    n_positions=256,
                    n_embd=8,
                    n_layer=1,
                    n_head=1,
                    bos_token_id=2,
                    eos_token_id=1,
                )
            )
        else:
            # A composite checkpoint exercises real text-weight extraction, both
            # hybrid layer kinds and cached generation without the 9B download.
            model = Qwen3_5ForConditionalGeneration(
                Qwen3_5Config(
                    text_config=Qwen3_5TextConfig(
                        vocab_size=7,
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=2,
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        head_dim=8,
                        linear_num_key_heads=1,
                        linear_num_value_heads=2,
                        linear_key_head_dim=8,
                        linear_value_head_dim=8,
                        layer_types=["linear_attention", "full_attention"],
                        max_position_embeddings=256,
                        rope_parameters={
                            "rope_type": "default",
                            "rope_theta": 10000,
                            "partial_rotary_factor": 1.0,
                            "mrope_section": [1, 1, 2],
                        },
                        bos_token_id=2,
                        # Qwen's decoder EOS differs from its chat tokenizer EOS.
                        eos_token_id=2,
                    ).to_dict(),
                    vision_config=Qwen3_5VisionConfig(
                        depth=1,
                        hidden_size=16,
                        intermediate_size=32,
                        num_heads=2,
                        out_hidden_size=16,
                        num_position_embeddings=16,
                    ).to_dict(),
                )
            )
    model.save_pretrained(tmp_path)

    def snapshot(repo_id: str, *, revision: str, allow_patterns: list[str]) -> str:
        return str(tmp_path)

    monkeypatch.setattr(transformers_runtime, "snapshot_download", snapshot)
    config = InferenceConfig(
        model_id="offline/tiny",
        model_revision="a" * 40,
        tokenizer_id="offline/tiny",
        tokenizer_revision="b" * 40,
        device="cpu",
        precision="float32",
        context_tokens=256,
        max_new_tokens=4,
        deadline_seconds=30,
        fallback_seed=5,
        enable_thinking=False if architecture == "qwen3_5" else None,
    )
    runtime = transformers_runtime.TransformersRuntime(config)
    messages = (Message(role="user", content="A0 A1"),)
    encoded = runtime.encode(messages)
    expected = tokenizer.apply_chat_template(
        [m.model_dump() for m in messages],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        **({"enable_thinking": False} if architecture == "qwen3_5" else {}),
    )
    assert encoded.token_ids == tuple(expected)
    assert encoded.token_ids.count(2) == 1  # No duplicate BOS from tokenization.
    assert runtime.identity.chat_template == tokenizer.chat_template
    assert runtime.identity.generation_config["do_sample"] is False
    assert runtime.identity.generation_config["max_new_tokens"] == 4
    if architecture == "qwen3_5":
        assert runtime.identity.model_class == "Qwen3_5ForCausalLM"
        assert not hasattr(runtime.model.model, "visual")
        for name, weight in runtime.model.model.state_dict().items():
            torch.testing.assert_close(
                weight, model.model.language_model.state_dict()[name], rtol=0, atol=0
            )
        torch.testing.assert_close(runtime.model.lm_head.weight, model.lm_head.weight)
        assert encoded.text.endswith("<think></think>")
    else:
        assert "<think>" not in encoded.text
    result = select(
        DecisionInput(
            text="A0 A1",
            legal_references=("A0", "A1"),
            representation_id="offline-runtime-test",
            source_identity={"model": "random"},
        ),
        config,
        runtime,
    )
    assert result.failure is None
    assert result.origin in {"initial", "retry", "fallback"}
    assert all(0 < a.output_tokens <= 4 for a in result.attempts)
    assert result.peak_gpu_bytes is None
    assert result.config.enable_thinking == config.enable_thinking
    if architecture == "qwen3_5":
        assert all(a.prompt.text.endswith("<think></think>") for a in result.attempts)

    # Force the tokenizer's EOS through real generation. It must stop at that
    # token and exclude it from answer text even when decoder EOS is different.
    runtime.generation.suppress_tokens = [0, 2, 3, 4, 5, 6]
    response = runtime.generate(encoded, deadline=time.monotonic() + 30)
    assert response.token_ids == (1,)
    assert response.text == ""
    assert response.finish_reason == "eos"
    runtime.generation.suppress_tokens = None

    # The same real tokenizer must preserve an overlong input in its receipt.
    overflow = select(
        DecisionInput(
            text="A0 " * 300,
            legal_references=("A0",),
            representation_id="offline-runtime-test",
            source_identity={},
        ),
        config,
        runtime,
    )
    assert overflow.failure.kind == "overflow"
    assert overflow.attempts[0].prompt_tokens > 256
    assert overflow.attempts[0].response is None

    # Composite configs must enforce the text limit, before loading weights.
    with pytest.raises(ValueError, match="declared context"):
        transformers_runtime.TransformersRuntime(
            config.model_copy(update={"context_tokens": 257})
        )
