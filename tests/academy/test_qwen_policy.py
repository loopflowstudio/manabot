"""A tiny untrained Qwen proves local adapter execution, not playing strength."""

from pathlib import Path

import pytest

import managym
from managym.academy.server import _parse_game_config
from managym.policy import PolicyRegistry, PolicySpec


@pytest.fixture
def qwen_directory(tmp_path: Path) -> Path:
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    import torch
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tokenizer = Tokenizer(
        models.WordLevel({"A0": 0, "<eos>": 1, "<unk>": 2}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>"
    )
    wrapped.chat_template = "{{ messages[0]['content'] }}"
    wrapped.save_pretrained(tmp_path)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=3,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=8192,
            eos_token_id=1,
            bos_token_id=1,
        )
    )
    # A controlled legal response without a remote model, training or weight claim.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.save_pretrained(tmp_path)
    return tmp_path


def test_local_qwen_executes_without_inventing_probabilities(
    qwen_directory: Path,
) -> None:
    from manabot.sim.qwen_provider import provider

    registry = PolicyRegistry(providers={"qwen": provider})
    key = registry.register(
        PolicySpec("qwen", str(qwen_directory), {"max_new_tokens": 1})
    )
    descriptor = registry.descriptor(key)
    assert not descriptor.supports_search
    assert "model.safetensors" in descriptor.artifacts
    assert "tokenizer.json" in descriptor.artifacts
    config = _parse_game_config({"villain_type": "random"})
    env = managym.Env(seed=42)
    observation, _ = env.reset(config.to_rust())
    runner = registry.load(
        key, config.to_rust(), seat=int(observation.agent.player_index), seed=0
    )
    before = env.state_digest()
    try:
        assert runner(env, observation) == 0
        assert runner.last_decision is not None
        assert runner.last_decision.probabilities == []
        assert runner.last_decision.value is None
        assert env.state_digest() == before
    finally:
        runner.close()


def test_tokenizer_changes_refuse_pinned_qwen(qwen_directory: Path) -> None:
    from manabot.sim.qwen_provider import provider

    registry = PolicyRegistry(providers={"qwen": provider})
    key = registry.register(PolicySpec("qwen", str(qwen_directory)))
    path = qwen_directory / "tokenizer_config.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        registry.load(
            key,
            _parse_game_config({"villain_type": "random"}).to_rust(),
            seat=0,
            seed=0,
        )
