"""The shared checkpoint container and canonical names, without any framework."""

from pathlib import Path

import numpy as np

from manabot.policy.checkpoint import (
    canonical_name,
    read_safetensors,
    write_safetensors,
)


def test_container_files_round_trip_arrays_and_metadata(tmp_path: Path) -> None:
    arrays = {
        "attention.mha.qkv_weight": np.arange(12, dtype=np.float32).reshape(6, 2),
        "buffer/semantic_cards.tokens": np.array([[3, -1], [7, 2]], dtype=np.int64),
        "buffer/_is_agent_template": np.array([True, False, True]),
        "empty": np.zeros((0, 4), dtype=np.float32),
    }
    path = tmp_path / "policy.safetensors"
    write_safetensors(path, arrays, {"source": '{"commit": "7c1e9a2"}'})
    read, metadata = read_safetensors(path)
    assert metadata == {"source": '{"commit": "7c1e9a2"}'}
    assert read.keys() == arrays.keys()
    for name, array in arrays.items():
        assert read[name].dtype == array.dtype
        np.testing.assert_array_equal(read[name], array)


def test_canonical_names_rename_only_framework_idioms() -> None:
    assert (
        canonical_name("facts.program_sequence.weight_ih_l0")
        == "facts.program_sequence.input_weight"
    )
    assert (
        canonical_name("shared_effects.reducer.gru.bias_hh_l0")
        == "shared_effects.reducer.gru.hidden_bias"
    )
    assert (
        canonical_name("extra_attention.2.mha.in_proj_bias")
        == "extra_attention.2.mha.qkv_bias"
    )
    assert canonical_name("value_head.5.weight") == "value_head.5.weight"
    assert (
        canonical_name("attention.mha.out_proj.weight")
        == "attention.mha.out_proj.weight"
    )
