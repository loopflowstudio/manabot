"""The shared checkpoint container and canonical names, without any framework."""

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from manabot.policy.checkpoint import (
    NotAContainer,
    canonical_name,
    read_container,
    read_metadata,
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


def test_reading_a_file_that_is_not_a_container_says_to_convert(
    tmp_path: Path,
) -> None:
    torch_file = tmp_path / "checkpoint.pt"
    torch_file.write_bytes(b"PK\x03\x04" + bytes(range(256)) * 4)
    with pytest.raises(NotAContainer, match="manabot.backends.convert"):
        read_container(torch_file)
    bare = tmp_path / "checkpoint.mlx.safetensors"
    write_safetensors(bare, {"w": np.zeros(2, dtype=np.float32)}, {})
    with pytest.raises(NotAContainer, match="no hypers, world_binding"):
        read_container(bare)


def test_metadata_reader_never_reads_tensor_payload(tmp_path: Path) -> None:
    path = tmp_path / "policy.safetensors"
    metadata = {
        key: json.dumps(value)
        for key, value in {
            "hypers": {"agent_hypers": {"hidden_dim": 8}},
            "world_binding": {"world": "fixture"},
            "architecture": None,
            "source": {"commit": "fixture"},
            "bc": {"value_semantic": "signed_outcome"},
        }.items()
    }
    write_safetensors(path, {"weight": np.arange(8, dtype=np.float32)}, metadata)
    data = path.read_bytes()
    (length,) = struct.unpack("<Q", data[:8])
    # A metadata inspector needs no tensor payload, even if it is unavailable.
    path.write_bytes(data[: 8 + length])
    assert read_metadata(path) == {
        key: json.loads(value) for key, value in metadata.items()
    }
