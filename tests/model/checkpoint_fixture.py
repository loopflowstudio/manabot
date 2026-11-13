"""Editable container fixtures, including deliberate metadata/admission defects."""

import json
from pathlib import Path
from typing import Any

import torch

from manabot.policy.checkpoint import (
    canonical_name,
    checkpoint_metadata,
    read_safetensors,
    torch_name,
    write_safetensors,
)


def read_checkpoint(path: str | Path, **unused: object) -> dict[str, Any]:
    arrays, metadata = read_safetensors(Path(path))
    record = {
        key: json.loads(value)
        for key, value in metadata.items()
        if key != "converted_from"
    }
    record["model_state_dict"] = {
        key.removeprefix("buffer/")
        if key.startswith("buffer/")
        else torch_name(key): torch.from_numpy(value.copy())
        for key, value in arrays.items()
    }
    return record


def write_checkpoint(record: dict[str, Any], path: str | Path) -> None:
    parameters = set(record["parameter_names"])
    arrays = {
        canonical_name(key) if key in parameters else "buffer/" + key: tensor.detach()
        .cpu()
        .numpy()
        for key, tensor in record["model_state_dict"].items()
    }
    metadata = checkpoint_metadata(record)
    metadata["parameter_names"] = json.dumps(record["parameter_names"])
    write_safetensors(Path(path), arrays, metadata)
