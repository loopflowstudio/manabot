"""Kept lookup: advanced indexing at every card-program read (734f2cd5c)."""

import torch


def lookup(definitions: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    return definitions[rows]
