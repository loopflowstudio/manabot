"""Device-specific arithmetic boundaries that preserve learning semantics."""

import torch
from torch import Tensor, nn


def embedding_rows(embedding: nn.Embedding, rows: Tensor) -> Tensor:
    """Embed rows; an empty lookup returns an empty slice of the weights.

    Torch 2.10 MPS embedding backward rejects empty indices. The slice has the
    same empty shape and the same connected zero gradient on every device, so
    Adam still applies existing momentum. Nonempty lookups are unchanged.
    """
    if rows.numel() == 0:
        return embedding.weight[:0].reshape(*rows.shape, embedding.embedding_dim)
    return embedding(rows)


def buffer_on(module: nn.Module, name: str, device: torch.device) -> Tensor:
    """An immutable lookup buffer of `module` on `device`, for admission checks.

    CPU copies are cached per buffer object and version, so admitting CPU
    observations never reads from an accelerator. The cache is not module state.
    """
    buffer = getattr(module, name)
    if device.type != "cpu" or buffer.device.type == "cpu":
        return buffer.to(device)
    cache = module.__dict__.setdefault("_host_buffers", {})
    key = (id(buffer), buffer._version)
    if name not in cache or cache[name][0] != key:
        cache[name] = (key, buffer.detach().cpu())
    return cache[name][1]
