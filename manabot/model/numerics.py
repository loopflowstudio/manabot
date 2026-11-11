"""Device-specific arithmetic boundaries that preserve learning semantics."""

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


def float64_operand(tensor: Tensor) -> Tensor:
    """Move MPS operands to the host, where float64 is supported.

    MPS has no float64 tensors. The CPU transfer preserves autograd; callers
    retain their explicit double casts and keep companion operands together.
    Elsewhere this is a no-op relabel: the identical tensor, dtype and device
    are returned, with no transfer or additional tensor operation.
    """
    return tensor.cpu() if tensor.device.type == "mps" else tensor
