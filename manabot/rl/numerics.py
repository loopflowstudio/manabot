"""Named numerical refusals, and the host boundary for float64 arithmetic.

A learning piece that cannot trust a tensor raises `NumericalError` naming
the invariant it checked, carrying the tensors that broke it; callers attach
more evidence and decide whether the run survives.
"""

from collections.abc import Mapping

from torch import Tensor


class NumericalError(ValueError):
    """An identified invariant plus tensors for bounded private failure capture."""

    def __init__(self, invariant: str, tensors: Mapping[str, Tensor]) -> None:
        super().__init__(invariant)
        self.invariant = invariant
        self.health: dict[str, int | float] = {}
        self.tensors = {key: value.detach() for key, value in tensors.items()}


def require(condition: Tensor, invariant: str, **tensors: Tensor) -> None:
    """Raise `NumericalError(invariant)` unless every element of `condition` holds."""
    if not bool(condition.all()):
        raise NumericalError(invariant, {**tensors, "failed": ~condition})


def float64_operand(tensor: Tensor) -> Tensor:
    """Move MPS operands to the host, where float64 is supported.

    MPS has no float64 tensors. The CPU transfer preserves autograd; callers
    retain their explicit double casts and keep companion operands together.
    Elsewhere this is a no-op relabel: the identical tensor, dtype and device
    are returned, with no transfer or additional tensor operation.
    """
    return tensor.cpu() if tensor.device.type == "mps" else tensor
