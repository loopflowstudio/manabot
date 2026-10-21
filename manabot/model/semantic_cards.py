"""Complete typed card programs joined to viewer-safe card and knowledge rows."""

from dataclasses import dataclass

import torch
from torch import nn

from manabot.semantic.policy_catalog import bind_policy_catalog


@dataclass(frozen=True)
class CatalogShape:
    """Separate catalog dimensions; lengths include expanded definition references.

    Counts describe representation cost, not encountered-card diversity or an
    experimentally established difficulty score. Padding is embedded before
    packing, while the GRU processes only expanded_tokens positions.
    """

    definitions: int
    unique_tokens: int
    expanded_tokens: int
    maximum_program_tokens: int
    padded_tokens: int


def admit_definition_ids(
    ids: torch.Tensor, size: int, lookup: torch.Tensor | None = None
) -> torch.Tensor:
    """Check transport IDs before indexing; optionally resolve semantic rows."""
    kind = "semantic" if lookup is not None else "card"
    message = f"policy received an unadmitted {kind} definition"
    if (
        not torch.equal(ids, ids.floor())
        or bool((ids < 0).any())
        or bool((ids >= size).any())
    ):
        raise ValueError(message)
    rows = ids.long()
    if lookup is not None:
        rows = lookup[rows]
        if bool((rows < 0).any()):
            raise ValueError(message)
    return rows


@dataclass(frozen=True)
class CardReferences:
    """Admitted catalog indexes for visible cards, permanents and public hands."""

    cards: torch.Tensor
    permanents: torch.Tensor
    known: torch.Tensor | None

    def to(self, device: torch.device) -> "CardReferences":
        return CardReferences(
            self.cards.to(device),
            self.permanents.to(device),
            None if self.known is None else self.known.to(device),
        )


class SemanticCardEncoder(nn.Module):
    def __init__(self, pack_key: str, hidden_dim: int) -> None:
        super().__init__()
        pack = bind_policy_catalog(pack_key)
