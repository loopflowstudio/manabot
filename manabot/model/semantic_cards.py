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


class SemanticCardEncoder(nn.Module):
    def __init__(self, pack_key: str, hidden_dim: int) -> None:
        super().__init__()
        pack = bind_policy_catalog(pack_key)
        catalog = pack.catalog
        pairs = list(
            zip(
                map(int, catalog.token_kind), map(int, catalog.token_value), strict=True
            )
        )
        vocabulary = {pair: index + 1 for index, pair in enumerate(sorted(set(pairs)))}
        references = dict(
            zip(
                map(int, catalog.definition_ref_source_tokens),
                map(int, catalog.definition_ref_target_rows),
                strict=True,
            )
        )

        def definition_tokens(row: int, ancestors: tuple[int, ...] = ()) -> list[int]:
            if row in ancestors:
                raise ValueError("cyclic semantic definition reference is not admitted")
            start, stop = catalog.definition_offsets[row : row + 2]
            spans = [(int(start), int(stop))]
            for program in pack.program_rows(row):
                start, stop = catalog.program_offsets[program : program + 2]
                spans.append((int(start), int(stop)))
            tokens: list[int] = []
            for start, stop in spans:
                for offset in range(start, stop):
                    tokens.append(vocabulary[pairs[offset]])
                    if offset in references:
                        tokens.extend(
                            definition_tokens(references[offset], (*ancestors, row))
                        )
            return tokens

        # Structure, execution order, program boundaries and referenced token
        # definitions come from the checked catalog. Nothing is truncated.
        rows = [definition_tokens(row) for row in range(pack.definition_count)]
        width = max(map(len, rows))
        self.token_vocabulary = tuple(sorted(vocabulary))
        self.catalog_shape = CatalogShape(
            definitions=len(rows),
            unique_tokens=len(vocabulary),
            expanded_tokens=sum(map(len, rows)),
            maximum_program_tokens=width,
            padded_tokens=len(rows) * width,
        )
        self.token_embedding = nn.Embedding(len(vocabulary) + 1, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.register_buffer(
            "tokens",
            torch.tensor(
                [row + [0] * (width - len(row)) for row in rows], dtype=torch.long
            ),
        )
        self.register_buffer("lengths", torch.tensor(list(map(len, rows))))
        # IDs are transport into the complete catalog, never learned identity.
        lookup = torch.full(
            (max(pack.definition_row_by_card_def_id) + 2,), -1, dtype=torch.long
        )
        lookup[0] = 0
        for definition, row in pack.definition_row_by_card_def_id.items():
            lookup[definition + 1] = row + 1
        self.register_buffer("definition_rows", lookup)

    def _rows(self, ids: torch.Tensor) -> torch.Tensor:
        return admit_definition_ids(
            ids, len(self.definition_rows), self.definition_rows
        )

    def definitions(self) -> torch.Tensor:
        """Encode the immutable catalog once for visible and historical references."""
        tokens = self.token_embedding(self.tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            tokens, self.lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.encoder(packed)
        definitions = torch.cat((state.new_zeros((1, state.shape[-1])), state[0]))
        return definitions

    def embed_definitions(
        self, ids: torch.Tensor, definitions: torch.Tensor
    ) -> torch.Tensor:
        """Join visible cards or permanents to the same checked card programs."""
        return definitions[self._rows(ids)]

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        definitions: torch.Tensor | None = None,
        references: CardReferences | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if definitions is None:
            definitions = self.definitions()
        cards = definitions[
            self._rows(obs["semantic_cards"])
            if references is None
            else references.cards
        ]
        knowledge = obs["known_hand"]
        known_rows = (
            self._rows(knowledge[..., 0]) if references is None else references.known
        )
        assert known_rows is not None
        known = definitions[known_rows]
        known = (known * knowledge[..., 1, None] / 60.0).sum(dim=2)
        return cards, known
