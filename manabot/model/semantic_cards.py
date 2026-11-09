"""Complete typed card programs joined to viewer-safe card and knowledge rows.

`SemanticCardEncoder` owns the one token vocabulary of card text. With
`records`, it also owns the exact vocabulary of fact and shared-effect ids
(`RecordNamespace` keys): every name native `learning_record_names` lists reads
the row of the card token for the same concept, or a row appended after the
card tokens when no card text contains it. Nothing is hashed; an id outside
the vocabulary fails loudly.
"""

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import nn

from manabot.semantic.policy_catalog import bind_policy_catalog
import managym


class RecordNamespace(IntEnum):
    """Id column families of fact and shared-effect rows (native namespaces)."""

    TOKEN = 1
    NAME = 2
    RESTRICTION = 3


# Learning token kinds whose value is a number, not an id.
NUMBER_TOKEN_KINDS = (4, 5)


def record_key(high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    """One int64 key from a family (namespace or token kind) and a u32 word."""
    return (high.long() << 32) | (low.long() & 0xFFFFFFFF)


@dataclass(frozen=True)
class _RecordTables:
    """Token-embedding rows of record ids: `names` by (namespace, value),
    `tokens` by (kind, value) over card tokens and appended ones, `numbers` by
    numeric token kind. `size` is the embedding's row count."""

    names: dict[tuple[int, int], int]
    tokens: dict[tuple[int, int], int]
    numbers: dict[int, int]
    size: int


def _record_tables(vocabulary: dict[tuple[int, int], int]) -> _RecordTables:
    """A name whose card token exists reads that token's row, so a granted
    keyword and the printed keyword share one embedding; every other name,
    token and numeric kind gets a row appended after the card tokens."""
    tokens = dict(vocabulary)
    names: dict[tuple[int, int], int] = {}
    size = len(vocabulary) + 1

    def append() -> int:
        nonlocal size
        size += 1
        return size - 1

    for space, name, value, kind, token in managym.learning_record_names():
        key = (int(RecordNamespace[space.upper()]), value)
        if key in names:
            raise ValueError(f"record name {name!r} shares an id with another")
        if kind is None or token is None:
            names[key] = append()
        else:
            if (kind, token) not in tokens:
                tokens[(kind, token)] = append()
            names[key] = tokens[(kind, token)]
    numbers = {kind: append() for kind in NUMBER_TOKEN_KINDS}
    return _RecordTables(names, tokens, numbers, size)


def _sorted_keys(
    table: dict[tuple[int, int], int],
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = torch.tensor(list(table), dtype=torch.long)
    keys = record_key(pairs[:, 0], pairs[:, 1])
    order = keys.argsort()
    return keys[order], torch.tensor(list(table.values()))[order]


def _lookup(
    keys: torch.Tensor, rows: torch.Tensor, query: torch.Tensor, message: str
) -> torch.Tensor:
    if len(query) == 0:
        return query.new_zeros(0, dtype=torch.long)
    index = torch.searchsorted(keys, query).clamp_max(len(keys) - 1)
    if not torch.equal(keys[index], query):
        raise ValueError(message)
    return rows[index]


@dataclass(frozen=True)
class CatalogShape:
    """Separate catalog dimensions; lengths include expanded definition references.

    Counts describe representation cost, not encountered-card diversity or an
    experimentally established difficulty score. Only real tokens are embedded
    and recurrently encoded; padded_tokens describes the stored token table.
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
    """Admitted catalog indexes for visible cards, permanents, public hands and
    known decklists."""

    cards: torch.Tensor
    permanents: torch.Tensor
    known: torch.Tensor | None
    decklists: torch.Tensor | None = None


class SemanticCardEncoder(nn.Module):
    def __init__(
        self,
        pack_key: str,
        hidden_dim: int,
        decklists: bool = False,
        *,
        records: bool = False,
    ) -> None:
        """`decklists` (policy input v5) adds a projection of the pooled
        decklist rows into each player's knowledge vector; `records` adds the
        fact and shared-effect id vocabulary (see the module docstring)."""
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
        tables = _record_tables(vocabulary) if records else None
        # Appended record rows follow the card tokens, whose rows are unchanged.
        self.token_embedding = nn.Embedding(
            len(vocabulary) + 1 if tables is None else tables.size, hidden_dim
        )
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
        # One count-weighted definition pool per decklist count column (main,
        # unseen, sideboard, remaining sideboard), projected to one vector.
        self.decklist_projection = (
            nn.Linear(4 * hidden_dim, hidden_dim) if decklists else None
        )
        self.records = records
        if tables is not None:
            # Sorted int64 keys for `searchsorted`.
            for prefix, table in (
                ("record_name", tables.names),
                ("record_token", tables.tokens),
            ):
                keys, rows = _sorted_keys(table)
                self.register_buffer(f"{prefix}_keys", keys)
                self.register_buffer(f"{prefix}_rows", rows)
            self.register_buffer(
                "record_number_rows",
                torch.tensor([tables.numbers.get(kind, 0) for kind in range(6)]),
            )

    def record_names(
        self, namespace: RecordNamespace | torch.Tensor, words: torch.Tensor
    ) -> torch.Tensor:
        """Embed name ids (u32 words) of one namespace, or of one namespace
        each; [*] -> [*, D]."""
        rows = _lookup(
            self.record_name_keys,
            self.record_name_rows,
            record_key(torch.as_tensor(namespace), words).reshape(-1),
            "policy received an id outside the record vocabulary",
        )
        return self.token_embedding(rows).reshape(
            *words.shape, self.token_embedding.embedding_dim
        )

    def record_tokens(self, kinds: torch.Tensor, words: torch.Tensor) -> torch.Tensor:
        """Embed program learning tokens: ids read their vocabulary row; numbers
        read their kind's row (callers add the exact value). [*] -> [*, D]."""
        kinds, words = kinds.long().reshape(-1), words.reshape(-1)
        number = (kinds == NUMBER_TOKEN_KINDS[0]) | (kinds == NUMBER_TOKEN_KINDS[1])
        if bool(((kinds < 1) | (kinds > 9)).any()):
            raise ValueError("policy received an invalid program token kind")
        rows = self.record_number_rows[kinds.clamp(0, 5)].clone()
        rows[~number] = _lookup(
            self.record_token_keys,
            self.record_token_rows,
            record_key(kinds[~number], words[~number]),
            "policy received a program token outside the vocabulary",
        )
        return self.token_embedding(rows)

    def _rows(self, ids: torch.Tensor) -> torch.Tensor:
        return admit_definition_ids(
            ids, len(self.definition_rows), self.definition_rows
        )

    def definitions(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Encode catalog programs into a [definitions + 1, hidden] table; row 0
        is the zero padding row and every consumer indexes by catalog row + 1.

        With `rows` (unique catalog rows + 1, from `referenced_rows`), only
        those programs run through the GRU and the rest of the table is zero.
        Each program is encoded independently, so selected rows and parameter
        gradients agree with full encoding up to floating-point roundoff;
        unreferenced rows receive none. Training uses this because the catalog
        grows with the world while a batch references a small subset. Inference
        caches the full table once per policy scope instead.
        """
        count = len(self.tokens)
        if rows is None:
            rows = torch.arange(1, count + 1, device=self.tokens.device)
        table = self.token_embedding.weight.new_zeros(
            (count + 1, self.token_embedding.embedding_dim)
        )
        if len(rows) == 0:
            return table
        # CUDA's packed GRU can return Half even inside BF16 autocast. The
        # definition table follows the embedding weights (also for empty rows),
        # so normalize at this join without detaching the program gradients.
        encoded = self._encode_programs(rows).to(table.dtype)
        return table.index_copy(0, rows, encoded)

    def _encode_programs(self, rows: torch.Tensor) -> torch.Tensor:
        """Encode nonempty one-based catalog rows, preserving their supplied order.

        Packing sorts programs by length. Within each chunk, completed programs
        occupy the hidden state's suffix; only the active prefix continues.
        """
        # Pack integer IDs first: padding needs neither embeddings nor gradients.
        packed = nn.utils.rnn.pack_padded_sequence(
            self.tokens[rows - 1],
            self.lengths[rows - 1].cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        # Packed GRU backward creates a packed-input-sized slice gradient at
        # each time step. Bound that copying to 64 steps per call. This is full
        # backpropagation through time: hidden states keep their gradients at
        # every chunk boundary. The same GRU encodes every real token once.
        batches = packed.batch_sizes.split(64)
        sizes = [int(batch.sum()) for batch in batches]
        chunks = self.token_embedding(packed.data).split(sizes)
        state: torch.Tensor | None = None
        finished: list[torch.Tensor] = []
        for index, (data, batch) in enumerate(zip(chunks, batches, strict=True)):
            active = int(batch[0])
            _, state = self.encoder(
                nn.utils.rnn.PackedSequence(data, batch),
                None if state is None else state[:, :active],
            )
            remaining = int(batches[index + 1][0]) if index + 1 < len(batches) else 0
            finished.append(state[:, remaining:])
        # Packed sequences end shortest first. Reverse their completed suffixes
        # to restore descending-length order, then undo the original sorting.
        encoded = torch.cat(finished[::-1], dim=1)[0]
        assert packed.unsorted_indices is not None
        return encoded[packed.unsorted_indices]

    def referenced_rows(self, ids: list[torch.Tensor]) -> torch.Tensor:
        """Unique catalog rows + 1 named by admissible transport IDs in `ids`.

        Inadmissible IDs are skipped here, not rejected: each consumer admits
        its own references and raises its own error, and padding is ignored.
        """
        found = []
        for values in ids:
            values = values.reshape(-1)
            admissible = (
                (values == values.floor())
                & (values >= 0)
                & (values < len(self.definition_rows))
            )
            found.append(self.definition_rows[values[admissible].long()])
        rows = torch.cat(found).unique()
        return rows[rows > 0]

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
        if self.decklist_projection is not None:
            decklists = obs["decklists"]
            deck_rows = (
                self._rows(decklists[..., 0])
                if references is None or references.decklists is None
                else references.decklists
            )
            # [batch, seat, row, hidden] x [batch, seat, row, 4 counts]
            # -> [batch, seat, 4 * hidden]; a closed decklist pools to zero.
            pooled = torch.einsum(
                "bsrh,bsrc->bsch", definitions[deck_rows], decklists[..., 1:] / 60.0
            ).flatten(start_dim=2)
            known = known + self.decklist_projection(pooled)
        return cards, known


@dataclass(frozen=True)
class RecordEmbeddings:
    """The card encoder's record vocabulary bound to one forward pass's
    definition table: what catalog fact and shared-effect ids read."""

    cards: SemanticCardEncoder
    definitions: torch.Tensor

    def names(
        self, namespace: RecordNamespace | torch.Tensor, words: torch.Tensor
    ) -> torch.Tensor:
        return self.cards.record_names(namespace, words)

    def tokens(self, kinds: torch.Tensor, words: torch.Tensor) -> torch.Tensor:
        return self.cards.record_tokens(kinds, words)

    def sources(self, ids: torch.Tensor) -> torch.Tensor:
        """Source definition ids (definition id + 1) to their program rows."""
        return self.cards.embed_definitions(ids, self.definitions)
