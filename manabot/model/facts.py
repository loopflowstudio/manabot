"""
Policy-input fact rows, joined to the player or permanent each describes.

A fact row states one open-vocabulary fact (a counter kind, an added subtype, a
set name or creature type, a spent ability, a continuous-effect record) about
one object; `manabot.env.observation` documents the columns. Each row is
embedded from its kind, its vocabulary id and effect detail (learning-token
values rebuilt from their 16-bit halves), its amounts, duration, timestamp rank
and source, then summed into its object's pre-attention embedding.

Two id vocabularies exist. `hashed` hashes ids and sources into `buckets`
learned rows each. `catalog` reads ids
through the card encoder's token embedding and sources through its row for
that definition (`RecordEmbeddings`); ordinal ids (an ability index, a
continuation frame) are numbers, not vocabulary.

The output layer starts at zero, so a fresh model's objects are unchanged until
the facts earn weight; gradients reach every fact row that names a valid object.

Captured programs are encoded in row order and preserve signed numeric tokens.

`admit_facts` checks rows once at observation admission; the encoder assumes
admitted rows and keeps only its data-dependent compaction and packing.
"""

from typing import Literal

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, unpack_sequence

from manabot.env.observation import (
    FACT_COMBAT_RESTRICTION,
    FACT_CONTINUATION_PROGRAM,
    FACT_COPY_EXCEPTION,
    FACT_DETAIL_INDEX,
    FACT_DIM,
    FACT_DURATION_INDEX,
    FACT_EFFECT,
    FACT_EFFECT_PROGRAM,
    FACT_ID_INDEX,
    FACT_KIND_INDEX,
    FACT_KINDS,
    FACT_OBJECT_INDEX,
    FACT_ORDER_INDEX,
    FACT_SET_NAME,
    FACT_SOURCE_INDEX,
    FACT_TIMESTAMP_SCALE,
    FACT_TRANSFORMS,
    FACT_USED_ABILITY,
    FACT_VALID_INDEX,
    FACT_VALUE_INDEX,
)
from manabot.model.numerics import buffer_on
from manabot.model.records import program_number
from manabot.model.semantic_cards import (
    RecordEmbeddings,
    RecordNamespace,
    SemanticCardEncoder,
    admit_definition_ids,
)

RecordVocabulary = Literal["hashed", "catalog"]

# Kinds whose id column is an ordinal (an ability index or a continuation
# frame + 1), never a vocabulary id.
_ORDINAL_ID_KINDS = (
    FACT_TRANSFORMS,
    FACT_USED_ABILITY,
    FACT_CONTINUATION_PROGRAM,
    FACT_COMBAT_RESTRICTION,
)
_PROGRAM_KINDS = (FACT_EFFECT_PROGRAM, FACT_CONTINUATION_PROGRAM)
# Kinds whose detail names a keyword, subtype or color (0: none).
_NAMED_DETAIL_KINDS = (FACT_EFFECT, FACT_COPY_EXCEPTION)


def _word(rows: torch.Tensor, index: int) -> torch.Tensor:
    """The u32 id at `index` (two 16-bit halves)."""
    return (rows[..., index].long() << 16) | rows[..., index + 1].long()


def _isin(kinds: torch.Tensor, values: tuple[int, ...]) -> torch.Tensor:
    return torch.isin(kinds, kinds.new_tensor(values))


def _catalog_columns(
    kinds: torch.Tensor, detail: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masks of rows with a named id, a named detail and a program token, and
    the namespaces of their ids and details."""
    named = (kinds > 0) & ~_isin(kinds, _ORDINAL_ID_KINDS)
    program = _isin(kinds, _PROGRAM_KINDS)
    restriction = kinds == FACT_COMBAT_RESTRICTION
    named_detail = (_isin(kinds, _NAMED_DETAIL_KINDS) & (detail != 0)) | restriction
    id_space = torch.where(
        kinds == FACT_SET_NAME, RecordNamespace.NAME, RecordNamespace.TOKEN
    )
    detail_space = torch.where(
        restriction, RecordNamespace.RESTRICTION, RecordNamespace.TOKEN
    )
    return named, named_detail, program, id_space, detail_space


def _program_lengths(rows: torch.Tensor) -> list[int]:
    """Lengths of the contiguous programs in program rows [P, FACT_DIM].

    Each admitted program starts at ordinal zero, so starts are the zeros.
    """
    starts = (rows[:, FACT_VALUE_INDEX + 1] == 0).nonzero().flatten().tolist()
    ends = [*starts[1:], len(rows)]
    return [end - start for start, end in zip(starts, ends, strict=True)]


class FactEncoder(nn.Module):
    def __init__(
        self,
        width: int,
        max_facts: int,
        buckets: int = 4096,
        *,
        vocabulary: RecordVocabulary = "hashed",
    ) -> None:
        super().__init__()
        self.buckets = buckets
        self.max_facts = max_facts
        self.vocabulary = vocabulary
        self.kind = nn.Embedding(FACT_KINDS + 1, width)
        if vocabulary == "hashed":
            self.ident = nn.Embedding(buckets, width)
            self.detail = nn.Embedding(buckets, width)
            self.source = nn.Embedding(buckets, width)
        # Amounts (2), duration one-hot (2) and timestamp rank; catalog ids
        # add the ordinal id.
        self.numeric = nn.Linear(5 if vocabulary == "hashed" else 6, width)
        self.program_numeric = nn.Linear(1, width)
        self.program_sequence = nn.GRU(width, width, batch_first=True)
        self.output = nn.Linear(width, width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # Bucketed shapes look ids up at the fact axis's shape instead of
        # selecting live rows, and keep every slot on CPU inference.
        self.bucketed = False

    def _programs(
        self, hidden: torch.Tensor, facts: torch.Tensor, kinds: torch.Tensor
    ) -> torch.Tensor:
        """Reduce each complete program independently, visiting only live tokens.

        Native/Python encoders emit contiguous programs with zero-based ordinals.
        An ordinal reset starts a new occurrence, even for duplicate grants with
        identical provenance. Padding and independent program order carry no
        meaning; token order within each program does. No state crosses programs.
        """
        program = _isin(kinds, _PROGRAM_KINDS)
        rows = facts[program]
        if len(rows) == 0:
            return hidden
        lengths = _program_lengths(rows)
        word = _word(rows, FACT_DETAIL_INDEX)
        signed = torch.where(word >= 2**31, word - 2**32, word).float()
        token_kind = rows[:, FACT_VALUE_INDEX]
        # TokenKind Integer (4) and Boolean (5) retain their numeric value.
        number = torch.where((token_kind == 4) | (token_kind == 5), signed, 0)
        tokens = hidden[program] + self.program_numeric(
            program_number(number).unsqueeze(-1)
        )
        packed, _ = self.program_sequence(
            pack_sequence(list(tokens.split(lengths)), enforce_sorted=False)
        )
        sequence = torch.cat(unpack_sequence(packed))
        return hidden.masked_scatter(
            program.unsqueeze(-1), (tokens + sequence).to(hidden.dtype)
        )

    def _catalog_ids(
        self, facts: torch.Tensor, kinds: torch.Tensor, catalog: RecordEmbeddings
    ) -> torch.Tensor:
        """Id, detail and source embeddings [B,F,D] from the card vocabulary.

        Padding rows have kind 0 and contribute nothing. Admission rejects a
        detail on a kind that names none and any id outside the vocabulary.
        """
        ident, detail = _word(facts, FACT_ID_INDEX), _word(facts, FACT_DETAIL_INDEX)
        named, named_detail, program, id_space, detail_space = _catalog_columns(
            kinds, detail
        )
        if self.bucketed:
            return (
                catalog.masked_names(id_space, ident, named)
                + catalog.masked_names(detail_space, detail, named_detail)
                + catalog.masked_tokens(facts[..., FACT_VALUE_INDEX], detail, program)
                + catalog.sources(facts[..., FACT_SOURCE_INDEX])
            )
        zeros = catalog.cards.token_embedding.weight.new_zeros(
            (*kinds.shape, self.kind.embedding_dim)
        )

        def scatter(mask: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
            return zeros.index_put(
                torch.nonzero(mask, as_tuple=True), rows.to(zeros.dtype)
            )

        return (
            scatter(named, catalog.names(id_space[named], ident[named]))
            + scatter(
                named_detail,
                catalog.names(detail_space[named_detail], detail[named_detail]),
            )
            + scatter(
                program,
                catalog.tokens(facts[..., FACT_VALUE_INDEX][program], detail[program]),
            )
            + catalog.sources(facts[..., FACT_SOURCE_INDEX])
        )

    def forward(
        self,
        facts: torch.Tensor,
        objects: torch.Tensor,
        catalog: RecordEmbeddings | None = None,
    ) -> torch.Tensor:
        """Add each admitted fact's embedding to its object; `objects` is [B,N,D].

        Catalog ids need `catalog`; hashed ids ignore it. Bucketed callers may
        pass facts trimmed to fewer rows than `max_facts`.
        """
        valid = facts[..., FACT_VALID_INDEX].bool()
        if (
            facts.device.type == "cpu"
            and not torch.is_grad_enabled()
            and not self.bucketed
        ):
            # Padding contributes no object delta. Select the ordered union of
            # live columns for this forward only, including the empty union;
            # ordinals, batch ownership and duplicate program occurrences stay
            # intact. No encoded state survives a call. Learning retains fixed
            # shapes and device inference avoids a host synchronization.
            used = valid.any(0).nonzero().flatten()
            facts = facts.index_select(1, used)
            valid = valid.index_select(1, used)
        facts = torch.where(valid.unsqueeze(-1), facts, 0)
        links = facts[..., FACT_OBJECT_INDEX]
        kinds = facts[..., FACT_KIND_INDEX]
        duration = facts[..., FACT_DURATION_INDEX]
        columns = [
            facts[..., FACT_VALUE_INDEX],
            facts[..., FACT_VALUE_INDEX + 1],
            (duration == 1).float(),
            (duration == 2).float(),
            facts[..., FACT_ORDER_INDEX] / FACT_TIMESTAMP_SCALE,
        ]
        if self.vocabulary == "hashed":
            ids = (
                self.ident(_word(facts, FACT_ID_INDEX) % self.buckets)
                + self.detail(_word(facts, FACT_DETAIL_INDEX) % self.buckets)
                + self.source(facts[..., FACT_SOURCE_INDEX].long() % self.buckets)
            )
        elif catalog is None:
            raise ValueError("catalog fact ids require the card encoder")
        else:
            ordinal = _isin(kinds.long(), _ORDINAL_ID_KINDS)
            columns.append(
                torch.where(ordinal, _word(facts, FACT_ID_INDEX), 0)
                / FACT_TIMESTAMP_SCALE
            )
            ids = self._catalog_ids(facts, kinds.long(), catalog)
        hidden = (
            self.kind(kinds.long()) + ids + self.numeric(torch.stack(columns, dim=-1))
        )
        hidden = self._programs(hidden, facts, kinds)
        encoded = self.output(torch.tanh(hidden))
        linked = (valid & (links > 0)).unsqueeze(-1)
        targets = (
            (links.long() - 1)
            .clamp_min(0)
            .unsqueeze(-1)
            .expand(-1, -1, objects.shape[-1])
        )
        delta = torch.zeros_like(objects).scatter_add(
            1, targets, encoded.masked_fill(~linked, 0).to(objects.dtype)
        )
        return objects + delta


def admit_facts(
    facts: torch.Tensor,
    max_facts: int,
    object_count: int,
    cards: SemanticCardEncoder | None,
) -> None:
    """Reject malformed fact rows before any forward.

    Checks shape, binary validity, object joins into `object_count` objects,
    kinds and contiguous programs; padding rows are ignored. With `cards`
    (catalog ids), ids, details, program tokens and sources must be in its
    vocabulary.
    """
    if facts.shape[1:] != (max_facts, FACT_DIM):
        raise ValueError(f"expected [batch, {max_facts}, {FACT_DIM}] facts")
    valid = facts[..., FACT_VALID_INDEX]
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError("fact validity must be binary")
    facts = torch.where(valid.bool().unsqueeze(-1), facts, 0)
    links = facts[..., FACT_OBJECT_INDEX]
    kinds = facts[..., FACT_KIND_INDEX]
    if not ((links == links.floor()) & (links >= 0) & (links <= object_count)).all():
        raise ValueError("invalid fact object reference")
    if not ((kinds == kinds.floor()) & (kinds >= 0) & (kinds <= FACT_KINDS)).all():
        raise ValueError("invalid fact kind")
    if cards is not None:
        _admit_catalog_ids(facts, kinds.long(), cards)
    _admit_programs(facts, kinds)


def _admit_catalog_ids(
    facts: torch.Tensor, kinds: torch.Tensor, cards: SemanticCardEncoder
) -> None:
    ident, detail = _word(facts, FACT_ID_INDEX), _word(facts, FACT_DETAIL_INDEX)
    named, named_detail, program, id_space, detail_space = _catalog_columns(
        kinds, detail
    )
    if bool((~(program | named_detail) & (detail != 0)).any()):
        raise ValueError("fact kind carries no detail")
    cards.require_record_names(id_space[named], ident[named])
    cards.require_record_names(detail_space[named_detail], detail[named_detail])
    cards.require_record_tokens(facts[..., FACT_VALUE_INDEX][program], detail[program])
    lookup = buffer_on(cards, "definition_rows", facts.device)
    admit_definition_ids(facts[..., FACT_SOURCE_INDEX], len(lookup), lookup)


def _admit_programs(facts: torch.Tensor, kinds: torch.Tensor) -> None:
    """Programs are contiguous runs with consecutive zero-based ordinals."""
    program = _isin(kinds, _PROGRAM_KINDS)
    if not program.any():
        return
    positions = program.nonzero()
    rows = facts[program]
    ordinals = rows[:, FACT_VALUE_INDEX + 1]
    starts = (ordinals == 0).nonzero().flatten().tolist()
    if not starts or starts[0] != 0:
        raise ValueError("fact program must start at ordinal zero")
    lengths = _program_lengths(rows)
    expected = torch.cat(
        [torch.arange(length, device=facts.device) for length in lengths]
    )
    if not torch.equal(ordinals, expected):
        raise ValueError("fact program ordinals must be consecutive")
    # A sequence cannot change batch, holder, kind, frame/modification ID,
    # duration, timestamp or source halfway through its program.
    owners = torch.cat(
        (
            positions[:, :1],
            rows[:, :FACT_DETAIL_INDEX],
            rows[:, FACT_DURATION_INDEX:FACT_VALID_INDEX],
        ),
        -1,
    )
    expected_owners = torch.repeat_interleave(
        owners[starts], owners.new_tensor(lengths, dtype=torch.long), dim=0
    )
    if not torch.equal(owners, expected_owners):
        raise ValueError("fact program changes owner or provenance")
