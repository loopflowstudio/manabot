"""Experimental typed record encoders over the frozen, viewer-safe v5 tensors.

SharedFactEncoder factors *complete fact fragments*, not native effects: equal
payloads share computation but retain every object membership and multiplicity.
TypedContinuationEncoder separates frame headers from their variable-length
edge lists, uses parent handles only as joins, and pools stack, execution and
trigger frames separately. Both treatments read identical fields; their only
within-record difference is a sum versus a GRU. Neither retains temporal state.
The v5 admission limits and missing exact effect-source links still apply.
"""

from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence

from manabot.env.observation import (
    FACT_DIM,
    FACT_KINDS,
    FACT_ROWS,
    RELATION_DIM_V5,
    RELATION_ROWS,
    RELATION_VALID_INDEX,
)

RecordReduction = Literal["typed_pool", "typed_gru"]


def _rows(
    value: torch.Tensor, count: int, width: int, valid_column: int
) -> torch.Tensor:
    """Mask padding before any arithmetic or indexing; reject malformed live rows."""
    if value.ndim != 3 or value.shape[1:] != (count, width):
        raise ValueError(f"expected [batch, {count}, {width}] state records")
    valid = value[..., valid_column]
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError("state record validity must be binary")
    clean = torch.where(valid.bool().unsqueeze(-1), value, 0)
    if not torch.isfinite(clean).all():
        raise ValueError("state records must be finite")
    return clean


def _indexes(values: torch.Tensor, maximum: int, name: str) -> torch.Tensor:
    if not ((values == values.floor()) & (values >= 0) & (values <= maximum)).all():
        raise ValueError(f"invalid {name} reference")
    return values.long()


def _join(values: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    """One-based local references; zero is explicit absence, never object zero."""
    return values[(indexes - 1).clamp_min(0)].masked_fill(
        (indexes == 0).unsqueeze(-1), 0
    )


class RecordReducer(nn.Module):
    """Reduce nonempty ordered token lists; no hidden state crosses calls."""

    def __init__(self, width: int, reduction: RecordReduction) -> None:
        super().__init__()
        self.gru = (
            nn.GRU(width, width, batch_first=True) if reduction == "typed_gru" else None
        )
        self.output = nn.Linear(width, width)

    def forward(self, records: list[torch.Tensor]) -> torch.Tensor:
        if self.gru is None:
            pooled = torch.stack([record.sum(0) for record in records])
        else:
            # Embedding happens before packing only for actual tokens, not a
            # batch-wide padded record array. Gradients reach every input token.
            _, hidden = self.gru(pack_sequence(records, enforce_sorted=False))
            pooled = hidden[-1]
        return self.output(torch.tanh(pooled))


class SharedFactEncoder(nn.Module):
    def __init__(self, width: int, reduction: RecordReduction) -> None:
        super().__init__()
        self.kind = nn.Embedding(FACT_KINDS + 1, width)
        self.ident = nn.Embedding(4096, width)
        self.detail = nn.Embedding(4096, width)
        self.source = nn.Embedding(4096, width)
        # Include every v5 duration, including WhileYouControlSource=3.
        self.numeric = nn.Linear(6, width)
        self.reducer = RecordReducer(width, reduction)

    def forward(self, facts: torch.Tensor, objects: torch.Tensor) -> torch.Tensor:
        facts = _rows(facts, FACT_ROWS, FACT_DIM, 11)
        results: list[torch.Tensor] = []
        for rows, table in zip(facts, objects, strict=True):
            active = rows[rows[:, 11].bool()]
            links = _indexes(active[:, 0], len(table), "fact object")
            if (links == 0).any():
                raise ValueError("live fact has no object")
            if not len(active):
                results.append(table)
                continue
            # Group all payload columns, including source, timestamp and detail.
            # The inverse preserves duplicates and different object memberships.
            payload, inverse = torch.unique(active[:, 1:11], dim=0, return_inverse=True)
            kinds = _indexes(payload[:, 0], FACT_KINDS, "fact kind")
            if (kinds == 0).any():
                raise ValueError("live fact has no kind")
            ids = _indexes(payload[:, 1:5], 65535, "fact vocabulary half")
            source = _indexes(payload[:, 9], 2**24, "fact source")
            duration = _indexes(payload[:, 7], 3, "fact duration")
            numeric = torch.cat(
                (
                    payload[:, 5:7],
                    torch.stack([(duration == i).float() for i in (1, 2, 3)], -1),
                    payload[:, 8:9] / FACT_ROWS,
                ),
                -1,
            )
            tokens = torch.stack(
                (
                    self.kind(kinds),
                    self.ident(((ids[:, 0] << 16) | ids[:, 1]) % 4096),
                    self.detail(((ids[:, 2] << 16) | ids[:, 3]) % 4096),
                    self.source(source % 4096),
                    self.numeric(numeric),
                ),
                1,
            )
            encoded = self.reducer(list(tokens.unbind(0)))[inverse]
            delta = torch.zeros_like(table).index_add(0, links - 1, encoded)
            results.append(table + delta)
        return torch.stack(results)


class TypedContinuationEncoder(nn.Module):
    """Headers plus edge sequences, preserving object/definition/action joins.

    Frames are addressed by their existing handles, never their enumeration in
    Python. Public stack rank is derived from v5's ordered stack handles; other handles are
    not learned features. Parent messages read the parent's local representation
    once (not recursively), which bounds computation even on malformed cycles;
    cycles and dangling parents are rejected before encoding.
    """

    def __init__(self, width: int, reduction: RecordReduction) -> None:
        super().__init__()
        # Header values: kind/controller, mode..decline, decision, resolutions,
        # X/keywords/delayed duration; references are separate learned joins.
        self.header_columns = (3, 4, *range(7, 22), 28, 30, *range(33, 49))
        # Child values include mode/mana/restriction because mana-output rows
        # override those header fields; target ordinal/eligibility stay explicit.
        self.edge_columns = (0, 7, *range(12, 20), 22, *range(25, 30), 32)
        self.header = nn.Linear(len(self.header_columns) + 2 * width + 1, width)
        self.edge = nn.Linear(len(self.edge_columns) + 2 * width, width)
        self.reducer = RecordReducer(width, reduction)
        self.parent = nn.Linear(width, width, bias=False)
        self.structures = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in range(3)
        )

    def forward(
        self, relations: torch.Tensor, objects: torch.Tensor, definitions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = _rows(relations, RELATION_ROWS, RELATION_DIM_V5, RELATION_VALID_INDEX)
        all_frames: list[torch.Tensor] = []
        summaries: list[torch.Tensor] = []
        for batch, table in enumerate(objects):
            live = rows[batch, :, RELATION_VALID_INDEX].bool()
            active = rows[batch, live]
            defs = definitions[batch, live]
            frames = table.new_zeros((RELATION_ROWS, table.shape[-1]))
            if not len(active):
                all_frames.append(frames)
                summaries.append(table.new_zeros(table.shape[-1]))
                continue
            handles = _indexes(active[:, 1], RELATION_ROWS, "frame")
            parents = _indexes(active[:, 2], RELATION_ROWS, "parent")
            kinds = _indexes(active[:, 3], 10, "frame kind")
            row_kinds = _indexes(active[:, 0], 6, "relation kind")
            if (handles == 0).any() or (kinds == 0).any() or (row_kinds == 0).any():
                raise ValueError("live relation lacks frame or kind")
            sources = _indexes(active[:, 6], len(table), "source object")
            targets = _indexes(active[:, 24], len(table), "target object")
            headers = (row_kinds == 1).nonzero().flatten()
            header_handles = handles[headers]
            if len(header_handles.unique()) != len(headers) or set(
                handles.tolist()
            ) != set(header_handles.tolist()):
                raise ValueError("each frame needs exactly one header")
            # Parent validation uses the sparse routing graph, never numeric
            # identity as a feature. A fixed 64-node traversal bounds admission.
            parent_map = dict(
                zip(header_handles.tolist(), parents[headers].tolist(), strict=True)
            )
            for handle in parent_map:
                seen: set[int] = set()
                cursor = handle
                while cursor:
                    if cursor in seen or cursor not in parent_map:
                        raise ValueError("cyclic or dangling continuation parent")
                    seen.add(cursor)
                    cursor = parent_map[cursor]
            records: list[torch.Tensor] = []
            stack_handles = sorted(handles[headers][kinds[headers] <= 3].tolist())
            for index in headers.tolist():
                kind = int(kinds[index])
                stack_rank = (
                    stack_handles.index(int(handles[index])) + 1 if kind <= 3 else 0
                )
                rank = active.new_tensor([stack_rank / RELATION_ROWS])
                header = torch.cat(
                    (
                        active[index, self.header_columns],
                        defs[index, 0],
                        _join(table, sources[index : index + 1])[0],
                        rank,
                    )
                )
                children = (
                    ((handles == handles[index]) & (row_kinds != 1)).nonzero().flatten()
                )
                # Targets are ordered by clause/ordinal; other edge sets use a
                # canonical payload order. Native target ordinals are unique;
                # equal hybrid/mana rows are identical tokens. Routing IDs
                # never enter a numeric projection or an edge ordering key.
                children_list = children.tolist()
                children_list.sort(
                    key=lambda i: tuple(active[i, self.edge_columns].tolist())
                )
                tokens = [torch.tanh(self.header(header))]
                for child in children_list:
                    edge = torch.cat(
                        (
                            active[child, self.edge_columns],
                            defs[child, 1],
                            _join(table, targets[child : child + 1])[0],
                        )
                    )
                    tokens.append(torch.tanh(self.edge(edge)))
                records.append(torch.stack(tokens))
            local = self.reducer(records)
            frames = frames.index_add(0, header_handles - 1, local)
            parent_messages = self.parent(_join(frames, parents[headers]))
            resolved = local + parent_messages
            frames = frames.index_copy(0, header_handles - 1, resolved)
            groups = (
                kinds[headers] <= 3,
                (kinds[headers] >= 4) & (kinds[headers] <= 6),
                kinds[headers] >= 7,
            )
            summary = table.new_zeros(table.shape[-1])
            for mask, projection in zip(groups, self.structures, strict=True):
                if mask.any():
                    summary = summary + projection(resolved[mask].mean(0))
            all_frames.append(frames)
            summaries.append(summary)
        return torch.stack(summaries), torch.stack(all_frames)
