"""Experimental typed record encoders over the frozen, viewer-safe v5 tensors.

SharedFactEncoder factors *complete fact fragments*, not native effects: equal
payloads share computation but retain every object membership and multiplicity.
TypedContinuationEncoder separates frame headers from their variable-length
edge lists, uses parent handles only as joins, and pools stack, execution and
trigger frames separately. Both treatments read identical fields; their only
within-record difference is a sum versus a GRU. Neither retains temporal state.
The v5 admission limits and missing exact effect-source links still apply.
"""

import torch
from torch import nn

from manabot.env.observation import (
    FACT_DIM,
    FACT_KINDS,
    FACT_TIMESTAMP_SCALE,
    RELATION_DIM_V5,
    RELATION_ROWS,
    RELATION_VALID_INDEX,
)
from manabot.infra.config import ObservationConfig
from manabot.model.records import (
    RecordReducer,
    RecordReduction,
    join_record_rows,
    masked_record_rows,
    record_indexes,
)


class SharedFactEncoder(nn.Module):
    def __init__(self, width: int, reduction: RecordReduction) -> None:
        super().__init__()
        self.rows = ObservationConfig(policy_input_version=5).fact_rows
        self.kind = nn.Embedding(FACT_KINDS + 1, width)
        self.ident = nn.Embedding(4096, width)
        self.detail = nn.Embedding(4096, width)
        self.source = nn.Embedding(4096, width)
        # Include every v5 duration, including WhileYouControlSource=3.
        self.numeric = nn.Linear(6, width)
        self.reducer = RecordReducer(width, reduction)

    def forward(self, facts: torch.Tensor, objects: torch.Tensor) -> torch.Tensor:
        facts = masked_record_rows(facts, self.rows, FACT_DIM, 11)
        if facts.shape[0] != objects.shape[0]:
            raise ValueError("fact and object batch sizes differ")
        results = list(objects.unbind(0))
        # One routing transfer for the batch. Empty observations need no
        # per-sample CUDA indexing/validation synchronizations.
        active_batches = facts[..., 11].bool().any(-1).nonzero().flatten().tolist()
        for batch in active_batches:
            rows, table = facts[batch], objects[batch]
            active = rows[rows[:, 11].bool()]
            links = record_indexes(active[:, 0], len(table), "fact object")
            if (links == 0).any():
                raise ValueError("live fact has no object")
            # Group all payload columns, including source, timestamp and detail.
            # The inverse preserves duplicates and different object memberships.
            payload, inverse = torch.unique(active[:, 1:11], dim=0, return_inverse=True)
            kinds = record_indexes(payload[:, 0], FACT_KINDS, "fact kind")
            if (kinds == 0).any():
                raise ValueError("live fact has no kind")
            ids = record_indexes(payload[:, 1:5], 65535, "fact vocabulary half")
            source = record_indexes(payload[:, 9], 2**24, "fact source")
            duration = record_indexes(payload[:, 7], 3, "fact duration")
            numeric = torch.cat(
                (
                    payload[:, 5:7],
                    torch.stack([(duration == i).float() for i in (1, 2, 3)], -1),
                    payload[:, 8:9] / FACT_TIMESTAMP_SCALE,
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
            encoded = self.reducer(tokens.flatten(0, 1), [5] * len(tokens))[inverse]
            delta = torch.zeros_like(table).index_add(0, links - 1, encoded)
            results[batch] = table + delta
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
        # Admit routing once on the host; all learned features below still use
        # the original device tensors. Handles remain joins, never features.
        rows = masked_record_rows(
            relations.detach().cpu(),
            RELATION_ROWS,
            RELATION_DIM_V5,
            RELATION_VALID_INDEX,
        )
        batch, object_count, width = objects.shape
        if rows.shape[0] != batch:
            raise ValueError("relation and object batch sizes differ")
        positions = rows[..., RELATION_VALID_INDEX].nonzero()
        if len(positions) == 0:
            return objects.new_zeros((batch, width)), objects.new_zeros(
                (batch, RELATION_ROWS, width)
            )
        active = rows[positions[:, 0], positions[:, 1]]
        handles = record_indexes(active[:, 1], RELATION_ROWS, "frame")
        parents = record_indexes(active[:, 2], RELATION_ROWS, "parent")
        kinds = record_indexes(active[:, 3], 12, "frame kind")
        row_kinds = record_indexes(active[:, 0], 9, "relation kind")
        if (handles == 0).any() or (kinds == 0).any() or (row_kinds == 0).any():
            raise ValueError("live relation lacks frame or kind")
        sources = record_indexes(active[:, 6], object_count, "source object")
        targets = record_indexes(active[:, 24], object_count, "target object")
        by_batch: dict[int, list[int]] = {}
        for index, (b, _) in enumerate(positions.tolist()):
            by_batch.setdefault(b, []).append(index)
        handle_list, parent_list, kind_list, row_kind_list = (
            value.tolist() for value in (handles, parents, kinds, row_kinds)
        )
        edge_keys = active[:, self.edge_columns].tolist()
        header_indexes: list[int] = []
        token_indexes: list[int] = []
        lengths: list[int] = []
        ranks = [0.0] * len(active)
        for indexes in by_batch.values():
            headers = [i for i in indexes if row_kind_list[i] == 1]
            header_handles = [handle_list[i] for i in headers]
            if len(set(header_handles)) != len(headers) or {
                handle_list[i] for i in indexes
            } != set(header_handles):
                raise ValueError("each frame needs exactly one header")
            parent_map = {handle_list[i]: parent_list[i] for i in headers}
            for handle in parent_map:
                seen: set[int] = set()
                cursor = handle
                while cursor:
                    if cursor in seen or cursor not in parent_map:
                        raise ValueError("cyclic or dangling continuation parent")
                    seen.add(cursor)
                    cursor = parent_map[cursor]
            stack_handles = sorted(handle_list[i] for i in headers if kind_list[i] <= 3)
            stack_ranks = {
                handle: rank / RELATION_ROWS
                for rank, handle in enumerate(stack_handles, start=1)
            }
            edges_by_handle: dict[int, list[int]] = {h: [] for h in header_handles}
            for index in indexes:
                if row_kind_list[index] != 1:
                    edges_by_handle[handle_list[index]].append(index)
            for index in headers:
                ranks[index] = stack_ranks.get(handle_list[index], 0.0)
                edges = edges_by_handle[handle_list[index]]
                # Payload order is canonical; routing handles are not features.
                edges.sort(key=edge_keys.__getitem__)
                header_indexes.append(index)
                token_indexes.extend((index, *edges))
                lengths.append(1 + len(edges))
        device = objects.device
        batches = positions[:, 0]
        flat_indexes = (batches * RELATION_ROWS + positions[:, 1]).to(device)
        features = relations.reshape(-1, RELATION_DIM_V5)[flat_indexes]
        defs = definitions.reshape(-1, 2, width)[flat_indexes]
        header = torch.tanh(
            self.header(
                torch.cat(
                    (
                        features[:, self.header_columns],
                        defs[:, 0],
                        join_record_rows(objects, batches, sources),
                        features.new_tensor(ranks).unsqueeze(-1),
                    ),
                    -1,
                )
            )
        )
        tokens = header
        edge_rows = (row_kinds != 1).nonzero().flatten()
        edge_indexes = edge_rows.to(device)
        # Preserve absent gradients for unused modules: Adam treats None
        # differently from a zero gradient when momentum already exists.
        if len(edge_indexes):
            edge = torch.tanh(
                self.edge(
                    torch.cat(
                        (
                            features[edge_indexes][:, self.edge_columns],
                            defs[edge_indexes, 1],
                            join_record_rows(
                                objects, batches[edge_rows], targets[edge_rows]
                            ),
                        ),
                        -1,
                    )
                )
            )
            tokens = header.index_copy(0, edge_indexes, edge)
        local = self.reducer(
            tokens[torch.tensor(token_indexes, device=device)], lengths
        )
        header_batches = batches[header_indexes]
        frame_indexes = (
            header_batches * RELATION_ROWS + handles[header_indexes] - 1
        ).to(device)
        frames = objects.new_zeros((batch * RELATION_ROWS, width)).index_add(
            0, frame_indexes, local
        )
        header_parents = parents[header_indexes]
        resolved = local + self.parent(
            join_record_rows(
                frames.reshape(batch, RELATION_ROWS, width),
                header_batches,
                header_parents,
            )
        )
        frames = frames.index_copy(0, frame_indexes, resolved)
        header_kinds = kinds[header_indexes]
        groups = torch.where(header_kinds <= 3, 0, torch.where(header_kinds <= 6, 1, 2))
        group_indexes = (header_batches * 3 + groups).to(device)
        sums = objects.new_zeros((batch * 3, width)).index_add(
            0, group_indexes, resolved
        )
        counts = objects.new_zeros(batch * 3).index_add(
            0, group_indexes, objects.new_ones(len(resolved))
        )
        means = (sums / counts.clamp_min(1).unsqueeze(-1)).reshape(batch, 3, width)
        summary = objects.new_zeros((batch, width))
        for group, projection in enumerate(self.structures):
            if (groups == group).any():
                summary = summary + projection(means[:, group])
        return summary, frames.reshape(batch, RELATION_ROWS, width)
