"""Decision memory encoders and their shared record featurizer.

Policy/value gradients flow through the J1 summaries into the featurizer and
encoder. Causal folds share exact prefix nodes within an update. Collection
reuses detached complete-node results for one parameter-bound inference scope;
open tails are recomputed. No neural state enters game facts or persistence.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal
from weakref import WeakKeyDictionary, ref

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from manabot.env.memory_store import PrefixIdentity, Tail
from manabot.policy.decision_memory import MemoryRows, retained_memory

if TYPE_CHECKING:
    from manabot.env.observation import ObservationBuffers
    from manabot.policy.prepare import Prepared


@dataclass(frozen=True)
class CachedNode:
    lifetime: tuple[PrefixIdentity, int | None]
    references: torch.Tensor
    records: torch.Tensor
    state: torch.Tensor


class CollectionMemory:
    """Weak node results for one unchanged inference scope; open tails never enter.

    Scope guards bind parameter versions, device and module modes. Values own only
    detached node-local tensors, never their weak keys or a whole execution tree.
    """

    def __init__(self) -> None:
        self.nodes: WeakKeyDictionary[Tail, CachedNode] = WeakKeyDictionary()

    def clear(self) -> None:
        self.nodes.clear()

    def roots(self, obs: "ObservationBuffers") -> dict[str, int]:
        retained = retained_memory(obs)
        if retained is None:
            return {}
        roots: dict[str, int] = {}
        rejected = {prefix.tail.key for prefix in retained.prefixes if prefix.tail}
        for prefix in retained.prefixes:
            for node in prefix.tail.chunks() if prefix.tail else ():
                cached = self.nodes.get(node)
                if cached is None or cached.lifetime != (
                    prefix.identity,
                    prefix.native_epoch,
                ):
                    rejected.add(node.key)
                    roots.pop(node.key, None)
                elif node.key not in rejected:
                    roots[node.key] = len(cached.references)
        return roots

    def bind(
        self, batch: "Prepared[torch.Tensor]", obs: "ObservationBuffers"
    ) -> "Prepared[torch.Tensor]":
        retained = retained_memory(obs)
        if retained is None or batch.memory is None or batch.memory.fold is None:
            return batch
        fold = batch.memory.fold
        nodes = {
            node.key: node
            for prefix in retained.prefixes
            for node in (prefix.tail.chunks() if prefix.tail else ())
        }
        lifetimes = {
            node.key: (prefix.identity, prefix.native_epoch)
            for prefix in retained.prefixes
            for node in (prefix.tail.chunks() if prefix.tail else ())
        }
        root_keys = {
            key
            for index, key in enumerate(fold.keys)
            if fold.record_slices[index][0] < 0
        }
        cached = [self.nodes[nodes[key]] for key in fold.keys if key in root_keys]
        open_keys = {prefix.tail.key for prefix in retained.prefixes if prefix.tail}
        complete = {
            node.previous.key for node in nodes.values() if node.previous is not None
        } - open_keys
        pending = [
            (i, ref(nodes[key]), lifetimes[key])
            for i, key in enumerate(fold.keys)
            if key in complete and key not in root_keys
        ]

        reference_slices, record_slices = fold.reference_slices, fold.record_slices

        def retain(
            states: torch.Tensor, references: torch.Tensor, records: torch.Tensor
        ) -> None:
            for index, weak, lifetime in pending:
                node = weak()
                if node is None:
                    continue
                start, count = reference_slices[index]
                record_start, record_count = record_slices[index]
                self.nodes[node] = CachedNode(
                    lifetime,
                    references[start : start + count].detach().clone(),
                    records[record_start : record_start + record_count]
                    .detach()
                    .clone(),
                    states[index + 1].detach().clone(),
                )

        fold = replace(
            fold,
            root_references=torch.cat([node.references for node in cached])
            if cached
            else None,
            root_states=torch.stack([node.state for node in cached])
            if cached
            else None,
            retain=retain,
        )
        return replace(batch, memory=replace(batch.memory, fold=fold))


class DecisionMemory(nn.Module):
    def __init__(
        self, encoder: Literal["transformer", "gru"], summaries: int, board_width: int
    ) -> None:
        super().__init__()
        self.summaries = summaries
        self.encoder = encoder
        # Bind labels, typed values and positions before any pooling. Keeping
        # independent linear sums here would erase field/value associations.
        self.detail_map = nn.Sequential(
            nn.Linear(board_width * 2 + 5, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU()
        )
        self.reference_map = nn.Sequential(
            nn.Linear(board_width + 64 + 8, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU()
        )
        self.member_map = nn.Sequential(
            nn.Linear(65, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU()
        )
        self.record_output = nn.Sequential(
            nn.Linear(38 + board_width * 3 + 3 * 65, 64), nn.GELU(), nn.LayerNorm(64)
        )
        if encoder == "transformer":
            layer = nn.TransformerEncoderLayer(
                64, 4, 128, dropout=0, batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(
                layer, 2, enable_nested_tensor=False
            )
            # Recency is a sequence position signal, never a record feature.
            self.recency = nn.Linear(1, 64, bias=False)
            self.queries = nn.Parameter(torch.randn(1, summaries, 64) / 8)
            self.gru = None
            self.output = nn.Linear(64, board_width)
        else:
            self.transformer = None
            self.queries = None
            self.gru = nn.GRU(64, 64, batch_first=True)
            self.output = nn.Linear(64, summaries * board_width)
        self.join_norm = nn.LayerNorm(board_width)
        self.memory_type = nn.Parameter(
            torch.randn(1, summaries, board_width) / board_width**0.5
        )

    def _summaries(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map a final recurrent state to the shared board-token contract."""
        return self.join_norm(
            self.output(hidden).reshape(-1, self.summaries, self.memory_type.shape[-1])
            + self.memory_type
        )

    def forward(
        self,
        rows: MemoryRows[torch.Tensor],
        embedding: nn.Embedding,
        definitions: torch.Tensor,
        names: nn.Embedding,
    ) -> torch.Tensor:
        batch = len(rows.lengths)
        board_width = self.memory_type.shape[-1]
        # Empty histories bypass recurrence/attention and produce exactly zero,
        # including the type embedding, in both encoders.
        if not rows.schedule.sizes and (rows.fold is None or not rows.fold.node_count):
            return self.memory_type.new_zeros((batch, self.summaries, board_width))
        details = self.detail_map(
            torch.cat(
                (
                    embedding(rows.detail_fields),
                    embedding(rows.detail_values),
                    rows.detail_numbers,
                ),
                -1,
            )
        )
        reference_count = len(rows.reference_numbers)
        counts = details.new_zeros((reference_count, 1)).index_add(
            0, rows.detail_references, details.new_ones((len(details), 1))
        )
        pooled = details.new_zeros((reference_count, 64)).index_add(
            0, rows.detail_references, details
        )
        references = self.reference_map(
            torch.cat(
                (
                    definitions[rows.reference_definitions],
                    rows.reference_numbers,
                    pooled / counts.clamp_min(1).sqrt(),
                ),
                -1,
            )
        )
        if rows.fold is not None and rows.fold.root_references is not None:
            references = torch.cat((rows.fold.root_references, references))
        members = self.member_map(
            torch.cat((references[rows.member_references], rows.member_positions), -1)
        )
        record_count = len(rows.numbers)
        counts = members.new_zeros((record_count * 3, 1)).index_add(
            0, rows.member_groups, members.new_ones((len(members), 1))
        )
        grouped = members.new_zeros((record_count * 3, 64)).index_add(
            0, rows.member_groups, members
        )
        grouped = torch.cat(
            (grouped / counts.clamp_min(1).sqrt(), (counts > 0).to(grouped.dtype)), -1
        ).reshape(record_count, 3 * 65)
        records = self.record_output(
            torch.cat(
                (
                    rows.numbers,
                    embedding(rows.kinds),
                    embedding(rows.choices),
                    names(rows.choice_names),
                    grouped,
                ),
                -1,
            )
        )
        if self.gru is not None:
            if rows.fold is not None:
                # A predecessor is evaluated once per execution chunk. Shared
                # descendants contribute gradients to that same state; no state
                # survives the forward/backward or a parameter update.
                states = records.new_zeros((rows.fold.node_count + 1, 64))
                if rows.fold.root_states is not None:
                    assert rows.fold.root_nodes is not None
                    states = states.index_copy(
                        0, rows.fold.root_nodes, rows.fold.root_states
                    )
                for level in rows.fold.levels:
                    packed = PackedSequence(
                        records[level.records], torch.tensor(level.schedule.sizes)
                    )
                    _, hidden = self.gru(packed, states[level.parents][None])
                    states = states.index_copy(
                        0, level.nodes, hidden[-1][level.schedule.restore]
                    )
                if rows.fold.retain is not None:
                    rows.fold.retain(states, references, records)
                summaries = self._summaries(states[rows.fold.tails])
                return summaries * (rows.fold.tails != 0)[:, None, None]
            packed = PackedSequence(
                records[rows.schedule.gather], torch.tensor(rows.schedule.sizes)
            )
            _, hidden = self.gru(packed)
            summaries = self._summaries(hidden[-1][rows.schedule.restore])
            result = records.new_zeros((batch, self.summaries, board_width))
            return result.index_copy(0, rows.nonempty_batches, summaries)
        assert self.transformer is not None and self.queries is not None
        lengths = records.new_tensor(rows.lengths)
        recency = (lengths[rows.record_batches] - 1 - rows.record_slots)[:, None] / 128
        records = records + self.recency(recency)
        sequence = records.new_zeros((batch, rows.width, 64))
        sequence = sequence.index_put((rows.record_batches, rows.record_slots), records)
        present = torch.zeros(
            (batch, rows.width), dtype=torch.bool, device=records.device
        )
        present = present.index_put(
            (rows.record_batches, rows.record_slots),
            torch.ones_like(rows.record_batches, dtype=torch.bool),
        )
        sequence = torch.cat((sequence, self.queries.expand(batch, -1, -1)), 1)
        padding = torch.cat(
            (
                ~present,
                torch.zeros(
                    (batch, self.summaries), dtype=torch.bool, device=records.device
                ),
            ),
            1,
        )
        summaries = (
            self.output(
                self.transformer(sequence, src_key_padding_mask=padding)[
                    :, -self.summaries :
                ]
            )
            + self.memory_type
        )
        nonempty = present.any(1).to(summaries.dtype)
        return self.join_norm(summaries) * nonempty[:, None, None]
