"""`ActorCritic` in MLX: the same encoder, attention stack, action heads and critic.

`MLXActorCritic` mirrors `manabot.backends.torch.policy.ActorCritic` module for
module, for the configurations `require_supported` admits (the w160l8 regime's: a
catalog world with catalog record ids, table attention, masked-mean scalar
critic, policy input v9). Its float arithmetic follows the PyTorch forward line
by line; the integer bookkeeping comes precomputed in a `Prepared` batch
(`manabot.policy.prepare`).

The catalog GRU always encodes every definition. PyTorch learning encodes
only the rows a batch references; the others never reach the loss, so their
outputs receive no gradient either way. GRU input and recurrent biases remain
separate parameters, preserving Adam state and global gradient-norm clipping.

Parameter paths equal the PyTorch state-dict names except where MLX containers
differ: `nn.Sequential` children live under `layers`, attention in-projections
are `query_proj`/`key_proj`/`value_proj`, and GRUs hold `Wx`, `Wh`, `input_bias`, `hidden_bias`
(`manabot.backends.mlx.checkpoint` maps them from the canonical names).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from manabot.backends.mlx.arrays import to_numpy
from manabot.backends.mlx.rl import NumericalError
from manabot.env.observation import ObservationBuffers, ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy import admission
from manabot.policy.checkpoint import Dimensions
from manabot.policy.prepare import (
    Packed,
    Prepared,
    Tables,
    catalog_schedule,
    prepare,
)

# -- building blocks -----------------------------------------------------------------


class Projection(nn.Module):
    """`ProjectionLayer`: Linear, ReLU, Linear."""

    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(inputs, outputs), nn.ReLU(), nn.Linear(outputs, outputs)
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.projection(x)


def _gru_cell(
    x: mx.array, previous: mx.array, recurrent_weight: mx.array, hidden_bias: mx.array
) -> mx.array:
    """One GRU step: `x` [L,3H] (input projection with bias), `previous` [L,H]."""
    hidden = previous.shape[1]
    recurrent = previous @ recurrent_weight.T + hidden_bias
    gates = mx.sigmoid(x[:, : 2 * hidden] + recurrent[:, : 2 * hidden])
    reset, update = gates[:, :hidden], gates[:, hidden:]
    new = mx.tanh(x[:, 2 * hidden :] + reset * recurrent[:, 2 * hidden :])
    return (1 - update) * new + update * previous


# `mx.compile` fuses each step's elementwise gate arithmetic; it traces once per
# live-row count, which the static catalog schedule repeats on every call.
_compiled_gru_cell = mx.compile(_gru_cell)


class PackedGRU(nn.Module):
    """A one-layer GRU over packed variable-length sequences (`Packed`).

    Gates are reset, update, new, as in PyTorch and `mlx.nn.GRU`:
    r, z = sigmoid(x Wx[:2H]^T + input_bias[:2H] + h Wh[:2H]^T + hidden_bias[:2H]),
    n = tanh(x Wx[2H:]^T + input_bias[2H:] + r * (h Wh[2H:]^T + hidden_bias[2H:])),
    h' = (1 - z) n + z h. Every sequence starts from a zero state.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        bound = hidden_size**-0.5
        self.Wx = mx.random.uniform(-bound, bound, (3 * hidden_size, input_size))
        self.Wh = mx.random.uniform(-bound, bound, (3 * hidden_size, hidden_size))
        self.input_bias = mx.zeros((3 * hidden_size,))
        self.hidden_bias = mx.zeros((3 * hidden_size,))

    def run(
        self, table: mx.array, steps: Sequence[mx.array]
    ) -> tuple[mx.array, mx.array]:
        """Step t reads rows `steps[t]` of `table` [T,D] (its live sequences, in
        sorted order) -> (every step's state [T,H] in packed order, each
        sequence's final state [S,H] in sorted order).

        Each step gathers its inputs from `table` rather than slicing one
        projected array: MLX materializes a whole-input zero cotangent for
        every slice or split output, which makes the backward quadratic in
        the number of steps (15.5 s against 0.39 s at 713 steps on the M2).
        """
        state = mx.zeros((steps[0].shape[0], self.hidden_size), dtype=table.dtype)
        outputs: list[mx.array] = []
        finished: list[mx.array] = []
        for step, index in enumerate(steps):
            live = index.shape[0]
            x = mx.take(table, index, axis=0) @ self.Wx.T + self.input_bias
            state = _compiled_gru_cell(x, state[:live], self.Wh, self.hidden_bias)
            outputs.append(state)
            following = steps[step + 1].shape[0] if step + 1 < len(steps) else 0
            if following < live:
                finished.append(state[following:])
        return mx.concatenate(outputs), mx.concatenate(finished[::-1])


def packed_steps(rows: np.ndarray, schedule: Packed[np.ndarray]) -> list[mx.array]:
    """Split packed-order `rows` into the schedule's steps (`PackedGRU.run`)."""
    return [mx.array(part) for part in np.split(rows, np.cumsum(schedule.sizes)[:-1])]


def _take(table: mx.array, rows: mx.array) -> mx.array:
    return mx.take(table, rows, axis=0)


def attend(mha: nn.MultiHeadAttention, x: mx.array, mask: mx.array) -> mx.array:
    """`nn.MultiheadAttention` self-attention over `x` [B,N,D] with the packed
    in-projection split into `query_proj`/`key_proj`/`value_proj`; `mask`
    [B,1,1,N] is additive."""
    batch, count, dims = x.shape
    heads = mha.num_heads

    def split(y: mx.array) -> mx.array:
        return y.reshape(batch, count, heads, dims // heads).transpose(0, 2, 1, 3)

    attended = mx.fast.scaled_dot_product_attention(
        split(mha.query_proj(x)),
        split(mha.key_proj(x)),
        split(mha.value_proj(x)),
        scale=(dims // heads) ** -0.5,
        mask=mask,
    )
    return mha.out_proj(attended.transpose(0, 2, 1, 3).reshape(batch, count, dims))


class ObjectAttention(nn.Module):
    """`GameObjectAttention`: perspective, self-attention, two post-norm blocks."""

    def __init__(
        self, dims: int, heads: int, feedforward: int, *, ownership: bool
    ) -> None:
        super().__init__()
        if ownership:
            self.perspective = mx.zeros((dims,))
        self.mha = nn.MultiHeadAttention(dims, heads, bias=True)
        self.norm1 = nn.LayerNorm(dims, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(dims, feedforward), nn.ReLU(), nn.Linear(feedforward, dims)
        )
        self.norm2 = nn.LayerNorm(dims, eps=1e-5)

    def __call__(
        self, objects: mx.array, sign: mx.array, mask: mx.array, keep: mx.array
    ) -> mx.array:
        """`objects` [B,N,D]; `sign` [1,N,1] is +1 for the acting viewer's
        objects; `mask` [B,1,1,N] is additive (-inf on padding keys); `keep`
        [B,N,1] zeros padding outputs."""
        owned = objects + sign * self.perspective if "perspective" in self else objects
        attended = attend(self.mha, owned, mask)
        x = self.norm1(owned + attended)
        return self.norm2(x + self.mlp(x)) * keep


class SemanticCards(nn.Module):
    def __init__(self, dims: Dimensions) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(dims.token_rows, dims.hidden)
        self.encoder = PackedGRU(dims.hidden, dims.hidden)
        self.decklist_projection = nn.Linear(4 * dims.hidden, dims.hidden)


class Facts(nn.Module):
    def __init__(self, dims: Dimensions) -> None:
        super().__init__()
        self.kind = nn.Embedding(dims.fact_kinds + 1, dims.hidden)
        self.numeric = nn.Linear(6, dims.hidden)
        self.program_numeric = nn.Linear(1, dims.hidden)
        self.program_sequence = PackedGRU(dims.hidden, dims.hidden)
        self.output = nn.Linear(dims.hidden, dims.hidden)


class RecordReducer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gru = PackedGRU(width, width)
        self.output = nn.Linear(width, width)


class SharedEffects(nn.Module):
    def __init__(self, dims: Dimensions) -> None:
        super().__init__()
        self.detail_kind = nn.Embedding(20, dims.hidden)
        self.detail_number = nn.Linear(1, dims.hidden)
        self.header = nn.Linear(8 + 2 * dims.hidden, dims.hidden)
        self.reducer = RecordReducer(dims.hidden)


class NameCandidates(nn.Module):
    def __init__(self, dims: Dimensions) -> None:
        super().__init__()
        self.embedding = nn.Embedding(dims.name_rows, dims.hidden)


class Continuation(nn.Module):
    def __init__(self, dims: Dimensions) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(dims.relation - 5 + 4 * dims.hidden, dims.hidden),
            nn.Tanh(),
            nn.Linear(dims.hidden, dims.hidden),
        )


# -- the policy ----------------------------------------------------------------------


@dataclass
class _Static:
    """Per-model constants derived from `Tables`, kept off the parameter tree."""

    catalog_steps: list[mx.array]  # token ids of each catalog GRU step
    catalog: Packed[np.ndarray]
    named_rows: mx.array  # program rows of names that have a definition
    named_names: mx.array  # their name rows
    sign: mx.array  # [1,N,1]
    tables: Tables
    space: ObservationSpace
    cache: dict[str, mx.array] = field(default_factory=dict)


class MLXActorCritic(nn.Module):
    def __init__(
        self,
        dims: Dimensions,
        tables: Tables,
        config: ModelConfig,
        space: ObservationSpace,
    ) -> None:
        super().__init__()
        self.dims = dims
        # The model configuration the weights were trained under, as the
        # torch policy carries it; the updates read its `value_kind`.
        self.config = config
        hidden = dims.hidden
        self.semantic_cards = SemanticCards(dims)
        self.player_embedding = Projection(dims.player, hidden)
        self.card_embedding = Projection(dims.card, hidden)
        self.perm_embedding = Projection(dims.permanent, hidden)
        self.action_embedding = Projection(dims.action - 2, hidden)
        self.attention = ObjectAttention(
            hidden, dims.heads, dims.feedforward, ownership=True
        )
        self.action_layer = Projection((dims.focus + 1) * hidden, hidden)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Identity(),  # MeanPoolingLayer's slot; pooling is masked below
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.extra_attention = [
            ObjectAttention(hidden, dims.heads, dims.feedforward, ownership=False)
            for _ in range(dims.layers - 1)
        ]
        self.link_from_target = [nn.Linear(hidden, hidden) for _ in range(dims.links)]
        self.link_to_target = [nn.Linear(hidden, hidden) for _ in range(dims.links)]
        self.facts = Facts(dims)
        self.shared_effects = SharedEffects(dims)
        self.name_candidates = NameCandidates(dims)
        self.continuation = Continuation(dims)
        tokens, schedule = catalog_schedule(tables)
        named = np.nonzero(tables.name_definitions != 0)[0]
        sign = np.where(tables.is_agent, 1.0, -1.0).astype(np.float32)
        # Underscore attributes outside the module tree: never parameters.
        object.__setattr__(
            self,
            "_static",
            _Static(
                packed_steps(tokens, schedule),
                schedule,
                mx.array(tables.name_program_rows[named]),
                mx.array(named),
                mx.array(sign[None, :, None]),
                tables,
                space,
            ),
        )

    def prepare(
        self, obs: ObservationBuffers, *, inference: bool = False
    ) -> Prepared[np.ndarray]:
        """Admit and index raw host observations; malformed inputs are refused."""
        self.check_observation(obs)
        return self.prepare_admitted(obs, inference=inference)

    def prepare_admitted(
        self, obs: ObservationBuffers, *, inference: bool = False
    ) -> Prepared[np.ndarray]:
        """Index unchanged rows previously checked for this policy's configuration."""
        return prepare(
            obs,
            self._static.tables,
            inference=inference,
            binding=admission.Spec(
                self.config, self._static.space, self._static.tables, 0
            ).binding,
        )

    def check_observation(self, obs: ObservationBuffers) -> None:
        """The shared admission checks on raw rows, before booleans or lookups."""
        tables = self._static.tables
        mask = np.asarray(obs["actions_valid"])
        try:
            legal = admission.legal_mask(mask)
        except ValueError as error:
            # Allocate framework evidence only for a rejected host input.
            failed = ~(np.isfinite(mask) & ((mask == 0) | (mask == 1)))
            raise NumericalError(
                "invalid_action_mask",
                {"mask": mx.array(mask), "failed": mx.array(failed)},
            ) from error
        admission.check(
            obs, admission.Spec(self.config, self._static.space, tables, 0), legal
        )

    # Programs ----------------------------------------------------------------------

    def programs(self) -> mx.array:
        """Encode the catalog into [definitions + 1, D]; row 0 is zero padding."""
        static: _Static = self._static
        cards = self.semantic_cards
        _, finished = cards.encoder.run(
            cards.token_embedding.weight, static.catalog_steps
        )
        encoded = _take(finished, mx.array(static.catalog.restore))
        return mx.concatenate([mx.zeros((1, encoded.shape[1]), encoded.dtype), encoded])

    def cache_programs(self) -> None:
        """Encode the catalog once for inference with fixed weights (collection)."""
        programs = self.programs()
        mx.eval(programs)
        self._static.cache["programs"] = programs

    def clear_cache(self) -> None:
        self._static.cache.clear()

    def evaluate(self, batch: Prepared[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Evaluated logits [B,A] and values [B] as NumPy (`BackendPolicy.evaluate`)."""
        logits, values = self(batch)
        mx.eval(logits, values)
        return to_numpy(logits), to_numpy(values)[:, 0]

    def forward_distribution(
        self, batch: Prepared[np.ndarray]
    ) -> tuple[mx.array, mx.array]:
        """The training forward the updates differentiate: `__call__`."""
        return self(batch)

    def _public(self, programs: mx.array) -> mx.array:
        """`NameCandidateEncoder.public_definitions`: add names to their programs."""
        static: _Static = self._static
        names = _take(self.name_candidates.embedding.weight, static.named_names)
        return programs.at[static.named_rows].add(names)

    # Forward -----------------------------------------------------------------------

    def __call__(self, batch: Prepared[np.ndarray]) -> tuple[mx.array, mx.array]:
        """Policy logits [B,A] (-1e8 off the legal support) and critic [B,1]."""
        if not isinstance(batch, Prepared):
            raise TypeError("policy forward requires a Prepared batch")
        static: _Static = self._static
        expected = admission.Spec(self.config, static.space, static.tables, 0).binding
        if batch.binding != expected:
            raise ValueError("prepared batch belongs to another policy configuration")
        dims = self.dims
        hidden = dims.hidden
        b = batch.batch
        programs = static.cache.get("programs")
        if programs is None:
            programs = self.programs()
        definitions = self._public(programs)
        cards = self.semantic_cards
        tokens = cards.token_embedding.weight

        def defs(rows: np.ndarray) -> mx.array:
            return _take(definitions, mx.array(rows.reshape(-1))).reshape(
                *rows.shape, hidden
            )

        # Objects: players, cards and permanents with their programs.
        card_programs = defs(batch.card_rows)
        known = (
            defs(batch.known_rows) * mx.array(batch.known_counts)[..., None] / 60
        ).sum(2)
        pooled = mx.einsum(
            "bsrh,bsrc->bsch", defs(batch.deck_rows), mx.array(batch.deck_counts)
        ).reshape(b, 2, 4 * hidden)
        known = known + cards.decklist_projection(pooled)
        agent_player = (
            self.player_embedding(mx.array(batch.agent_player)) + known[:, 0, None]
        )
        opponent_player = (
            self.player_embedding(mx.array(batch.opponent_player)) + known[:, 1, None]
        )
        agent_cards = (
            self.card_embedding(mx.array(batch.agent_cards)) + card_programs[:, 0]
        )
        opponent_cards = (
            self.card_embedding(mx.array(batch.opponent_cards)) + card_programs[:, 1]
        )
        permanents = self.perm_embedding(mx.array(batch.permanents)) + defs(
            batch.permanent_rows
        )
        objects = mx.concatenate(
            [
                agent_player,
                opponent_player,
                agent_cards,
                opponent_cards,
                permanents[:, 0],
                permanents[:, 1],
            ],
            axis=1,
        )
        count = objects.shape[1]
        objects = self._links(batch, objects)
        objects = self._facts(batch, objects, tokens, definitions)
        objects = self._effects(batch, objects, tokens, definitions)

        # Continuation relations: per-row encoding, shared and frame context.
        flat = objects.reshape(b * count, hidden)
        linked = [
            _take(flat, mx.array(index)).reshape(b, -1, hidden)
            * mx.array(mask)[..., None]
            for index, mask in zip(
                batch.relation_objects, batch.relation_object_masks, strict=True
            )
        ]
        relation_definitions = defs(batch.relation_rows)
        valid = mx.array(batch.relation_valid)
        encoded = (
            self.continuation.projection(
                mx.concatenate(
                    [
                        mx.array(batch.relation_features),
                        *linked,
                        relation_definitions[:, :, 0],
                        relation_definitions[:, :, 1],
                    ],
                    axis=-1,
                )
            )
            * valid[..., None]
        )
        rows = encoded.shape[1]
        shared = encoded.sum(1) / mx.maximum(valid.sum(1, keepdims=True), 1)
        frames = mx.array(batch.frame_indexes)
        pooled_frames = (
            mx.zeros((b * rows, hidden))
            .at[frames]
            .add(encoded.reshape(b * rows, hidden))
        )
        frame_counts = mx.zeros((b * rows,)).at[frames].add(valid.reshape(-1))
        frame_context = pooled_frames / mx.maximum(frame_counts, 1)[:, None]
        objects = objects + shared[:, None]

        # Attention over the object table.
        validity = mx.array(batch.validity)
        sign = static.sign
        attended, attended_validity = objects, validity
        columns = batch.attention_columns
        if columns is not None:
            selected = mx.array(columns)
            attended = mx.take(objects, selected, axis=1)
            attended_validity = mx.take(validity, selected, axis=1)
            sign = mx.take(sign, selected, axis=1)
        keep = attended_validity != 0
        mask = mx.where(keep, 0.0, -mx.inf)[:, None, None, :]
        keep = keep[..., None].astype(objects.dtype)
        encoded_objects = self.attention(attended, sign, mask, keep)
        for layer in self.extra_attention:
            encoded_objects = layer(encoded_objects, sign, mask, keep)
        if columns is not None:
            # Restore the original coordinates; unused columns are zero.
            positions = np.zeros(count, dtype=np.int64)
            positions[columns] = np.arange(len(columns))
            used = np.zeros(count, dtype=np.float32)
            used[columns] = 1
            encoded_objects = (
                mx.take(encoded_objects, mx.array(positions), axis=1)
                * mx.array(used)[None, :, None]
            )
        keep = (validity != 0)[..., None].astype(objects.dtype)

        # Legal actions: joined rows, focus objects, policy head.
        legal_batches = mx.array(batch.legal_batches)
        actions = self.action_embedding(mx.array(batch.action_features))
        actions = (
            actions
            + _take(frame_context, mx.array(batch.action_frames))
            * mx.array(batch.action_frame_mask)[:, None]
        )
        names = self.name_candidates.embedding.weight
        actions = (
            actions
            + _take(names, mx.array(batch.name_rows))
            * mx.array(batch.name_mask)[:, None]
        )
        prefix = mx.concatenate([mx.zeros((1, hidden)), mx.cumsum(names[1:], axis=0)])
        totals = _take(prefix, mx.array(batch.group_last)) - _take(
            prefix, mx.array(batch.group_first)
        )
        actions = (
            actions
            + totals
            / mx.array(batch.group_counts)[:, None]
            * mx.array(batch.group_mask)[:, None]
        )
        actions = actions + _take(programs, mx.array(batch.candidate_rows))
        actions = actions + _take(shared, legal_batches)
        post = encoded_objects.reshape(b * count, hidden)
        focus = batch.focus
        focused = _take(post, mx.array(focus.reshape(-1))).reshape(*focus.shape, hidden)
        focused = (focused * mx.array(batch.focus_mask)[..., None]).reshape(
            focus.shape[0], -1
        )
        informed = self.action_layer(mx.concatenate([actions, focused], axis=-1))
        scores = self.policy_head(informed)[:, 0]
        logits = mx.where(
            mx.array(batch.legal),
            _take(scores, mx.array(batch.logit_positions.reshape(-1))).reshape(
                batch.legal.shape
            ),
            -1e8,
        )

        # Masked-mean critic.
        value = self.value_head.layers
        projected = value[1](value[0](encoded_objects)) * keep
        pooled_value = projected.sum(1) / validity.sum(1, keepdims=True)
        return logits, value[5](value[4](value[3](pooled_value)))

    def _links(self, batch: Prepared[np.ndarray], objects: mx.array) -> mx.array:
        """`ActorCritic._join_permanent_links`."""
        b, count, hidden = objects.shape
        start = 2 + 2 * batch.agent_cards.shape[1]
        flat = objects.reshape(b * count, hidden)
        permanents = objects[:, start:]
        delta = mx.zeros_like(flat)
        local = mx.zeros_like(permanents)
        for targets, mask, from_target, to_target in zip(
            batch.link_targets,
            batch.link_masks,
            self.link_from_target,
            self.link_to_target,
            strict=True,
        ):
            linked = mx.array(mask)[..., None]
            index = mx.array(targets)
            delta = delta.at[index].add(
                (to_target(permanents) * linked).reshape(-1, hidden)
            )
            local = (
                local
                + from_target(_take(flat, index).reshape(permanents.shape)) * linked
            )
        delta = delta.reshape(b, count, hidden)
        local = mx.concatenate([mx.zeros((b, start, hidden)), local], axis=1)
        return objects + delta + local

    def _facts(
        self,
        batch: Prepared[np.ndarray],
        objects: mx.array,
        tokens: mx.array,
        definitions: mx.array,
    ) -> mx.array:
        """`FactEncoder.forward` with catalog ids."""
        facts = self.facts
        routes = batch.facts
        b, count, hidden = objects.shape
        shape = routes.kinds.shape

        def rows(table: mx.array, index: np.ndarray) -> mx.array:
            return _take(table, mx.array(index.reshape(-1))).reshape(*shape, hidden)

        ids = (
            rows(tokens, routes.id_rows) * mx.array(routes.id_mask)[..., None]
            + rows(tokens, routes.detail_rows) * mx.array(routes.detail_mask)[..., None]
            + rows(tokens, routes.token_rows) * mx.array(routes.token_mask)[..., None]
            + rows(definitions, routes.source_rows)
        )
        hidden_rows = (
            facts.kind(mx.array(routes.kinds))
            + ids
            + facts.numeric(mx.array(routes.columns))
        )
        programs = routes.programs
        if programs is not None:
            schedule = programs.schedule
            flat = hidden_rows.reshape(-1, hidden)
            program = _take(flat, mx.array(programs.rows)) + facts.program_numeric(
                _program_number(mx.array(programs.numbers))[:, None]
            )
            states, _ = facts.program_sequence.run(
                program, packed_steps(schedule.gather, schedule)
            )
            program = program + _take(states, mx.array(schedule.unpack))
            flat = mx.where(
                mx.array(programs.mask)[:, None],
                _take(program, mx.array(programs.positions)),
                flat,
            )
            hidden_rows = flat.reshape(*shape, hidden)
        encoded = (
            facts.output(mx.tanh(hidden_rows)) * mx.array(routes.linked)[..., None]
        )
        delta = (
            mx.zeros((b * count, hidden))
            .at[mx.array(routes.targets)]
            .add(encoded.reshape(-1, hidden))
        )
        return objects + delta.reshape(b, count, hidden)

    def _effects(
        self,
        batch: Prepared[np.ndarray],
        objects: mx.array,
        tokens: mx.array,
        definitions: mx.array,
    ) -> mx.array:
        """`SharedEffectEncoder.forward` with catalog ids."""
        routes = batch.effects
        if routes is None:
            return objects
        effects = self.shared_effects
        b, count, hidden = objects.shape
        flat = objects.reshape(b * count, hidden)
        controller = (
            _take(flat, mx.array(routes.controller))
            * mx.array(routes.controller_mask)[:, None]
        )
        source_object = (
            _take(flat, mx.array(routes.source_object))
            * mx.array(routes.source_object_mask)[:, None]
        )
        header = effects.header(
            mx.concatenate(
                [mx.array(routes.numeric), controller, source_object], axis=-1
            )
        )
        modification = _take(tokens, mx.array(routes.modification_rows))
        sources = _take(definitions, mx.array(routes.source_rows))
        detail = (
            _take(tokens, mx.array(routes.detail_rows))
            + effects.detail_kind(mx.array(routes.detail_kinds))
            + effects.detail_number(
                _program_number(mx.array(routes.detail_numbers))[:, None]
            )
            * mx.array(routes.detail_is_number)[:, None]
        )
        table = mx.concatenate(
            [
                mx.stack([modification, sources, mx.tanh(header)], axis=1).reshape(
                    -1, hidden
                ),
                detail,
            ]
        )
        sequence = _take(table, mx.array(routes.token_indexes))
        schedule = routes.schedule
        _, finished = effects.reducer.gru.run(
            sequence, packed_steps(schedule.gather, schedule)
        )
        pooled = _take(finished, mx.array(schedule.restore))
        encoded = effects.reducer.output(mx.tanh(pooled))
        delta = (
            mx.zeros_like(flat)
            .at[mx.array(routes.routes)]
            .add(_take(encoded, mx.array(routes.member_records)))
        )
        return objects + delta.reshape(b, count, hidden)


def _program_number(value: mx.array) -> mx.array:
    """Signed log magnitude in the framework graph, matching Torch's order."""
    return mx.sign(value) * mx.log1p(mx.abs(value)) / 10
