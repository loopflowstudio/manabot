"""`ActorCritic` in MLX: the same encoder, attention stack, action heads and critic.

`MLXActorCritic` mirrors `manabot.backends.torch.policy.ActorCritic` module for
module.
Its forward uses the integer lookups in a `Prepared` batch from
`manabot.policy.prepare`.

Learner batches select the catalog programs their observations reference.
Inference uses the full catalog schedule. The GRU keeps separate input and
recurrent biases for Adam and global gradient-norm clipping.

Compute precision is policy-owned. Unless given, it follows the model's width
(`manabot.backends.mlx.arrays.default_precision`): bfloat16 from
`BF16_MIN_HIDDEN_DIM`, float32 below. Half forwards cast weights
differentiably and restore float32 masters; inference and learning therefore
use the same dtype. Outputs remain float32.

Parameter paths equal the PyTorch state-dict names except where MLX containers
differ: `nn.Sequential` children live under `layers`, attention in-projections
are `query_proj`/`key_proj`/`value_proj`, and GRUs hold `Wx`, `Wh`, `input_bias`, `hidden_bias`
(`manabot.backends.mlx.checkpoint` maps them from the canonical names).
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from types import TracebackType
from typing import Literal, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
import numpy as np

from manabot.backends.mlx.arrays import (
    default_precision,
    mlx_device,
    precision_dtype,
    to_numpy,
)
from manabot.backends.mlx.decision_memory import DecisionMemory
from manabot.backends.mlx.gru import PackedGRU
from manabot.backends.mlx.live_tokens import attend_live
from manabot.backends.mlx.rl import (
    expected_outcome,
    outcome_probabilities,
)
from manabot.belief.encoding import (
    BeliefCheckpointBinding,
    BeliefEncodingSchema,
    BeliefRows,
)
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy import admission
from manabot.policy.buckets import (
    object_trimming,
)
from manabot.policy.checkpoint import Dimensions
from manabot.policy.collection_memory import CollectionMemory
from manabot.policy.prepare import (
    AttentionLayout,
    CatalogPrograms,
    CompactAttention,
    EventRows,
    FactLayout,
    PackedAttention,
    Prepared,
    Tables,
    prepare,
)
from manabot.policy.protocol import InferenceResult, Precision
from manabot.policy.readings import Readings
from manabot.policy.spans import (
    ObservationSpans,
    PolicyObservation,
    observation_spans,
)
import managym

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


def _take(table: mx.array, rows: mx.array) -> mx.array:
    return mx.take(table, rows, axis=0)


class RecentEventEncoder(nn.Module):
    """Join both public event references to current objects and pool the history."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(8 + 2 * (2 * width + 14), width),
            nn.Tanh(),
            nn.Linear(width, width),
        )

    def __call__(
        self, rows: EventRows[np.ndarray], objects: mx.array, definitions: mx.array
    ) -> mx.array:
        def categories(indices: np.ndarray, size: int) -> mx.array:
            return (mx.array(indices)[..., None] == mx.arange(size)).astype(
                objects.dtype
            )

        features = [
            categories(rows.kinds, 6),
            mx.array(rows.amounts),
            mx.log1p(mx.array(rows.ages))[..., None],
        ]
        flat = objects.reshape(-1, objects.shape[-1])
        for index in range(2):
            context = _take(flat, mx.array(rows.focus[..., index]))
            features.extend(
                (
                    _take(definitions, mx.array(rows.definitions[..., index])),
                    context * mx.array(rows.focus_mask[..., index, None]),
                    categories(rows.owners[..., index], 3),
                    categories(rows.zones[..., index], 8),
                    categories(rows.status[..., index], 3),
                )
            )
        encoded = self.projection(
            mx.concatenate(features, axis=-1).astype(objects.dtype)
        )
        return (encoded * mx.array(rows.validity)[..., None]).sum(1) / mx.maximum(
            mx.array(rows.counts).astype(objects.dtype), 1
        )


class ObjectAttention(nn.Module):
    """`GameObjectAttention`: perspective, self-attention, two post- or pre-norm blocks."""

    def __init__(
        self,
        dims: int,
        heads: int,
        feedforward: int,
        *,
        ownership: bool,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        if ownership:
            self.perspective = mx.zeros((dims,))
        self.mha = nn.MultiHeadAttention(dims, heads, bias=True)
        self.norm1 = nn.LayerNorm(dims, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(dims, feedforward), nn.ReLU(), nn.Linear(feedforward, dims)
        )
        self.norm2 = nn.LayerNorm(dims, eps=1e-5)

    def __call__(
        self,
        objects: mx.array,
        sign: mx.array,
        mask: mx.array,
        keep: mx.array,
        live: CompactAttention[mx.array] | None = None,
        summaries: mx.array | None = None,
    ) -> mx.array:
        """Attend over padded [B,N,D] or packed [L,D] objects. The sign is
        zero for a neutral critic token, which receives neither ownership nor memory."""
        if summaries is not None:
            delta = self.memory_projection(summaries.mean(1)) * mx.tanh(
                self.memory_gate_scale * self.memory_gate
            )
            objects = (
                objects
                + (delta[live.batches] if live is not None else delta[:, None])
                * (sign != 0)
                * keep
            )
        owned = objects + sign * self.perspective if "perspective" in self else objects
        attention_input = self.norm1(owned) if self.pre_norm else owned
        attended = (
            self.mha(attention_input, attention_input, attention_input, mask)
            if live is None
            else attend_live(self.mha, attention_input, live)
        )
        residual = owned + attended
        if self.pre_norm:
            return (residual + self.mlp(self.norm2(residual))) * keep
        x = self.norm1(residual)
        return self.norm2(x + self.mlp(x)) * keep


class AttentionReadout(nn.Module):
    """Pool valid final objects with one learned query for the critic alone."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.query = mx.random.normal((1, 1, width)) * width**-0.5
        self.mha = nn.MultiHeadAttention(width, heads, bias=True)

    def __call__(self, objects: mx.array, validity: mx.array) -> mx.array:
        query = mx.broadcast_to(self.query, (objects.shape[0], 1, objects.shape[-1]))
        mask = (validity != 0)[:, None, None, :]
        return self.mha(query, objects, objects, mask)[:, 0]


def padding_embedding(rows: int, width: int) -> nn.Embedding:
    """Match Torch's zero padding row on fresh token and name embeddings."""
    embedding = nn.Embedding(rows, width)
    embedding.weight = embedding.weight.at[0].multiply(0)
    return embedding


def zero_projection(width: int) -> nn.Linear:
    """Start additive and FiLM residuals at zero on fresh policies."""
    layer = nn.Linear(width, width)
    layer.weight = mx.zeros_like(layer.weight)
    layer.bias = mx.zeros_like(layer.bias)
    return layer


class SemanticCards(nn.Module):
    def __init__(self, dims: Dimensions, *, stream_norm: bool = False) -> None:
        super().__init__()
        self.token_embedding = padding_embedding(dims.token_rows, dims.hidden)
        self.encoder = PackedGRU(dims.hidden, dims.hidden)
        self.decklist_projection = nn.Linear(4 * dims.hidden, dims.hidden)
        self.library_projection = nn.Linear(dims.hidden, dims.hidden, bias=False)
        if stream_norm:
            self.hand_norm = nn.LayerNorm(dims.hidden, eps=1e-5)
            self.library_norm = nn.LayerNorm(dims.hidden, eps=1e-5)
            self.decklist_norm = nn.LayerNorm(dims.hidden, eps=1e-5)


class Facts(nn.Module):
    def __init__(self, dims: Dimensions, hashed_buckets: int) -> None:
        super().__init__()
        if hashed_buckets:
            self.ident = nn.Embedding(hashed_buckets, dims.hidden)
            self.detail = nn.Embedding(hashed_buckets, dims.hidden)
            self.source = nn.Embedding(hashed_buckets, dims.hidden)
        self.kind = nn.Embedding(dims.fact_kinds + 1, dims.hidden)
        self.numeric = nn.Linear(dims.fact, dims.hidden)
        self.program_numeric = nn.Linear(1, dims.hidden)
        self.program_sequence = PackedGRU(dims.hidden, dims.hidden)
        self.output = nn.Linear(dims.hidden, dims.hidden)


class RecordReducer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gru = PackedGRU(width, width)
        self.output = nn.Linear(width, width)


class SharedEffects(nn.Module):
    def __init__(self, dims: Dimensions, hashed_buckets: int) -> None:
        super().__init__()
        if hashed_buckets:
            self.modification = nn.Embedding(hashed_buckets, dims.hidden)
            self.detail = nn.Embedding(hashed_buckets, dims.hidden)
            self.source = nn.Embedding(hashed_buckets, dims.hidden)
        self.detail_kind = nn.Embedding(20, dims.hidden)
        self.detail_number = nn.Linear(1, dims.hidden)
        self.header = nn.Linear(dims.effect + 2 * dims.hidden, dims.hidden)
        self.reducer = RecordReducer(dims.hidden)


class NameCandidates(nn.Module):
    def __init__(self, dims: Dimensions) -> None:
        super().__init__()
        self.embedding = padding_embedding(dims.name_rows, dims.hidden)


class Continuation(nn.Module):
    def __init__(self, dims: Dimensions) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(dims.relation + 4 * dims.hidden, dims.hidden),
            nn.Tanh(),
            nn.Linear(dims.hidden, dims.hidden),
        )


# -- the policy ----------------------------------------------------------------------


@dataclass
class _Static:
    """Per-model constants derived from `Tables`, kept off the parameter tree."""

    named_rows: mx.array  # program rows of names that have a definition
    named_names: mx.array  # their name rows
    tables: Tables
    space: ObservationSpace
    cache: dict[str, mx.array] = field(default_factory=dict)
    binding_cache: admission.BindingCache = field(
        default_factory=admission.BindingCache
    )


def parameter_ids(parameters: dict[str, object]) -> tuple[int, ...]:
    return tuple(id(value) for _, value in tree_flatten(parameters))


class MLXActorCritic(nn.Module):
    _cast_state: tuple[tuple[int, ...], tuple[int, ...]] | None

    def __init__(
        self,
        dims: Dimensions,
        tables: Tables,
        config: ModelConfig,
        space: ObservationSpace,
        *,
        precision: Precision | None = None,
        bucketed_shapes: bool = False,
        live_tokens: bool = False,
    ) -> None:
        super().__init__()
        admission.history_config(config, space)
        if precision is None:
            precision = default_precision(dims.hidden)
        precision_dtype(precision)
        object.__setattr__(self, "_precision", precision)
        object.__setattr__(self, "_cast_state", None)
        self.dims = dims
        # The model configuration the weights were trained under, as the
        # torch policy carries it; the updates read its `value_kind`.
        self.config = config
        self.bucketed_shapes = bucketed_shapes
        self.live_tokens = live_tokens
        self.value_semantic: Literal["signed_outcome", "win_logit"] | None = (
            "signed_outcome" if config.value_kind == "categorical_wdl" else None
        )
        hidden = dims.hidden
        if tables.definition_rows is None:
            self.semantic_cards = None
            self.card_definitions = padding_embedding(
                managym.CARD_DEFINITION_COUNT + 1, hidden
            )
        else:
            self.semantic_cards = SemanticCards(dims, stream_norm=config.stream_norm)
        self.player_embedding = Projection(dims.player, hidden)
        self.card_embedding = Projection(dims.card, hidden)
        self.perm_embedding = Projection(dims.permanent, hidden)
        self.action_embedding = Projection(dims.action, hidden)
        if config.attention_on:
            self.attention = ObjectAttention(
                hidden,
                dims.heads,
                dims.feedforward,
                ownership=True,
                pre_norm=config.stream_norm,
            )
        self.action_layer = Projection((dims.focus + 1) * hidden, hidden)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Identity(),  # MeanPoolingLayer's slot; pooling follows the saved mode
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3 if config.value_kind == "categorical_wdl" else 1),
        )
        self.extra_attention = [
            ObjectAttention(
                hidden,
                dims.heads,
                dims.feedforward,
                ownership=False,
                pre_norm=config.stream_norm,
            )
            for _ in range(dims.layers - 1)
        ]
        if config.value_aggregation == "value_token":
            self.value_token = mx.random.normal((1, 1, hidden)) * hidden**-0.5
        if config.policy_film:
            self.policy_film = zero_projection(hidden)
        self.link_from_target = [nn.Linear(hidden, hidden) for _ in range(dims.links)]
        self.link_to_target = [nn.Linear(hidden, hidden) for _ in range(dims.links)]
        hashed_buckets = 0 if config.catalog_records() else tables.record_buckets
        self.facts = Facts(dims, hashed_buckets)
        self.shared_effects = SharedEffects(dims, hashed_buckets)
        self.name_candidates = NameCandidates(dims)
        if config.decision_memory != "none":
            self.decision_memory = DecisionMemory(
                config.decision_memory, config.decision_memory_summaries, hidden
            )
            for layer in (self.attention, *self.extra_attention):
                layer.memory_gate = mx.zeros(())
                layer.memory_gate_scale = config.decision_memory_gate_scale
                layer.memory_projection = nn.Linear(hidden, hidden, bias=False)
        if config.recent_events:
            self.recent_events = RecentEventEncoder(hidden)
        self.continuation = Continuation(dims)
        self.stack_embedding = Projection(dims.stack, hidden)
        self.stack_source = zero_projection(hidden)
        if config.stream_norm:
            self.final_norm = nn.LayerNorm(hidden, eps=1e-5)
            streams = ["player", "card", "permanent", "definition", "context", "stack"]
            self.stream_norms = {
                name: nn.LayerNorm(hidden, eps=1e-5) for name in streams
            }
        self.belief_binding: BeliefCheckpointBinding | None = None
        if config.belief_count_buckets > 0:
            if config.belief_card_vocab_size < 1:
                raise ValueError(
                    "belief_card_vocab_size must be positive when belief input is enabled"
                )
            self.belief_card_embedding = nn.Embedding(
                config.belief_card_vocab_size, hidden
            )
            self.belief_owner_embedding = nn.Embedding(
                config.belief_owner_role_vocab_size, hidden
            )
            self.belief_zone_embedding = nn.Embedding(
                config.belief_hidden_zone_vocab_size, hidden
            )
            self.belief_count_embedding = Projection(
                config.belief_count_buckets, hidden
            )
            self.belief_global_embedding = Projection(2, hidden)
        if config.focus_film:
            self.focus_film = nn.Linear(dims.focus * hidden, hidden, bias=False)
        # Draw the critic query last so the actor starts from the same weights
        # as a masked-mean policy constructed with the same seed.
        if config.value_aggregation == "attention_readout":
            self.value_readout = AttentionReadout(hidden, dims.heads)
        named = (
            np.flatnonzero(tables.name_definitions)
            if self.has_programs
            else np.empty(0, np.int64)
        )
        # Underscore attributes outside the module tree: never parameters.
        object.__setattr__(
            self,
            "_static",
            _Static(
                mx.array(tables.name_program_rows[named]),
                mx.array(named),
                tables,
                space,
            ),
        )

    @property
    def readings(self) -> Readings:
        return self._static.tables.layout.readings

    def buffers(self) -> dict[str, np.ndarray]:
        """Lookup buffers under their canonical state-dict names (`Tables.BUFFERS`)."""
        tables = self._static.tables
        exported: dict[str, np.ndarray] = {}
        for field_name, buffer_name in Tables.BUFFERS.items():
            value = getattr(tables, field_name)
            if value is None or (value.size == 0 and field_name != "is_agent"):
                continue
            exported[buffer_name] = (
                value[None] if field_name == "is_agent" and value.ndim == 1 else value
            )
        return exported

    def object_trimming(self) -> bool:
        """Whether buckets may trim object axes: padding objects must be inert."""
        return object_trimming(self.config)

    @property
    def inference_device(self) -> mx.Device | mx.DeviceType:
        return getattr(self, "_inference_device", mx.default_device())

    @property
    def device_type(self) -> str:
        return "cpu" if self.inference_device == mx.cpu else "mps"

    @property
    def parameter_count(self) -> int:
        return sum(value.size for _, value in tree_flatten(self.parameters()))

    @property
    def belief_count_buckets(self) -> int:
        return self.config.belief_count_buckets

    def admit_belief(self, belief_schema: BeliefEncodingSchema) -> None:
        belief_schema.admit_model(self.config)

    @staticmethod
    def normalize_logits(logits: np.ndarray, *, double: bool = False) -> np.ndarray:
        if double:
            # MLX has no float64; search's host-only normalization retains it.
            values = np.asarray(logits, np.float64)
            weights = np.exp(values - values.max(axis=-1, keepdims=True))
            return weights / weights.sum(axis=-1, keepdims=True)
        return to_numpy(mx.softmax(mx.array(logits), axis=-1))

    @property
    def resident_bytes(self) -> int:
        tables = self._static.tables
        buffers = [getattr(tables, name) for name in Tables.BUFFERS]
        return (
            sum(value.nbytes for _, value in tree_flatten(self.parameters()))
            + sum(value.nbytes for value in buffers if value is not None)
            + (
                (len(tables.catalog_lengths) + 1) * self.dims.hidden * 4
                if self.has_programs
                else 0
            )
        )

    @property
    def has_programs(self) -> bool:
        return self.semantic_cards is not None

    def inference_scope(self, *, cache_catalog: bool = True) -> "PolicyInference":
        return PolicyInference(self, cache_catalog=cache_catalog)

    @staticmethod
    def sample_rows(
        logits: np.ndarray, masks: np.ndarray, seeds: list[int], deterministic: bool
    ) -> list[int]:
        valid = masks != 0
        if (
            logits.shape != masks.shape
            or not valid.any(1).all()
            or not np.isfinite(logits[valid]).all()
        ):
            raise ValueError("invalid legal logits or empty action support")
        actions = []
        for row, seed in enumerate(seeds):
            values = mx.array(logits[row])
            action = (
                int(mx.argmax(values))
                if deterministic
                else int(mx.random.categorical(values, key=mx.random.key(seed)))
            )
            if not valid[row, action]:
                raise ValueError("policy selected a masked offer")
            actions.append(action)
        return actions

    @property
    def observation_space(self) -> ObservationSpace:
        return self._static.space

    def to(self, device: str) -> "MLXActorCritic":
        """Select the inference stream on MLX's shared-memory devices."""
        object.__setattr__(self, "_inference_device", mlx_device(device))
        return self

    def sample(
        self,
        batch: Prepared[np.ndarray],
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        sampling: Literal["categorical", "multinomial"] = "categorical",
    ) -> np.ndarray:
        if not batch.legal_support:
            raise ValueError("No valid actions available")
        with mx.stream(self.inference_device):
            logits, _ = self(batch)
            actions = (
                mx.argmax(logits, axis=-1)
                if deterministic
                else mx.random.categorical(logits / temperature)
            )
            mx.eval(actions)
        return to_numpy(actions).astype(np.int64)

    def sample_logits(
        self,
        logits: np.ndarray,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        sampling: Literal["categorical", "multinomial"] = "categorical",
    ) -> np.ndarray:
        with mx.stream(self.inference_device):
            values = mx.array(logits)
            actions = (
                mx.argmax(values, axis=-1)
                if deterministic
                else mx.random.categorical(values / temperature)
            )
            mx.eval(actions)
        return to_numpy(actions).astype(np.int64)

    def win_probabilities(self, batch: Prepared[np.ndarray]) -> np.ndarray:
        with mx.stream(self.inference_device):
            _, values = self(batch)
            probabilities = mx.sigmoid(values)
            mx.eval(probabilities)
        return to_numpy(probabilities)

    @property
    def precision(self) -> Precision:
        return self._precision

    @property
    def compute_dtype(self) -> mx.Dtype:
        """The dtype the weights hold now: float32 masters, or the selected
        precision's casts inside `forward_precision`."""
        return self.player_embedding.projection.layers[0].weight.dtype

    def set_precision(self, precision: Precision) -> None:
        """Select compute precision without changing stored master weights."""
        precision_dtype(precision)
        if self.compute_dtype != mx.float32:
            raise ValueError("change precision outside the forward precision scope")
        if precision != self.precision:
            self.clear_cache()
            object.__setattr__(self, "_precision", precision)

    @contextmanager
    def forward_precision(self) -> Iterator[None]:
        """Differentiable compute casts, restoring float32 masters on exit.

        A fixed-weight inference loop may hold this scope across calls to reuse
        cast weights. Close it before differentiating or updating the policy.
        Public forwards enter it automatically; nested scopes reuse the casts.
        """
        dtype = precision_dtype(self.precision)
        if self.compute_dtype == dtype:
            yield
            return
        masters = self.parameters()
        casts = tree_map(lambda p: p.astype(dtype), masters)
        self.update(casts)
        try:
            # A frozen inference scope can recognize these casts of its own masters.
            object.__setattr__(
                self, "_cast_state", (parameter_ids(masters), parameter_ids(casts))
            )
            yield
        finally:
            self.update(masters)
            object.__setattr__(self, "_cast_state", None)

    def memory_gates(self) -> dict[str, float]:
        """Effective memory gates at the update reporting boundary."""
        if self.config.decision_memory == "none":
            return {}
        return {
            f"gate_{i}": float(mx.tanh(layer.memory_gate_scale * layer.memory_gate))
            for i, layer in enumerate((self.attention, *self.extra_attention))
        }

    def _binding(self) -> str:
        return self._static.binding_cache.get(
            self.config, self.observation_space, self.readings
        )

    def prepare(
        self,
        obs: PolicyObservation,
        *,
        inference: bool = False,
        belief: BeliefRows[np.ndarray] | None = None,
        memory_roots: Mapping[str, int] | None = None,
    ) -> Prepared[np.ndarray]:
        """Admit and index raw host observations; malformed inputs are refused."""
        measured = self._admit_observation(obs, belief)
        return self._prepare_rows(measured, inference, belief, memory_roots)

    def prepare_admitted(
        self,
        obs: PolicyObservation,
        *,
        inference: bool = False,
        belief: BeliefRows[np.ndarray] | None = None,
        needs: np.ndarray | None = None,
        memory_roots: Mapping[str, int] | None = None,
    ) -> Prepared[np.ndarray]:
        """Index unchanged rows previously checked for this policy's configuration."""
        space = self._static.space
        rows = observation_spans(
            obs,
            space,
            bucketed=self.bucketed_shapes,
            objects=self.object_trimming(),
            minimum_needs=needs,
        )
        return self._prepare_rows(rows, inference, belief, memory_roots)

    def _prepare_rows(
        self,
        obs: ObservationSpans,
        inference: bool,
        belief: BeliefRows[np.ndarray] | None,
        memory_roots: Mapping[str, int] | None,
    ) -> Prepared[np.ndarray]:
        # Learner compaction requires inert object padding.
        compact_learner = (
            not inference and self.bucketed_shapes and self.object_trimming()
        )
        attention: AttentionLayout = "padded"
        if self.live_tokens and self.config.attention_on:
            attention = "live_tokens"
        elif inference:
            attention = "used_columns"
        elif compact_learner:
            attention = "packed_groups"
        facts: FactLayout = "padded"
        if inference and not self.bucketed_shapes:
            facts = "used_columns"
        elif compact_learner:
            facts = "live_rows"
        return prepare(
            obs,
            self._static.tables,
            attention=attention,
            belief=belief,
            facts=facts,
            select_programs=not inference,
            binding=self._binding(),
            memory_summaries=self.config.decision_memory_summaries
            if self.config.decision_memory != "none"
            else 0,
            memory_fold=self.config.decision_memory == "gru",
            memory_roots=memory_roots,
            value_token=self.config.value_aggregation == "value_token",
        )

    def check_observation(
        self, obs: PolicyObservation, *, belief: BeliefRows[np.ndarray] | None = None
    ) -> np.ndarray | None:
        """The shared admission checks on raw rows, before booleans or lookups."""
        measured = self._admit_observation(obs, belief)
        return measured.needs

    def _admit_observation(
        self, obs: PolicyObservation, belief: BeliefRows[np.ndarray] | None
    ) -> ObservationSpans:
        definition_count = (
            0 if self.has_programs else self.card_definitions.weight.shape[0]
        )
        spec = admission.Spec(
            self.config, self._static.space, self._static.tables, definition_count
        )
        return admission.admit(obs, spec, bucketed=self.bucketed_shapes, belief=belief)

    # Programs ----------------------------------------------------------------------

    def programs(
        self, programs: CatalogPrograms[np.ndarray] | None = None
    ) -> mx.array | None:
        """Encode selected definitions at their one-based catalog rows; others stay
        zero. None when the model has no semantic catalog."""
        with self.forward_precision():
            cards = self.semantic_cards
            if cards is None:
                return None
            full = self._static.tables.full_programs
            if programs is None:
                programs = full
            weights = cards.token_embedding.weight
            table = mx.zeros(
                (len(full.rows) + 1, self.dims.hidden), dtype=weights.dtype
            )
            if not len(programs.rows):
                return table
            _, finished = cards.encoder.run(
                weights, mx.array(programs.tokens), programs.schedule.sizes
            )
            encoded = _take(finished, mx.array(programs.schedule.restore))
            return table.at[mx.array(programs.rows)].add(encoded)

    def cache_programs(self) -> None:
        """Encode the catalog once for inference with fixed weights (collection)."""
        programs = self.programs()
        if programs is not None:
            mx.eval(programs)
            self._static.cache["programs"] = programs

    def clear_cache(self) -> None:
        self._static.cache.clear()

    def evaluate(self, batch: Prepared[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Evaluated logits [B,A] and values [B] as NumPy (`BackendPolicy.evaluate`)."""
        with mx.stream(self.inference_device):
            logits, values = self(batch)
            mx.eval(logits, values)
        return to_numpy(logits), to_numpy(values)

    def evaluate_probabilities(
        self, batch: Prepared[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        with mx.stream(self.inference_device):
            logits, values = self(batch)
            probabilities = mx.softmax(logits, axis=-1)
            mx.eval(probabilities, values)
        return to_numpy(probabilities), to_numpy(values)

    def forward_distribution(
        self, batch: Prepared[np.ndarray]
    ) -> tuple[mx.array, mx.array]:
        """Differentiable policy logits and raw scalar [B,1] or WDL [B,3] critic."""
        with self.forward_precision():
            logits, values = self.forward_heads(batch)
            return logits, values.astype(mx.float32)

    def _public(self, programs: mx.array) -> mx.array:
        """`NameCandidateEncoder.public_definitions`: the definition vectors objects
        read, with names added only under `definition_names`."""
        if not self.config.definition_names:
            return programs
        static: _Static = self._static
        names = _take(self.name_candidates.embedding.weight, static.named_names)
        return programs.at[static.named_rows].add(names)

    def _stream(self, name: str, x: mx.array) -> mx.array:
        """Normalize one stream summed into object tokens (`stream_norm`)."""
        if "stream_norms" not in self:
            return x
        return self.stream_norms[name](x)

    def _definition_table(self, table: mx.array) -> mx.array:
        """Normalize a definition table [definitions + 1, D] with row 0 kept zero."""
        if "stream_norms" not in self:
            return table
        normed = self.stream_norms["definition"](table[1:])
        return mx.concatenate([mx.zeros((1, table.shape[1]), normed.dtype), normed])

    def _masked_norm(
        self, norm: nn.LayerNorm | None, stream: mx.array, weights: np.ndarray
    ) -> mx.array:
        """A normalized knowledge source contributes zero for seats with no entries."""
        if norm is None:
            return stream
        present = (
            (weights > 0).reshape(weights.shape[0], weights.shape[1], -1).any(axis=-1)
        )
        return norm(stream) * self._floating(present)[..., None]

    # Forward -----------------------------------------------------------------------

    def _floating(self, values: np.ndarray) -> mx.array:
        """Features and masks follow compute weights, never promote half activations."""
        return mx.array(values).astype(self.compute_dtype)

    def __call__(self, batch: Prepared[np.ndarray]) -> tuple[mx.array, mx.array]:
        """Policy logits [B,A] and one acting-viewer value per decision."""
        logits, raw = self.forward_distribution(batch)
        if self.config.value_kind == "categorical_wdl":
            return logits, expected_outcome(outcome_probabilities(raw))
        return logits, raw.squeeze(-1)

    def forward_heads(self, batch: Prepared[np.ndarray]) -> tuple[mx.array, mx.array]:
        """Inside forward_precision, expose the critic before its host-facing float32 cast."""
        if not isinstance(batch, Prepared):
            raise TypeError("policy forward requires a Prepared batch")
        if batch.guard is not None:
            batch.guard(self)
        static: _Static = self._static
        if batch.binding != self._binding():
            raise ValueError("prepared batch belongs to another policy configuration")
        hidden = self.dims.hidden
        b = batch.batch
        programs = static.cache.get("programs")
        if programs is None:
            programs = self.programs(batch.programs)
        # Every object reader sees the normalized table; raw `programs` keep
        # scoring naming candidates.
        cards = self.semantic_cards
        if programs is None:
            # Torch's generic identity embedding excludes padding from gradients.
            table = self.card_definitions.weight
            definitions = self._definition_table(
                mx.concatenate([mx.stop_gradient(table[:1]), table[1:]])
            )
        else:
            public_programs = self._public(programs)
            definitions = self._definition_table(public_programs)
        tokens = cards.token_embedding.weight if self.config.catalog_records() else None

        def defs(rows: np.ndarray) -> mx.array:
            return _take(definitions, mx.array(rows))

        # Objects: players, cards, permanents and optional v11 stack rows.
        card_programs = defs(batch.card_rows)
        agent_player = self._stream(
            "player", self.player_embedding(self._floating(batch.agent_player))
        )
        opponent_player = self._stream(
            "player", self.player_embedding(self._floating(batch.opponent_player))
        )
        if cards is not None:
            attenuation = 1.0 if self.config.stream_norm else 60.0
            known = (
                defs(batch.known_rows)
                * self._floating(batch.known_counts)[..., None]
                / attenuation
            ).sum(2)
            known = self._masked_norm(
                getattr(cards, "hand_norm", None), known, batch.known_counts
            )
            library_pooled = (
                defs(batch.library_rows)
                * self._floating(batch.library_weights)[..., None]
                / attenuation
            ).sum(2)
            known = known + self._masked_norm(
                getattr(cards, "library_norm", None),
                cards.library_projection(library_pooled),
                batch.library_weights,
            )
            pooled = mx.einsum(
                "bsrh,bsrc->bsch",
                defs(batch.deck_rows),
                self._floating(batch.deck_counts),
            ).reshape(b, 2, 4 * hidden)
            known = known + self._masked_norm(
                getattr(cards, "decklist_norm", None),
                cards.decklist_projection(pooled),
                batch.deck_counts,
            )
            agent_player = agent_player + known[:, 0, None]
            opponent_player = opponent_player + known[:, 1, None]
        agent_cards = (
            self._stream("card", self.card_embedding(self._floating(batch.agent_cards)))
            + card_programs[:, 0]
        )
        opponent_cards = (
            self._stream(
                "card", self.card_embedding(self._floating(batch.opponent_cards))
            )
            + card_programs[:, 1]
        )
        permanents = self._stream(
            "permanent", self.perm_embedding(self._floating(batch.permanents))
        ) + defs(batch.permanent_rows)
        parts = [
            agent_player,
            opponent_player,
            agent_cards,
            opponent_cards,
            permanents[:, 0],
            permanents[:, 1],
        ]
        assert batch.stack is not None and batch.stack_rows is not None
        parts.append(
            self._stream("stack", self.stack_embedding(self._floating(batch.stack)))
            + defs(batch.stack_rows)
        )
        if (belief := batch.belief) is not None:
            rows = (
                self.belief_card_embedding(mx.array(belief.cards))
                + self.belief_owner_embedding(mx.array(belief.owners))
                + self.belief_zone_embedding(mx.array(belief.zones))
                + self.belief_count_embedding(self._floating(belief.probabilities))
            )
            parts.append(rows * self._floating(belief.validity)[..., None])
            parts.append(
                self.belief_global_embedding(self._floating(belief.globals))[:, None]
            )
        objects = mx.concatenate(parts, axis=1)
        count = objects.shape[1]
        objects = self._links(batch, objects)
        objects = self._join_stack_sources(batch, objects)
        objects = self._facts(batch, objects, tokens, definitions)
        objects = self._effects(batch, objects, tokens, definitions)

        # Continuation relations: per-row encoding, shared and frame context.
        flat = objects.reshape(b * count, hidden)
        linked = [
            _take(flat, mx.array(index)).reshape(b, -1, hidden)
            * self._floating(mask)[..., None]
            for index, mask in zip(
                batch.relation_objects, batch.relation_object_masks, strict=True
            )
        ]
        relation_definitions = defs(batch.relation_rows)
        valid = self._floating(batch.relation_valid)
        encoded = (
            self.continuation.projection(
                mx.concatenate(
                    [
                        self._floating(batch.relation_features),
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
        relation_count = valid.sum(1, keepdims=True)
        shared = encoded.sum(1) / mx.maximum(relation_count, 1)
        frames = mx.array(batch.frame_indexes)
        pooled_frames = (
            mx.zeros((b * rows, hidden), objects.dtype)
            .at[frames]
            .add(encoded.reshape(b * rows, hidden))
        )
        frame_counts = (
            mx.zeros((b * rows,), objects.dtype).at[frames].add(valid.reshape(-1))
        )
        frame_context = pooled_frames / mx.maximum(frame_counts, 1)[:, None]
        present = relation_count > 0
        if self.config.recent_events:
            assert batch.events is not None
            shared = shared + self.recent_events(batch.events, objects, definitions)
            present = present | mx.array(batch.events.validity.any(-1, keepdims=True))
        if "stream_norms" in self:
            shared = self._stream("context", shared) * present.astype(shared.dtype)
        # With attention off, historical pooling also sees the padding embeddings.
        # Only real objects receive continuation context.
        validity = self._floating(batch.validity)
        objects = objects + shared[:, None] * validity[..., None]

        summaries = None
        if self.config.decision_memory != "none":
            assert batch.memory is not None and programs is not None
            summaries = self.decision_memory(
                batch.memory,
                cards.token_embedding.weight,
                public_programs,
                self.name_candidates.embedding.weight,
            )
        encoded = self._attend_objects(objects, validity, batch, summaries)
        values = self._value_from_objects(encoded, validity)
        # The neutral critic token has no object coordinates for focus or FiLM.
        encoded_objects = encoded[:, :count]
        # A horizon bootstrap can contain only non-acting viewers.
        if not batch.legal_batches.size:
            return mx.full(batch.legal.shape, -1e8, dtype=mx.float32), values

        # Legal actions: joined rows, focus objects, policy head.
        legal_batches = mx.array(batch.legal_batches)
        actions = self.action_embedding(self._floating(batch.action_features))
        actions = (
            actions
            + _take(frame_context, mx.array(batch.action_frames))
            * self._floating(batch.action_frame_mask)[:, None]
        )
        names = self.name_candidates.embedding.weight
        actions = (
            actions
            + _take(names, mx.array(batch.name_rows))
            * self._floating(batch.name_mask)[:, None]
        )
        prefix = mx.concatenate(
            [mx.zeros((1, hidden), names.dtype), mx.cumsum(names[1:], axis=0)]
        )
        totals = _take(prefix, mx.array(batch.group_last)) - _take(
            prefix, mx.array(batch.group_first)
        )
        actions = (
            actions
            + totals
            / self._floating(batch.group_counts)[:, None]
            * self._floating(batch.group_mask)[:, None]
        )
        if programs is not None:
            actions = actions + _take(programs, mx.array(batch.candidate_rows))
        actions = actions + _take(shared, legal_batches)
        post = encoded_objects.reshape(b * count, hidden)
        focus = batch.focus
        focused = _take(post, mx.array(focus.reshape(-1))).reshape(*focus.shape, hidden)
        focused = (focused * self._floating(batch.focus_mask)[..., None]).reshape(
            focus.shape[0], -1
        )
        if "focus_film" in self:
            actions = actions * (1 + self.focus_film(focused))
        informed = self.action_layer(mx.concatenate([actions, focused], axis=-1))
        if "policy_film" in self:
            valid_mask = validity[..., None]
            pooled_state = (encoded_objects * valid_mask).sum(1) / mx.maximum(
                valid_mask.sum(1), 1
            )
            informed = informed * (
                1 + _take(self.policy_film(pooled_state), legal_batches)
            )
        # Loss arithmetic and the invalid-logit sentinel stay float32.
        scores = self.policy_head(informed)[:, 0].astype(mx.float32)
        logits = mx.where(
            mx.array(batch.legal),
            _take(scores, mx.array(batch.logit_positions.reshape(-1))).reshape(
                batch.legal.shape
            ),
            -1e8,
        )

        return logits, values

    def _attend_objects(
        self,
        objects: mx.array,
        validity: mx.array,
        batch: Prepared[np.ndarray],
        summaries: mx.array | None = None,
    ) -> mx.array:
        """Return objects in original focus coordinates plus any trailing critic token."""
        if not self.config.attention_on:
            return objects
        count = objects.shape[1]
        sign = self._floating(np.where(batch.is_agent, 1, -1))[..., None]
        if batch.packed_attention is not None:
            return self._attend_groups(objects, sign, batch.packed_attention)
        attended, attended_validity = objects, validity
        columns = batch.attention_columns
        if columns is not None:
            selected = mx.array(columns)
            attended = mx.take(objects, selected, axis=1)
            attended_validity = mx.take(validity, selected, axis=1)
            sign = mx.take(sign, selected, axis=1)
        if "value_token" in self:
            token = mx.broadcast_to(
                self.value_token, (objects.shape[0], 1, objects.shape[-1])
            )
            attended = mx.concatenate((attended, token), axis=1)
            sign = mx.concatenate((sign, mx.zeros_like(sign[:, :1])), axis=1)
            attended_validity = mx.concatenate(
                (attended_validity, mx.ones_like(attended_validity[:, :1])), axis=1
            )
            if columns is not None:
                columns = np.append(columns, count)
                count += 1
        shape = attended.shape
        live = None
        index = None
        if batch.live_index is not None:
            compact = batch.compact_attention
            assert compact is not None
            index = mx.array(batch.live_index)
            mask = mx.array(compact.key_mask, dtype=objects.dtype)
            live = CompactAttention(
                mx.array(compact.batches), mx.array(compact.slots), mask
            )
            attended = attended.reshape(-1, shape[-1])[index]
            sign = sign.reshape(-1, 1)[index]
            keep = mx.array(1, dtype=objects.dtype)
        else:
            keep = attended_validity[..., None].astype(objects.dtype)
            mask = mx.where(attended_validity != 0, 0.0, -mx.inf).astype(objects.dtype)[
                :, None, None, :
            ]
        for layer in (self.attention, *self.extra_attention):
            attended = layer(attended, sign, mask, keep, live, summaries)
        encoded_objects = (
            self.final_norm(attended) * keep if "final_norm" in self else attended
        )
        if index is not None:
            restored = mx.zeros((shape[0] * shape[1], shape[-1]), objects.dtype)
            restored[index] = encoded_objects
            return restored.reshape(shape)
        if columns is not None:
            restored = mx.zeros((shape[0], count, shape[-1]), encoded_objects.dtype)
            restored[:, mx.array(columns)] = encoded_objects
            return restored
        return encoded_objects

    def _attend_groups(
        self, objects: mx.array, sign: mx.array, packed: PackedAttention[np.ndarray]
    ) -> mx.array:
        """Attend each group of rows sorted by live length at its own width, then
        restore [B,N,D]; missing objects read a constant zero vector."""
        hidden = objects.shape[-1]
        encoded = []
        for group in packed.groups:
            slots = mx.array(group)
            indices = mx.maximum(slots, 0)
            attended = _take(objects.reshape(-1, hidden), indices)
            owners = _take(sign.reshape(-1, 1), indices)
            keep = slots >= 0
            mask = mx.where(keep, 0.0, -mx.inf).astype(attended.dtype)[:, None, None, :]
            keep = keep[..., None].astype(attended.dtype)
            for layer in (self.attention, *self.extra_attention):
                attended = layer(attended, owners, mask, keep)
            if "final_norm" in self:
                attended = self.final_norm(attended) * keep
            encoded.append(attended.reshape(-1, hidden))
        encoded.append(mx.zeros((1, hidden), objects.dtype))
        return _take(mx.concatenate(encoded), mx.array(packed.restore))

    def _value_from_objects(self, encoded: mx.array, validity: mx.array) -> mx.array:
        aggregation = self.config.value_aggregation
        if aggregation == "value_token":
            encoded = encoded[:, -1]
        elif aggregation == "attention_readout":
            encoded = self.value_readout(encoded, validity)
        head = self.value_head.layers
        projected = head[1](head[0](encoded))
        if aggregation == "historical_mean":
            # Historical policies include every projected slot, even padding.
            projected = projected.mean(1)
        elif aggregation == "masked_mean":
            projected = (projected * validity[..., None]).sum(1) / validity.sum(
                1, keepdims=True
            )
        return head[5](head[4](head[3](projected)))

    def _links(self, batch: Prepared[np.ndarray], objects: mx.array) -> mx.array:
        """`ActorCritic._join_permanent_links`."""
        b, count, hidden = objects.shape
        start = 2 + 2 * batch.agent_cards.shape[1]
        perm_count = 2 * batch.permanents.shape[2]
        flat = objects.reshape(b * count, hidden)
        permanents = objects[:, start : start + perm_count]
        delta = mx.zeros_like(flat)
        local = mx.zeros_like(permanents)
        for targets, mask, from_target, to_target in zip(
            batch.link_targets,
            batch.link_masks,
            self.link_from_target,
            self.link_to_target,
            strict=True,
        ):
            linked = self._floating(mask)[..., None]
            index = mx.array(targets)
            delta = delta.at[index].add(
                (to_target(permanents) * linked).reshape(-1, hidden)
            )
            local = (
                local
                + from_target(_take(flat, index).reshape(permanents.shape)) * linked
            )
        delta = delta.reshape(b, count, hidden)
        return (objects + delta).at[:, start : start + perm_count].add(local)

    def _join_stack_sources(
        self, batch: Prepared[np.ndarray], objects: mx.array
    ) -> mx.array:
        """`ActorCritic._join_stack_sources`: add source embeddings to v11 stack rows."""
        assert batch.stack is not None
        assert batch.stack_targets is not None and batch.stack_mask is not None
        b, total, hidden = objects.shape
        start = 2 + 2 * batch.agent_cards.shape[1] + 2 * batch.permanents.shape[2]
        count = batch.stack.shape[1]
        sources = _take(
            objects.reshape(b * total, hidden), mx.array(batch.stack_targets)
        ).reshape(b, count, hidden)
        delta = self.stack_source(sources) * self._floating(batch.stack_mask)[..., None]
        return objects.at[:, start : start + count].add(delta)

    def _facts(
        self,
        batch: Prepared[np.ndarray],
        objects: mx.array,
        tokens: mx.array | None,
        definitions: mx.array,
    ) -> mx.array:
        """`FactEncoder.forward` over prepared record ids."""
        facts = self.facts
        routes = batch.facts
        b, count, hidden = objects.shape
        shape = routes.kinds.shape

        def rows(table: mx.array, index: np.ndarray) -> mx.array:
            return _take(table, mx.array(index.reshape(-1))).reshape(*shape, hidden)

        if tokens is None:
            ids = (
                facts.ident(mx.array(routes.id_rows))
                + facts.detail(mx.array(routes.detail_rows))
                + facts.source(mx.array(routes.source_rows))
            )
        else:
            ids = (
                rows(tokens, routes.id_rows) * self._floating(routes.id_mask)[..., None]
                + rows(tokens, routes.detail_rows)
                * self._floating(routes.detail_mask)[..., None]
                + rows(tokens, routes.token_rows)
                * self._floating(routes.token_mask)[..., None]
                + rows(definitions, routes.source_rows)
            )
        hidden_rows = (
            facts.kind(mx.array(routes.kinds))
            + ids
            + facts.numeric(self._floating(routes.columns))
        )
        programs = routes.programs
        if programs is not None:
            schedule = programs.schedule
            flat = hidden_rows.reshape(-1, hidden)
            program = _take(flat, mx.array(programs.rows)) + facts.program_numeric(
                _program_number(mx.array(programs.numbers)).astype(self.compute_dtype)[
                    :, None
                ]
            )
            states, _ = facts.program_sequence.run(
                program, mx.array(schedule.gather), schedule.sizes
            )
            program = program + _take(states, mx.array(schedule.unpack))
            flat = mx.where(
                mx.array(programs.mask)[:, None],
                _take(program, mx.array(programs.positions)),
                flat,
            )
            hidden_rows = flat.reshape(*shape, hidden)
        encoded = (
            facts.output(mx.tanh(hidden_rows))
            * self._floating(routes.linked)[..., None]
        )
        delta = (
            mx.zeros((b * count, hidden), objects.dtype)
            .at[mx.array(routes.targets)]
            .add(encoded.reshape(-1, hidden))
        )
        return objects + delta.reshape(b, count, hidden)

    def _effects(
        self,
        batch: Prepared[np.ndarray],
        objects: mx.array,
        tokens: mx.array | None,
        definitions: mx.array,
    ) -> mx.array:
        """`SharedEffectEncoder.forward` over prepared record ids."""
        routes = batch.effects
        if routes is None:
            return objects
        effects = self.shared_effects
        b, count, hidden = objects.shape
        flat = objects.reshape(b * count, hidden)
        controller = (
            _take(flat, mx.array(routes.controller))
            * self._floating(routes.controller_mask)[:, None]
        )
        source_object = (
            _take(flat, mx.array(routes.source_object))
            * self._floating(routes.source_object_mask)[:, None]
        )
        header = effects.header(
            mx.concatenate(
                [self._floating(routes.numeric), controller, source_object], axis=-1
            )
        )
        if tokens is None:
            modification = effects.modification(mx.array(routes.modification_rows))
            sources = effects.source(mx.array(routes.source_rows))
            detail = effects.detail(mx.array(routes.detail_rows))
        else:
            modification = _take(tokens, mx.array(routes.modification_rows))
            sources = _take(definitions, mx.array(routes.source_rows))
            detail = _take(tokens, mx.array(routes.detail_rows))
        detail = (
            detail
            + effects.detail_kind(mx.array(routes.detail_kinds))
            + effects.detail_number(
                _program_number(mx.array(routes.detail_numbers)).astype(
                    self.compute_dtype
                )[:, None]
            )
            * self._floating(routes.detail_is_number)[:, None]
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
            sequence, mx.array(schedule.gather), schedule.sizes
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


def frozen_state(states: mx.array, index: int) -> mx.array:
    """Gather an independent state so the cache releases the full fold storage."""
    return mx.stop_gradient(mx.take(states, mx.array(index), axis=0))


class PolicyInference:
    """An unchanged inference policy and its optional weight-only program cache."""

    def __init__(self, agent: MLXActorCritic, *, cache_catalog: bool = True) -> None:
        if agent.compute_dtype != mx.float32:
            raise RuntimeError("open inference scopes on float32 master weights")
        self.agent = agent
        self.cache_catalog = cache_catalog
        self._closed = False
        self._weights = parameter_ids(agent.parameters())
        self._device = agent.inference_device
        self.memory = CollectionMemory[mx.array](
            mx.concatenate, mx.stack, frozen_state, mx.async_eval
        )
        self._precision = agent.precision
        self._binding = agent._binding()
        agent.clear_cache()

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("inference scope is closed")
        current = parameter_ids(self.agent.parameters())
        cast_state = self.agent._cast_state
        same_weights = (
            current == self._weights
            if cast_state is None
            else cast_state == (self._weights, current)
        )
        if (
            self._binding != self.agent._binding()
            or self._precision != self.agent.precision
            or self._device != self.agent.inference_device
            or not same_weights
        ):
            self.memory.clear()
            raise RuntimeError("inference policy changed within its scope")

    def cache_programs(self) -> None:
        self._check()
        if self.cache_catalog and "programs" not in self.agent._static.cache:
            self.agent.cache_programs()

    def __enter__(self) -> "PolicyInference":
        self._check()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._closed = True
        self.memory.clear()
        self.agent.clear_cache()

    def prepare(self, obs: PolicyObservation) -> Prepared[np.ndarray]:
        self._check()
        if self.agent.config.decision_memory != "gru":
            return self.agent.prepare(obs, inference=True)
        roots = self.memory.roots(obs)
        batch = self.agent.prepare(obs, inference=True, memory_roots=roots)
        # Routes stay NumPy. Only the optional serving states, references and
        # result sink on the fold use the backend array type.
        bound = self.memory.bind(cast(Prepared[mx.array], batch), obs, roots)
        return replace(cast(Prepared[np.ndarray], bound), guard=self.check_forward)

    def check_forward(self, agent: object) -> None:
        if agent is not self.agent:
            raise RuntimeError("prepared memory belongs to another inference policy")
        self._check()

    def infer(
        self,
        obs: PolicyObservation,
        *,
        span: Callable[[str], AbstractContextManager[object]] | None = None,
    ) -> InferenceResult:
        self._check()
        measure = span or (lambda name: nullcontext())
        with measure("input_admission_and_transfer_host"):
            batch = self.prepare(obs)
        with (
            measure("forward_host"),
            mx.stream(self.agent.inference_device),
        ):
            self.cache_programs()
            logits, _ = self.agent(batch)
            mx.eval(logits)
        with measure("output_transfer_host"):
            return InferenceResult(to_numpy(logits))
