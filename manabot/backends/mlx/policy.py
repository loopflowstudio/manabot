"""`ActorCritic` in MLX: the same encoder, attention stack, action heads and critic.

`MLXActorCritic` mirrors `manabot.backends.torch.policy.ActorCritic` module for
module, for the scalar catalog policies admitted by `require_supported`.
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

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from types import TracebackType
from typing import Literal

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
from manabot.backends.mlx.rl import NumericalError
from manabot.env.observation import ObservationBuffers, ObservationSpace
from manabot.infra.config import ModelConfig
from manabot.policy import admission
from manabot.policy.buckets import (
    BucketedObservation,
    bucket_observation,
    object_trimming,
)
from manabot.policy.checkpoint import Dimensions
from manabot.policy.prepare import (
    CatalogPrograms,
    Prepared,
    Tables,
    prepare,
)
from manabot.policy.protocol import InferenceResult, Precision
from manabot.policy.readings import Readings

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


def _gru_cell_backward(
    x: mx.array,
    previous: mx.array,
    recurrent_weight: mx.array,
    hidden_bias: mx.array,
    upstream: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Reverse one `_gru_cell` step from its inputs and dL/dh' `upstream` [L,H]
    -> (dL/dx [L,3H], dL/d(recurrent pre-activation) [L,3H], dL/dprevious [L,H]).

    The gates are recomputed rather than saved: the forward keeps only states.
    The recurrent gradient differs from dL/dx only in its new-gate third, which
    the reset gate scales.
    """
    hidden = previous.shape[1]
    recurrent = previous @ recurrent_weight.T + hidden_bias
    gates = mx.sigmoid(x[:, : 2 * hidden] + recurrent[:, : 2 * hidden])
    reset, update = gates[:, :hidden], gates[:, hidden:]
    recurrent_new = recurrent[:, 2 * hidden :]
    new = mx.tanh(x[:, 2 * hidden :] + reset * recurrent_new)
    new_pre = upstream * (1 - update) * (1 - new * new)
    reset_pre = new_pre * recurrent_new * reset * (1 - reset)
    update_pre = upstream * (previous - new) * update * (1 - update)
    dx = mx.concatenate([reset_pre, update_pre, new_pre], axis=1)
    drecurrent = mx.concatenate([reset_pre, update_pre, new_pre * reset], axis=1)
    dprevious = upstream * update + drecurrent @ recurrent_weight
    return dx, drecurrent, dprevious


# `mx.compile` fuses each step's elementwise gate arithmetic; it traces once per
# live-row count and reuses it across schedules.
_compiled_gru_cell = mx.compile(_gru_cell)
_compiled_gru_cell_backward = mx.compile(_gru_cell_backward)


def gru_recurrence(sizes: Sequence[int]) -> Callable[..., mx.array]:
    """The recurrence of a packed schedule with live counts `sizes`, as one
    differentiable function (projected [N,3H], Wh [3H,H], hidden_bias [3H]) ->
    every step's state [N,H] in packed order.

    Autograd never sees the per-step slices: MLX gives every slice output a
    whole-input zero cotangent, which makes a traced backward quadratic in the
    number of steps. The hand-written VJP runs the steps in reverse with a
    compiled cell, concatenates the input-projection gradient once and forms
    dWh and the hidden-bias gradient as one matmul and one sum over every
    packed row.
    """
    offsets = np.concatenate(([0], np.cumsum(sizes))).tolist()
    total = offsets[-1]
    # Packed row of each row's previous state; step 0 reads an appended zero row.
    previous_rows = mx.array(
        np.concatenate(
            [np.full(sizes[0], total)]
            + [
                offsets[step - 1] + np.arange(sizes[step])
                for step in range(1, len(sizes))
            ]
        ).astype(np.int32)
    )

    @mx.custom_function
    def recurrence(
        projected: mx.array, recurrent_weight: mx.array, hidden_bias: mx.array
    ) -> mx.array:
        state = mx.zeros((sizes[0], recurrent_weight.shape[1]), dtype=projected.dtype)
        outputs: list[mx.array] = []
        for step, live in enumerate(sizes):
            x = projected[offsets[step] : offsets[step] + live]
            state = _compiled_gru_cell(x, state[:live], recurrent_weight, hidden_bias)
            outputs.append(state)
        return mx.concatenate(outputs)

    @recurrence.vjp
    def recurrence_vjp(
        primals: tuple[mx.array, mx.array, mx.array],
        cotangent: mx.array,
        output: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        projected, recurrent_weight, hidden_bias = primals
        zero = mx.zeros((1, output.shape[1]), dtype=output.dtype)
        previous = mx.take(mx.concatenate([output, zero]), previous_rows, axis=0)
        dx: list[mx.array] = []
        drecurrent: list[mx.array] = []
        carried = None
        for step in reversed(range(len(sizes))):
            start, live = offsets[step], sizes[step]
            upstream = cotangent[start : start + live]
            if carried is not None:
                pad = live - carried.shape[0]
                upstream = upstream + (
                    mx.pad(carried, [(0, pad), (0, 0)]) if pad else carried
                )
            step_dx, step_drecurrent, carried = _compiled_gru_cell_backward(
                projected[start : start + live],
                previous[start : start + live],
                recurrent_weight,
                hidden_bias,
                upstream,
            )
            dx.append(step_dx)
            drecurrent.append(step_drecurrent)
        all_drecurrent = mx.concatenate(drecurrent[::-1])
        return (
            mx.concatenate(dx[::-1]),
            all_drecurrent.T @ previous,
            all_drecurrent.sum(axis=0),
        )

    return recurrence


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
        self, table: mx.array, rows: mx.array, sizes: Sequence[int]
    ) -> tuple[mx.array, mx.array]:
        """Packed row p reads row `rows[p]` of `table` [T,D]; step t holds the
        `sizes[t]` live sequences, in sorted order -> (every step's state [N,H]
        in packed order, each sequence's final state [S,H] in sorted order).

        Every token is gathered and projected once; `gru_recurrence` owns the
        steps and their gradient.
        """
        projected = mx.take(table, rows, axis=0) @ self.Wx.T + self.input_bias
        states = gru_recurrence(sizes)(projected, self.Wh, self.hidden_bias)
        return states, mx.take(states, mx.array(final_rows(sizes)), axis=0)


def final_rows(sizes: Sequence[int]) -> np.ndarray:
    """Packed row of each sorted sequence's last step: sequence s is live in the
    steps whose size exceeds s, a prefix because sizes never increase."""
    counts = np.asarray(sizes)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    sequences = np.arange(counts[0])
    lengths = np.searchsorted(-counts, -sequences, side="left")
    return (offsets[lengths - 1] + sequences).astype(np.int32)


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
        self, objects: mx.array, sign: mx.array, mask: mx.array, keep: mx.array
    ) -> mx.array:
        """`objects` [B,N,D]; `sign` [1,N,1] is +1 for the acting viewer's
        objects; `mask` [B,1,1,N] is additive (-inf on padding keys); `keep`
        [B,N,1] zeros padding outputs."""
        owned = objects + sign * self.perspective if "perspective" in self else objects
        attention_input = self.norm1(owned) if self.pre_norm else owned
        attended = attend(self.mha, attention_input, mask)
        residual = owned + attended
        if self.pre_norm:
            return (residual + self.mlp(self.norm2(residual))) * keep
        x = self.norm1(residual)
        return self.norm2(x + self.mlp(x)) * keep


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
    def __init__(
        self, dims: Dimensions, *, library: bool = False, stream_norm: bool = False
    ) -> None:
        super().__init__()
        self.token_embedding = padding_embedding(dims.token_rows, dims.hidden)
        self.encoder = PackedGRU(dims.hidden, dims.hidden)
        self.decklist_projection = nn.Linear(4 * dims.hidden, dims.hidden)
        if library:
            self.library_projection = nn.Linear(dims.hidden, dims.hidden, bias=False)
        if stream_norm:
            self.hand_norm = nn.LayerNorm(dims.hidden, eps=1e-5)
            if library:
                self.library_norm = nn.LayerNorm(dims.hidden, eps=1e-5)
            self.decklist_norm = nn.LayerNorm(dims.hidden, eps=1e-5)


class Facts(nn.Module):
    def __init__(self, dims: Dimensions) -> None:
        super().__init__()
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
    def __init__(self, dims: Dimensions) -> None:
        super().__init__()
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
    ) -> None:
        super().__init__()
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
        self.stack_tokens = space.one_token
        self.value_semantic: Literal["signed_outcome", "win_logit"] | None = None
        hidden = dims.hidden
        self.semantic_cards = SemanticCards(
            dims, library=space.one_token, stream_norm=config.stream_norm
        )
        self.player_embedding = Projection(dims.player, hidden)
        self.card_embedding = Projection(dims.card, hidden)
        self.perm_embedding = Projection(dims.permanent, hidden)
        self.action_embedding = Projection(dims.action, hidden)
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
            nn.Identity(),  # MeanPoolingLayer's slot; pooling is masked below
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
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
        if config.policy_film:
            self.policy_film = zero_projection(hidden)
        self.link_from_target = [nn.Linear(hidden, hidden) for _ in range(dims.links)]
        self.link_to_target = [nn.Linear(hidden, hidden) for _ in range(dims.links)]
        self.facts = Facts(dims)
        self.shared_effects = SharedEffects(dims)
        self.name_candidates = NameCandidates(dims)
        self.continuation = Continuation(dims)
        if space.one_token:
            self.stack_embedding = Projection(dims.stack, hidden)
            self.stack_source = zero_projection(hidden)
        if config.stream_norm:
            self.final_norm = nn.LayerNorm(hidden, eps=1e-5)
            streams = ["player", "card", "permanent", "definition", "context"]
            if space.one_token:
                streams.append("stack")
            self.stream_norms = {
                name: nn.LayerNorm(hidden, eps=1e-5) for name in streams
            }
        if config.focus_film:
            self.focus_film = nn.Linear(dims.focus * hidden, hidden, bias=False)
        named = np.nonzero(tables.name_definitions != 0)[0]
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
            + (len(tables.catalog_lengths) + 1) * self.dims.hidden * 4
        )

    @property
    def has_programs(self) -> bool:
        return True

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
            probabilities = mx.sigmoid(values[:, 0])
            mx.eval(probabilities)
        return to_numpy(probabilities)

    @property
    def precision(self) -> Precision:
        return self._precision

    @property
    def compute_dtype(self) -> mx.Dtype:
        """The dtype the weights hold now: float32 masters, or the selected
        precision's casts inside `forward_precision`."""
        return self.semantic_cards.token_embedding.weight.dtype

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
        """MLX does not implement decision memory."""
        return {}

    def _binding(self) -> str:
        return self._static.binding_cache.get(
            self.config, self.observation_space, self.readings
        )

    def prepare(
        self, obs: ObservationBuffers, *, inference: bool = False
    ) -> Prepared[np.ndarray]:
        """Admit and index raw host observations; malformed inputs are refused."""
        measured = self._admit_observation(obs)
        return self._prepare_rows(measured or obs, inference)

    def prepare_admitted(
        self,
        obs: ObservationBuffers,
        *,
        inference: bool = False,
        needs: np.ndarray | None = None,
    ) -> Prepared[np.ndarray]:
        """Index unchanged rows previously checked for this policy's configuration."""
        space = self._static.space
        bucketed = (
            bucket_observation(obs, space, needs, objects=self.object_trimming())
            if self.bucketed_shapes
            else None
        )
        return self._prepare_rows(bucketed or obs, inference)

    def _prepare_rows(
        self, obs: ObservationBuffers | BucketedObservation, inference: bool
    ) -> Prepared[np.ndarray]:
        return prepare(
            obs,
            self._static.tables,
            inference=inference,
            select_programs=not inference,
            compact_learner=self.bucketed_shapes,
            bucketed=self.bucketed_shapes,
            binding=self._binding(),
        )

    def check_observation(self, obs: ObservationBuffers) -> np.ndarray | None:
        """The shared admission checks on raw rows, before booleans or lookups."""
        measured = self._admit_observation(obs)
        return None if measured is None else measured.needs

    def _admit_observation(self, obs: ObservationBuffers) -> BucketedObservation | None:
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
        spec = admission.Spec(
            self.config, self.observation_space, self._static.tables, 0
        )
        return admission.admit(obs, spec, legal, bucketed=self.bucketed_shapes)

    # Programs ----------------------------------------------------------------------

    def programs(self, programs: CatalogPrograms[np.ndarray] | None = None) -> mx.array:
        """Encode selected definitions at their one-based catalog rows. Others stay zero."""
        with self.forward_precision():
            full = self._static.tables.full_programs
            if programs is None:
                programs = full
            weights = self.semantic_cards.token_embedding.weight
            table = mx.zeros(
                (len(full.rows) + 1, self.dims.hidden), dtype=weights.dtype
            )
            if not len(programs.rows):
                return table
            _, finished = self.semantic_cards.encoder.run(
                weights, mx.array(programs.tokens), programs.schedule.sizes
            )
            encoded = _take(finished, mx.array(programs.schedule.restore))
            return table.at[mx.array(programs.rows)].add(encoded)

    def cache_programs(self) -> None:
        """Encode the catalog once for inference with fixed weights (collection)."""
        programs = self.programs()
        mx.eval(programs)
        self._static.cache["programs"] = programs

    def clear_cache(self) -> None:
        self._static.cache.clear()

    def evaluate(self, batch: Prepared[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Evaluated logits [B,A] and values [B] as NumPy (`BackendPolicy.evaluate`)."""
        with mx.stream(self.inference_device):
            logits, values = self(batch)
            mx.eval(logits, values)
        return to_numpy(logits), to_numpy(values)[:, 0]

    def evaluate_probabilities(
        self, batch: Prepared[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        with mx.stream(self.inference_device):
            logits, values = self(batch)
            probabilities = mx.softmax(logits, axis=-1)
            mx.eval(probabilities, values)
        return to_numpy(probabilities), to_numpy(values)[:, 0]

    def forward_distribution(
        self, batch: Prepared[np.ndarray]
    ) -> tuple[mx.array, mx.array]:
        """The training forward the updates differentiate: `__call__`."""
        return self(batch)

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
        """Policy logits [B,A] (-1e8 off the legal support) and critic [B,1]."""
        with self.forward_precision():
            return self._forward(batch)

    def _attend_objects(
        self,
        batch: Prepared[np.ndarray],
        objects: mx.array,
        validity: mx.array,
        sign: mx.array,
    ) -> mx.array:
        count, hidden = objects.shape[1:]
        packed = batch.packed_attention
        if packed is not None:
            encoded = []
            for group in packed.groups:
                slots = mx.array(group)
                indices = mx.maximum(slots, 0)
                attended = _take(objects.reshape(-1, hidden), indices)
                owners = _take(sign.reshape(-1, 1), indices)
                values = self._attention_stack(attended, owners, slots >= 0)
                encoded.append(values.reshape(-1, hidden))
            # Missing objects all point to the final, constant zero vector.
            encoded.append(mx.zeros((1, hidden), objects.dtype))
            return _take(mx.concatenate(encoded), mx.array(packed.restore))

        attended, attended_validity = objects, validity
        columns = batch.attention_columns
        if columns is not None:
            selected = mx.array(columns)
            attended = mx.take(objects, selected, axis=1)
            attended_validity = mx.take(validity, selected, axis=1)
            sign = mx.take(sign, selected, axis=1)
        encoded_objects = self._attention_stack(attended, sign, attended_validity != 0)
        if columns is not None:
            # Restore the original coordinates; unused columns are zero.
            positions = np.zeros(count, dtype=np.int64)
            positions[columns] = np.arange(len(columns))
            used = np.zeros(count, dtype=np.float32)
            used[columns] = 1
            encoded_objects = (
                mx.take(encoded_objects, mx.array(positions), axis=1)
                * self._floating(used)[None, :, None]
            )
        return encoded_objects

    def _attention_stack(
        self, attended: mx.array, sign: mx.array, keep: mx.array
    ) -> mx.array:
        mask = mx.where(keep, 0.0, -mx.inf).astype(attended.dtype)[:, None, None, :]
        keep = keep[..., None].astype(attended.dtype)
        encoded_objects = self.attention(attended, sign, mask, keep)
        for layer in self.extra_attention:
            encoded_objects = layer(encoded_objects, sign, mask, keep)
        if "final_norm" in self:
            encoded_objects = self.final_norm(encoded_objects) * keep
        return encoded_objects

    def _forward(self, batch: Prepared[np.ndarray]) -> tuple[mx.array, mx.array]:
        if not isinstance(batch, Prepared):
            raise TypeError("policy forward requires a Prepared batch")
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
        definitions = self._definition_table(self._public(programs))
        cards = self.semantic_cards
        tokens = cards.token_embedding.weight

        def defs(rows: np.ndarray) -> mx.array:
            return _take(definitions, mx.array(rows))

        # Objects: players, cards, permanents and optional v11 stack rows.
        card_programs = defs(batch.card_rows)
        attenuation = 1.0 if self.config.stream_norm else 60.0
        known = (
            defs(batch.known_rows)
            * self._floating(batch.known_counts)[..., None]
            / attenuation
        ).sum(2)
        known = self._masked_norm(
            getattr(cards, "hand_norm", None), known, batch.known_counts
        )
        if "library_projection" in cards:
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
            "bsrh,bsrc->bsch", defs(batch.deck_rows), self._floating(batch.deck_counts)
        ).reshape(b, 2, 4 * hidden)
        known = known + self._masked_norm(
            getattr(cards, "decklist_norm", None),
            cards.decklist_projection(pooled),
            batch.deck_counts,
        )
        agent_player = (
            self._stream(
                "player", self.player_embedding(self._floating(batch.agent_player))
            )
            + known[:, 0, None]
        )
        opponent_player = (
            self._stream(
                "player", self.player_embedding(self._floating(batch.opponent_player))
            )
            + known[:, 1, None]
        )
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
        if self.stack_tokens:
            assert batch.stack is not None and batch.stack_rows is not None
            parts.append(
                self._stream("stack", self.stack_embedding(self._floating(batch.stack)))
                + defs(batch.stack_rows)
            )
        objects = mx.concatenate(parts, axis=1)
        count = objects.shape[1]
        objects = self._links(batch, objects)
        if self.stack_tokens:
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
        if "stream_norms" in self:
            shared = self._stream("context", shared) * (relation_count > 0).astype(
                shared.dtype
            )
        objects = objects + shared[:, None]

        # Attention over the object table.
        validity = self._floating(batch.validity)
        sign = self._floating(np.where(batch.is_agent, 1, -1))[..., None]
        encoded_objects = self._attend_objects(batch, objects, validity, sign)
        keep = (validity != 0)[..., None].astype(objects.dtype)

        # Masked-mean critic.
        value = self.value_head.layers
        projected = value[1](value[0](encoded_objects)) * keep
        pooled_value = projected.sum(1) / validity.sum(1, keepdims=True)
        values = value[5](value[4](value[3](pooled_value))).astype(mx.float32)
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
        modification = _take(tokens, mx.array(routes.modification_rows))
        sources = _take(definitions, mx.array(routes.source_rows))
        detail = (
            _take(tokens, mx.array(routes.detail_rows))
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


class PolicyInference:
    """An unchanged inference policy and its optional weight-only program cache."""

    def __init__(self, agent: MLXActorCritic, *, cache_catalog: bool = True) -> None:
        if agent.compute_dtype != mx.float32:
            raise RuntimeError("open inference scopes on float32 master weights")
        self.agent = agent
        self.cache_catalog = cache_catalog
        self._closed = False
        self._weights = parameter_ids(agent.parameters())
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
            or not same_weights
        ):
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
        self.agent.clear_cache()

    def prepare(self, obs: ObservationBuffers) -> Prepared[np.ndarray]:
        self._check()
        return self.agent.prepare(obs, inference=True)

    def infer(
        self,
        obs: ObservationBuffers,
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
