"""
Shared viewer-observation encoder, legal-action policy and scalar or WDL critic.

ActorCritic.forward preserves the serving policy/value interface. Training uses
forward_distribution when it needs loss/draw/win logits rather than their mean.
Both consume device Prepared rows produced by host preparation and admission;
the forward does no validation host reads.

Object attention runs over the fixed object tables. With `bucketed_shapes=True`
(`manabot.policy.buckets`), host preparation trims the variable axes to
power-of-two buckets; the forward returns logits at full capacity.
With `live_tokens=True` (`manabot.backends.torch.live_tokens`), table attention runs
over the batch's valid objects packed into one sequence; either setting
composes with the other and neither changes the model.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from manabot.backends.torch._continuation import ContinuationEncoder
from manabot.backends.torch._facts import FactEncoder, RecordVocabulary
from manabot.backends.torch._name_candidates import NameCandidateEncoder
from manabot.backends.torch._numerics import buffer_on
from manabot.backends.torch._recent_events import RecentEventEncoder
from manabot.backends.torch._semantic_cards import (
    SemanticCardEncoder,
)
from manabot.backends.torch._shared_effects import SharedEffectEncoder

# Local imports
from manabot.backends.torch.arrays import _array_fields, device_of, on_device, to_numpy
from manabot.backends.torch.live_tokens import LiveTokens
from manabot.backends.torch.rl import NumericalError
from manabot.env.observation import (
    PERMANENT_DEFINITION_INDEX,
    PERMANENT_LINK_INDEXES,
    TURN_STATE_DIM,
    ObservationBuffers,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig
from manabot.infra.log import getLogger
from manabot.model.architecture import ArchitectureReceipt, architecture_receipt
from manabot.policy import admission
from manabot.policy.buckets import row_needs, sizes_of, trim_observation
from manabot.policy.checkpoint import layout
from manabot.policy.prepare import BeliefRows, Prepared, Tables, prepare
from manabot.policy.protocol import InferenceResult
import managym

if TYPE_CHECKING:
    from manabot.belief.encoding import BeliefCheckpointBinding

# Validity of the visible object table, in its concatenation order.
OBJECT_VALIDITY_KEYS = (
    "agent_player_valid",
    "opponent_player_valid",
    "agent_cards_valid",
    "opponent_cards_valid",
    "agent_permanents_valid",
    "opponent_permanents_valid",
)


@dataclass
class EmbeddedObservation:
    """One encoding shared by object attention, action joins and the critic."""

    objects: torch.Tensor
    is_agent: torch.Tensor
    validity: torch.Tensor
    shared_context: torch.Tensor
    frame_context: torch.Tensor
    # Raw executable programs; object inputs separately add public name identity.
    programs: torch.Tensor | None


def sample_rows(
    logits: np.ndarray, masks: np.ndarray, seeds: list[int], deterministic: bool
) -> list[int]:
    """Preserve scalar Categorical draws independently of arrival/batch order."""
    values, valid = torch.from_numpy(logits), torch.from_numpy(masks) != 0
    if valid.shape != values.shape:
        raise ValueError("sampling requires matching CPU logits and masks")
    if not valid.any(1).all() or not torch.isfinite(values[valid]).all():
        raise ValueError("invalid legal logits or empty action support")
    actions: list[int] = []
    for row, seed in enumerate(seeds):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            logits_row = values[row : row + 1]
            action = (
                logits_row.argmax(-1)
                if deterministic
                else torch.distributions.Categorical(logits=logits_row).sample()
            )
            selected = int(action.item())
        if not valid[row, selected]:
            raise ValueError("policy selected a masked offer")
        actions.append(selected)
    return actions


def _catalog_order(lengths: np.ndarray) -> np.ndarray:
    """Keep Torch's packed-GRU tie order, sorting only admitted host lengths."""
    return torch.from_numpy(lengths).sort(descending=True).indices.numpy()


class ActorCritic(nn.Module):
    """
    Shared actor/critic network that:
      1. Encodes game objects with typed embeddings.
      2. Uses an attention mechanism to allow information exchange.
      3. Augments action representations with focus object information.
      4. Computes policy logits and a value estimate.

    Debug logging uses explicit category tags to help diagnose issues,
    such as why the model appears to always select the default action.
    """

    def __init__(
        self,
        observation_space: ObservationSpace,
        config: ModelConfig,
        *,
        bucketed_shapes: bool = False,
        live_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.belief_binding: BeliefCheckpointBinding | None = None
        self.observation_space = observation_space
        self.config = config
        # Scalar heads need the checkpoint's training-target declaration. WDL
        # means have architectural signed units.
        self.value_semantic: Literal["signed_outcome", "win_logit"] | None = (
            "signed_outcome" if config.value_kind == "categorical_wdl" else None
        )
        self.logger = getLogger(__name__)

        # Extract dimensions from the observation encoder.
        enc = observation_space.encoder
        if config.recent_events and (
            config.world is None or enc.config.policy_history_version != 1
        ):
            raise ValueError("recent events require world and policy_history_version=1")
        if not config.recent_events and enc.config.policy_history_version != 0:
            raise ValueError(
                "history observation contract requires recent_events ActorCritic"
            )
        player_dim = enc.player_dim
        card_dim = enc.card_dim
        # Input v10 appends each seat's turn-state row to its player row and
        # each card row's turn-state value to that card row.
        self.turn_state = enc.config.policy_input_version == 10
        if self.turn_state:
            player_dim += TURN_STATE_DIM
            card_dim += 1
        # Definition IDs and permanent links are categorical joins, never
        # numeric permanent features.
        self._link_columns = PERMANENT_LINK_INDEXES
        join_columns = (PERMANENT_DEFINITION_INDEX, *self._link_columns)
        self._numeric_permanent_columns = [
            column for column in range(enc.permanent_dim) if column not in join_columns
        ]
        perm_dim = len(self._numeric_permanent_columns)
        action_dim = enc.action_dim
        self.max_focus_objects = enc.max_focus_objects
        embed_dim = config.hidden_dim

        self.semantic_cards = None
        self.card_definitions: nn.Embedding | None = None
        if config.world is not None:
            self.semantic_cards = SemanticCardEncoder(
                config.world,
                embed_dim,
                decklists=True,
                records=config.catalog_records(),
            )
        else:
            # Generic worlds still distinguish card definitions. The content
            # manifest binds the vocabulary; this is identity, not rules transfer.
            self.card_definitions = nn.Embedding(
                managym.CARD_DEFINITION_COUNT + 1, embed_dim, padding_idx=0
            )

        # Set up typed object embeddings.
        self.player_embedding = ProjectionLayer(player_dim, embed_dim)
        self.card_embedding = ProjectionLayer(card_dim, embed_dim)
        self.perm_embedding = ProjectionLayer(perm_dim, embed_dim)
        # Embed policy features; validity only masks legal rows.
        self.action_embedding = ProjectionLayer(action_dim - 2, embed_dim)

        self.logger.info(f"Player embedding: ({player_dim} -> {embed_dim})")
        self.logger.info(f"Card embedding: ({card_dim} -> {embed_dim})")
        self.logger.info(f"Permanent embedding: ({perm_dim} -> {embed_dim})")
        self.logger.info(f"Action embedding: ({action_dim - 2} -> {embed_dim})")

        # Global game state processor via attention.
        if self.config.attention_on:
            num_heads = self.config.num_attention_heads
            self.attention = GameObjectAttention(
                embed_dim,
                num_heads=num_heads,
                feedforward_dim=config.attention_feedforward_dim,
            )
            self.logger.info(
                f"Attention: {embed_dim} -> {embed_dim} with {num_heads} heads"
            )
        else:
            self.logger.info("Attention is off")

        # Action processing: Concatenate action embedding with focus object embeddings.
        actions_with_focus_dim = (self.max_focus_objects + 1) * embed_dim
        self.action_layer = ProjectionLayer(actions_with_focus_dim, embed_dim)
        self.logger.info(f"Action layer: ({actions_with_focus_dim} -> {embed_dim})")

        # Policy and value heads.
        self.policy_head = nn.Sequential(
            layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, 1), gain=0.01),
        )
        value_dim = 3 if config.value_kind == "categorical_wdl" else 1
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.ReLU(),
            MeanPoolingLayer(dim=1),
            layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, value_dim)),
        )
        self.logger.info(f"Policy head: ({embed_dim} -> 1)")
        self.logger.info(f"Value head: ({embed_dim} -> {value_dim})")

        self.belief_count_buckets = int(config.belief_count_buckets)
        if self.belief_count_buckets > 0:
            if config.belief_card_vocab_size < 1:
                raise ValueError(
                    "belief_card_vocab_size must be positive when belief input is enabled"
                )
            self.belief_card_embedding = nn.Embedding(
                config.belief_card_vocab_size, embed_dim
            )
            self.belief_owner_embedding = nn.Embedding(
                config.belief_owner_role_vocab_size, embed_dim
            )
            self.belief_zone_embedding = nn.Embedding(
                config.belief_hidden_zone_vocab_size, embed_dim
            )
            self.belief_count_embedding = ProjectionLayer(
                self.belief_count_buckets, embed_dim
            )
            self.belief_global_embedding = ProjectionLayer(2, embed_dim)
            self.logger.info(
                "Belief channel: on (%d count buckets, %d card definitions)",
                self.belief_count_buckets,
                config.belief_card_vocab_size,
            )
        else:
            self.belief_card_embedding = None
            self.belief_owner_embedding = None
            self.belief_zone_embedding = None
            self.belief_count_embedding = None
            self.belief_global_embedding = None

        # Static ownership mask over the concatenated visible object sequence.
        is_agent_parts = [
            torch.ones(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            torch.ones(enc.cards_per_player, dtype=torch.bool),
            torch.zeros(enc.cards_per_player, dtype=torch.bool),
            torch.ones(enc.perms_per_player, dtype=torch.bool),
            torch.zeros(enc.perms_per_player, dtype=torch.bool),
        ]
        is_agent_template = torch.cat(is_agent_parts).unsqueeze(0)
        self.register_buffer("_is_agent_template", is_agent_template)

        # Construct optional parameters after the historical model to preserve
        # its initialization order and state-dict names.
        self.extra_attention = nn.ModuleList(
            GameObjectAttention(
                embed_dim,
                config.num_attention_heads,
                ownership=False,
                feedforward_dim=config.attention_feedforward_dim,
            )
            for _ in range(config.attention_layers - 1)
        )
        self.value_token = (
            nn.Parameter(torch.randn(1, 1, embed_dim) / embed_dim**0.5)
            if config.value_aggregation == "value_token"
            else None
        )
        self.recent_events = (
            RecentEventEncoder(embed_dim) if config.recent_events else None
        )
        # FiLM policy conditioning (Perez et al., 2018, scale only): the head
        # reads informed_action * (1 + W @ state) for the masked mean of the
        # post-attention objects. Without it, offer and state meet additively in
        # the ReLU heads; when no unit changes activation across a decision's
        # offers and states, the logit gap between offers cannot depend on the
        # state and equal logits are stationary ("prefer A at low life, B at
        # high life" stays at ln 2). The product has a gradient there from any
        # start. Zero weights keep the initial function additive, and skip_init
        # draws no RNG, so earlier parameters keep their historical values.
        self.policy_film: nn.Linear | None = None
        if config.policy_film:
            self.policy_film = torch.nn.utils.skip_init(nn.Linear, embed_dim, embed_dim)
            nn.init.zeros_(self.policy_film.weight)
            nn.init.zeros_(self.policy_film.bias)

        self.continuation = ContinuationEncoder(embed_dim, enc.relation_dim)

        # The policy input links permanents to other objects: an Aura or
        # Equipment to its host, a blocker to the attacker it blocks. Each
        # link exchanges pre-attention embeddings both ways: the linked
        # permanent reads its target and the target reads (summed) linked
        # permanents, so an Equipment's effect, an Honest Work host's set
        # characteristics and a blocking assignment reach both objects. Zero
        # weights start additive; skip_init draws no RNG.
        self.link_from_target = nn.ModuleList()
        self.link_to_target = nn.ModuleList()
        for _ in self._link_columns:
            for layers in (self.link_from_target, self.link_to_target):
                layer = torch.nn.utils.skip_init(nn.Linear, embed_dim, embed_dim)
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                layers.append(layer)
        # Fact rows (counter kinds, added subtypes, set characteristics, spent
        # abilities, programs) and shared continuous-effect records join their
        # objects. Built after the links so earlier parameters keep their draws.
        # Catalog ids share the card encoder's token and definition rows.
        record_vocabulary: RecordVocabulary = (
            "catalog" if config.catalog_records() else "hashed"
        )
        self.facts = FactEncoder(
            embed_dim, enc.config.max_facts, vocabulary=record_vocabulary
        )
        self.shared_effects = SharedEffectEncoder(
            embed_dim, vocabulary=record_vocabulary
        )
        self.name_candidates = NameCandidateEncoder(
            managym.naming_catalog(),
            embed_dim,
            self.semantic_cards.definition_rows
            if self.semantic_cards is not None
            else None,
        )

        # An execution choice of the training stage, never model state: no
        # parameter, buffer or configuration field records it.
        self.bucketed_shapes = bucketed_shapes
        self.live_tokens = live_tokens
        self.facts.bucketed = self.bucketed_shapes
        self.name_candidates.bucketed = self.bucketed_shapes

        # Built last: every other parameter keeps the masked-mean arm's draws
        # under the same seed, so the two readouts start from one shared model.
        self.value_readout = (
            AttentionReadout(embed_dim, config.num_attention_heads)
            if config.value_aggregation == "attention_readout"
            else None
        )

        # When True, forward() stashes legal logits at original coordinates on
        # self.last_raw_logits for offline diagnosis. Off by default: the
        # detach/cpu copy is measurable in the inference hot path.
        self.debug = False
        self.last_raw_logits: Optional[torch.Tensor] = None
        self._spec_cache: tuple[object, admission.Spec] | None = None
        self._program_cache: torch.Tensor | None = None
        self._cache_versions: tuple[tuple[int, int], ...] = ()

    def _binding(self) -> str:
        return (
            self.config.model_dump_json()
            + "\n"
            + self.observation_space.encoder.config.model_dump_json()
        )

    def _admission_spec(self) -> admission.Spec:
        key = (
            self._binding(),
            tuple(
                (name, id(value), value._version)
                for name, value in self.named_buffers()
            ),
        )
        if self._spec_cache is not None and self._spec_cache[0] == key:
            return self._spec_cache[1]
        buffers: dict[str, np.ndarray] = {}
        for name, _ in self.named_buffers():
            parent, _, field = name.rpartition(".")
            module = self.get_submodule(parent) if parent else self
            buffers[name] = to_numpy(buffer_on(module, field, torch.device("cpu")))
        spec = admission.Spec(
            self.config,
            self.observation_space,
            replace(
                Tables.from_buffers(buffers, layout(self.observation_space)),
                order_lengths=_catalog_order,
            ),
            self.card_definitions.num_embeddings
            if self.card_definitions is not None
            else 0,
        )
        self._spec_cache = (key, spec)
        return spec

    def prepare(
        self, obs: ObservationBuffers, *, inference: bool = False
    ) -> Prepared[np.ndarray]:
        """Validate host buffers and build the shared policy input exactly once."""
        self.check_observation(obs)
        return self.prepare_admitted(obs, inference=inference)

    def check_observation(self, obs: ObservationBuffers) -> None:
        mask = np.asarray(obs["actions_valid"])
        try:
            legal = admission.legal_mask(mask)
        except ValueError as error:
            # Allocate framework evidence only for a rejected host input.
            failed = ~(np.isfinite(mask) & ((mask == 0) | (mask == 1)))
            raise NumericalError(
                "invalid_action_mask",
                {"mask": torch.from_numpy(mask), "failed": torch.from_numpy(failed)},
            ) from error
        admission.check(obs, self._admission_spec(), legal)

    def prepare_admitted(
        self,
        obs: ObservationBuffers,
        *,
        inference: bool = False,
        needs: np.ndarray | None = None,
    ) -> Prepared[np.ndarray]:
        """Prepare unchanged rows of an already admitted source."""
        spec = self._admission_spec()
        capacity = obs["actions_valid"].shape[1]
        if self.bucketed_shapes:
            if needs is None:
                needs = row_needs(obs, self.observation_space.encoder)
            if needs is not None:
                obs = trim_observation(
                    obs,
                    sizes_of(
                        needs,
                        self.observation_space.encoder,
                        objects=self.object_trimming(),
                    ),
                )
        batch = prepare(
            obs,
            spec.tables,
            inference=inference and device_of(self).type == "cpu",
            bucketed=self.bucketed_shapes,
            live_tokens=self.live_tokens and self.config.attention_on,
            value_token=self.value_token is not None,
            binding=spec.binding,
            action_capacity=capacity,
            select_programs=torch.is_grad_enabled()
            and not inference
            and not self.bucketed_shapes,
        )

        return replace(
            batch,
            bucket_needs=None
            if needs is None
            else tuple(tuple(map(int, row)) for row in needs),
        )

    @torch.no_grad()
    def evaluate(self, batch: Prepared[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Run an admitted host batch without gradients and return NumPy outputs."""
        self.eval()
        logits, values = self(on_device(batch, device_of(self)))
        return to_numpy(logits), to_numpy(values)

    @property
    def device_type(self) -> str:
        return device_of(self).type

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def resident_bytes(self) -> int:
        size = sum(
            value.numel() * value.element_size()
            for value in (*self.parameters(), *self.buffers())
        )
        if self.semantic_cards is not None:
            size += (
                self.semantic_cards.definition_rows.numel() * self.config.hidden_dim * 4
            )
        return size

    @property
    def has_programs(self) -> bool:
        return self.semantic_cards is not None

    def inference_scope(self, *, cache_catalog: bool = True) -> "PolicyInference":
        return PolicyInference(self, cache_catalog=cache_catalog)

    sample_rows = staticmethod(sample_rows)

    @staticmethod
    def normalize_logits(logits: np.ndarray, *, double: bool = False) -> np.ndarray:
        tensor = torch.from_numpy(logits)
        if double:
            tensor = tensor.double()
        return to_numpy(torch.softmax(tensor, dim=-1))

    @staticmethod
    def probability_from_logit(value: float) -> float:
        return float(torch.sigmoid(torch.tensor(value, dtype=torch.float64)))

    @torch.no_grad()
    def sample(
        self,
        batch: Prepared[np.ndarray],
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        sampling: Literal["categorical", "multinomial"] = "categorical",
    ) -> np.ndarray:
        """Retain Torch RNG and categorical/multinomial rounding at the boundary."""
        prepared = on_device(batch, device_of(self))
        if sampling == "categorical" and temperature == 1.0:
            actions, _, _, _ = self.get_action_and_value(
                prepared, deterministic=deterministic
            )
        else:
            logits, _ = self(prepared)
            if deterministic:
                actions = logits.argmax(-1)
            else:
                if temperature != 1.0:
                    logits = logits / temperature
                actions = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(
                    -1
                )
        return to_numpy(actions).astype(np.int64)

    @torch.no_grad()
    def win_probabilities(self, batch: Prepared[np.ndarray]) -> np.ndarray:
        _, values = self(on_device(batch, device_of(self)))
        return to_numpy(torch.sigmoid(values).float())

    def _parameter_versions(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (id(value), value._version)
            for value in (*self.parameters(), *self.buffers())
        )

    @torch.no_grad()
    def cache_programs(self) -> None:
        """Cache the full catalog for an unchanged inference policy."""
        self._program_cache = (
            self.semantic_cards.definitions(
                on_device(self._admission_spec().tables.full_programs, device_of(self))
            )
            if self.semantic_cards is not None
            else None
        )
        self._cache_versions = self._parameter_versions()

    def clear_cache(self) -> None:
        self._program_cache = None
        self._cache_versions = ()

    def forward(self, obs: Prepared[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits [B,A] and acting-viewer critic [B].

        WDL expectations are signed; scalar units follow value_semantic.
        """
        logits, raw_value = self.forward_distribution(obs)
        if self.config.value_kind == "categorical_wdl":
            probabilities = raw_value.softmax(dim=-1)
            value = probabilities[:, 2] - probabilities[:, 0]
        else:
            value = raw_value.squeeze(-1)
        return logits, value

    def forward_distribution(
        self, obs: Prepared[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and raw critic output from one shared encoding.

        Categorical output [B,3] contains loss/draw/win logits from the acting
        viewer's perspective; scalar output [B,1] is an unconstrained value.
        Neither output is detached: categorical supervision can train the
        critic and shared encoder directly without passing through its mean.
        A distributional head alone does not select a categorical loss.
        """
        programs = None
        if not isinstance(obs, Prepared) or not isinstance(obs.legal, torch.Tensor):
            raise TypeError(
                "policy forward requires device Prepared rows from prepare/on_device"
            )
        if obs.guard is not None:
            obs.guard(self)
        if obs.binding != self._binding():
            raise ValueError(
                "observation was admitted for a different policy configuration"
            )
        if (
            programs is None
            and not torch.is_grad_enabled()
            and self._program_cache is not None
        ):
            if self._cache_versions != self._parameter_versions():
                raise RuntimeError(
                    "cached inference policy changed; clear_cache before reuse"
                )
            programs = self._program_cache
        capacity = obs.legal.shape
        state = self._gather_object_embeddings(obs, programs)
        batches, slots = obs.legal_batches, obs.legal_slots
        live = None
        if obs.live_index is not None and obs.live_offsets is not None:
            live = LiveTokens(
                obs.live_index,
                obs.live_offsets,
                obs.batch,
                state.objects.shape[1] + int(self.value_token is not None),
            )
        encoded = self._attend_objects(
            state.objects, state.is_agent, state.validity, obs.attention_columns, live
        )
        # The optional trailing token is a critic input, never an action focus.
        post_attention_objects = encoded[:, : state.validity.shape[1]]
        informed_actions = self._gather_informed_actions(
            obs, post_attention_objects, state, batches, slots
        )
        if self.policy_film is not None:
            valid = state.validity.unsqueeze(-1)
            pooled = (post_attention_objects * valid).sum(1) / valid.sum(1).clamp_min(1)
            informed_actions = informed_actions * (
                1 + self.policy_film(pooled)[batches]
            )
        scores = self.policy_head(informed_actions).squeeze(-1)
        # Public logits retain original coordinates, including interior padding.
        # Only the legal rows entered the action networks; objects/programs were
        # encoded once for the batch, independent of each row's support size.
        logits = scores.new_full(capacity, -1e8)
        logits = logits.index_put((batches, slots), scores.to(logits.dtype))
        if self.debug:
            self.last_raw_logits = logits.detach().cpu()
        return logits, self._value_from_objects(encoded, state.validity)

    def architecture_receipt(self) -> ArchitectureReceipt:
        """This model's receipt; `named_parameters` yields each shared parameter once."""
        return architecture_receipt(
            self.config,
            self.observation_space,
            {
                name: (parameter.numel(), parameter.requires_grad)
                for name, parameter in self.named_parameters()
            },
        )

    def object_trimming(self) -> bool:
        """Whether buckets may trim object axes: padding objects must be inert."""
        return (
            self.config.value_aggregation != "historical_mean"
            and self.belief_count_buckets == 0
        )

    def _attend_objects(
        self,
        objects: torch.Tensor,
        is_agent: torch.Tensor,
        validity: torch.Tensor,
        columns: torch.Tensor | None = None,
        live: LiveTokens | None = None,
    ) -> torch.Tensor:
        """Attend using host-prepared columns or live-token coordinates."""
        if self.live_tokens and self.config.attention_on:
            assert live is not None
            return self._attend_live(objects, is_agent, live)
        original_slots = objects.shape[1]
        if not self.config.attention_on:
            columns = None
        if columns is not None:
            objects = objects.index_select(1, columns)
            is_agent = is_agent.index_select(1, columns)
            validity = validity.index_select(1, columns)
        key_padding_mask = validity == 0
        if self.value_token is not None:
            objects = torch.cat(
                (objects, self.value_token.expand(objects.shape[0], -1, -1)), dim=1
            )
            key_padding_mask = torch.cat(
                (key_padding_mask, torch.zeros_like(key_padding_mask[:, :1])), dim=1
            )
        if self.config.attention_on:
            objects = self.attention(
                objects, is_agent, key_padding_mask=key_padding_mask
            )
            for layer in self.extra_attention:
                objects = layer(objects, is_agent, key_padding_mask)
        if columns is not None:
            # Restore the optional neutral value token in the same scatter,
            # at the slot immediately after the original object coordinates.
            if self.value_token is not None:
                columns = torch.cat((columns, columns.new_tensor([original_slots])))
                original_slots += 1
            objects = objects.new_zeros(
                (objects.shape[0], original_slots, objects.shape[2])
            ).index_copy(1, columns, objects)
        return objects

    def _attend_live(
        self,
        objects: torch.Tensor,
        is_agent: torch.Tensor,
        tokens: LiveTokens,
    ) -> torch.Tensor:
        """`_attend_objects` over valid objects only, returned at [B,N(+1),D].

        The value token is one more live token per observation, ownerless as
        on the padded path. Padding slots come back zero, as every padded
        layer leaves them, so focus joins and every value aggregation read the
        same values.
        """
        batch = objects.shape[0]
        owner = torch.where(is_agent, 1.0, -1.0).to(objects.dtype)
        if self.value_token is not None:
            objects = torch.cat(
                (objects, self.value_token.expand(batch, -1, -1).to(objects.dtype)),
                dim=1,
            )
            owner = torch.cat((owner, owner.new_zeros((batch, 1))), dim=1)
        packed = tokens.gather(objects)
        owners = tokens.gather(owner)
        packed = self.attention.forward_live(packed, owners, tokens)
        for layer in self.extra_attention:
            packed = layer.forward_live(packed, owners, tokens)
        return tokens.scatter(packed)

    def _value_from_objects(
        self, objects: torch.Tensor, validity: torch.Tensor
    ) -> torch.Tensor:
        """Summarize final objects [B,N(+1),D] and apply the critic MLP.

        historical_mean averages every slot, padding included. masked_mean
        averages valid objects after the per-object projection, so N similar
        permanents carry N shares of the summary. value_token reads the
        trailing token that attended alongside the objects (and so also changed
        their actor-facing outputs). attention_readout pools the final valid
        objects with one learned query after attention; the actor never reads
        it, though critic gradients still reach the shared encoder. All
        variants share value_head's parameter names and MLP.
        """
        if self.config.value_aggregation == "historical_mean":
            return self.value_head(objects)
        if self.value_token is not None:
            objects = objects[:, -1]
        elif self.value_readout is not None:
            objects = self.value_readout(objects, validity == 0)
        projected = self.value_head[1](self.value_head[0](objects))
        if self.config.value_aggregation == "masked_mean":
            projected = projected.masked_fill((validity == 0).unsqueeze(-1), 0)
            projected = projected.sum(1) / validity.sum(1, keepdim=True)
        # Indexing avoids constructing temporary Sequential modules per decision.
        return self.value_head[5](self.value_head[4](self.value_head[3](projected)))

    def _gather_informed_actions(
        self,
        obs: Prepared[torch.Tensor],
        post_attention_objects: torch.Tensor,
        state: EmbeddedObservation,
        batches: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """Embed selected legal rows [K,D], preserving their original joins.

        Coordinates select actions, continuation frames and focus objects from
        the same observation row.
        """
        actions = self.action_embedding(obs.action_features)
        linked = state.frame_context.flatten(0, 1)[obs.action_frames]
        actions = actions + linked.masked_fill(
            (obs.action_frame_mask == 0).unsqueeze(-1), 0
        )
        actions = actions + self.name_candidates(obs)
        if self.semantic_cards is not None and state.programs is not None:
            actions = actions + state.programs[obs.candidate_rows]
        actions = actions + state.shared_context[batches]
        focused = post_attention_objects.flatten(0, 1)[obs.focus]
        focused = focused * obs.focus_mask.unsqueeze(-1)
        return self.action_layer(torch.cat((actions, focused.flatten(1)), dim=-1))

    def _gather_object_embeddings(
        self, obs: Prepared[torch.Tensor], programs: torch.Tensor | None = None
    ) -> EmbeddedObservation:
        enc_agent_player = self.player_embedding(obs.agent_player)
        enc_opp_player = self.player_embedding(obs.opponent_player)
        enc_agent_cards = self.card_embedding(obs.agent_cards)
        enc_opp_cards = self.card_embedding(obs.opponent_cards)
        definitions = None
        if self.semantic_cards is not None:
            if programs is None:
                assert obs.programs is not None
                programs = self.semantic_cards.definitions(obs.programs)
            definitions = self.name_candidates.public_definitions(programs)
            cards, known = self.semantic_cards(obs, definitions)
            enc_agent_cards = enc_agent_cards + cards[:, 0]
            enc_opp_cards = enc_opp_cards + cards[:, 1]
            enc_agent_player = enc_agent_player + known[:, 0, None]
            enc_opp_player = enc_opp_player + known[:, 1, None]
            permanent_definitions = definitions[obs.permanent_rows]
        else:
            assert self.card_definitions is not None
            cards = self.card_definitions(obs.card_rows)
            enc_agent_cards = enc_agent_cards + cards[:, 0]
            enc_opp_cards = enc_opp_cards + cards[:, 1]
            permanent_definitions = self.card_definitions(obs.permanent_rows)
        # Keep the per-seat contiguous projection layout after biases train.
        enc_agent_perms = (
            self.perm_embedding(obs.permanents[:, 0].contiguous())
            + permanent_definitions[:, 0]
        )
        enc_opp_perms = (
            self.perm_embedding(obs.permanents[:, 1].contiguous())
            + permanent_definitions[:, 1]
        )
        parts = [
            enc_agent_player,
            enc_opp_player,
            enc_agent_cards,
            enc_opp_cards,
            enc_agent_perms,
            enc_opp_perms,
        ]
        if obs.belief is not None:
            rows, global_row = self._belief_rows(obs.belief)
            parts.extend((rows, global_row))
        objects = self._join_permanent_links(obs, torch.cat(parts, dim=1))
        token_embedding = (
            self.semantic_cards.token_embedding
            if self.semantic_cards is not None and self.semantic_cards.records
            else None
        )
        objects = self.facts(obs.facts, objects, token_embedding, definitions)
        objects = self.shared_effects(
            obs.effects, objects, token_embedding, definitions
        )
        relation_definitions = (
            definitions[obs.relation_rows]
            if definitions is not None
            else self.card_definitions(obs.relation_rows)
        )
        encoded_relations = self.continuation.encode_rows(
            obs, objects, relation_definitions
        )
        shared_context = encoded_relations.sum(1) / obs.relation_valid.sum(
            1, keepdim=True
        ).clamp_min(1)
        frame_context = self.continuation.frame_context(obs, encoded_relations)
        if self.recent_events is not None:
            assert definitions is not None and obs.events is not None
            shared_context = (
                self.recent_events(obs.events, objects, definitions) + shared_context
            )
        objects = objects + shared_context.unsqueeze(1)
        return EmbeddedObservation(
            objects, obs.is_agent, obs.validity, shared_context, frame_context, programs
        )

    def _join_permanent_links(
        self, obs: Prepared[torch.Tensor], objects: torch.Tensor
    ) -> torch.Tensor:
        """Exchange embeddings across the host-prepared permanent links."""
        start = 2 + 2 * obs.agent_cards.shape[1]
        count = 2 * obs.permanents.shape[2]
        delta = torch.zeros_like(objects)
        for targets, mask, from_target, to_target in zip(
            obs.link_targets,
            obs.link_masks,
            self.link_from_target,
            self.link_to_target,
            strict=True,
        ):
            permanents = objects[:, start : start + count]
            linked = mask.bool().unsqueeze(-1)
            # Preserve per-batch scatter order; host routes are flat coordinates.
            local = (
                targets.view(objects.shape[0], count)
                - torch.arange(objects.shape[0], device=objects.device)[:, None]
                * objects.shape[1]
            )
            indexes = local.unsqueeze(-1).expand_as(permanents)
            delta = delta.scatter_add(
                1,
                indexes,
                to_target(permanents).masked_fill(~linked, 0).to(delta.dtype),
            )
            delta[:, start : start + count] += from_target(
                objects.flatten(0, 1)[targets].view_as(permanents)
            ).masked_fill(~linked, 0)
        return objects + delta

    def _belief_rows(
        self, obs: BeliefRows[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build explicit semantic belief rows and one global diagnostic row."""
        assert (
            self.belief_card_embedding is not None
            and self.belief_owner_embedding is not None
            and self.belief_zone_embedding is not None
            and self.belief_count_embedding is not None
            and self.belief_global_embedding is not None
        )
        rows = (
            self.belief_card_embedding(obs.cards)
            + self.belief_owner_embedding(obs.owners)
            + self.belief_zone_embedding(obs.zones)
            + self.belief_count_embedding(obs.probabilities)
        )
        rows = rows * obs.validity.unsqueeze(-1)
        global_row = self.belief_global_embedding(obs.globals).unsqueeze(1)
        return rows, global_row

    def get_action_and_value(
        self,
        obs: Prepared[torch.Tensor],
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        if not obs.legal_support:
            raise ValueError("No valid actions available")
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs: Prepared[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            _, value = self.forward(obs)
            return value


class PolicyInference:
    """One unchanged behavior policy; batches expire with the scope.

    Preparation consumes owned NumPy buffers. A guard on the resulting Prepared
    batch rejects closed scopes, weight/configuration changes and autograd use.
    The policy owns its lazy catalog cache; neither cache nor guard is serialized.
    """

    def __init__(self, agent: ActorCritic, *, cache_catalog: bool = True) -> None:
        self.agent = agent
        self.cache_catalog = cache_catalog
        self.device = device_of(agent)
        self._closed = False
        self._versions = agent._parameter_versions()
        self._modes = tuple(module.training for module in agent.modules())
        self._binding = agent._binding()
        self.transfers = 0
        self.transfer_bytes = 0
        agent.clear_cache()

    def check(self, agent: object) -> None:
        if self._closed:
            raise RuntimeError("inference scope is closed")
        if (
            agent is not self.agent
            or self._versions != self.agent._parameter_versions()
        ):
            raise RuntimeError("inference policy changed within its scope")
        if (
            self._modes != tuple(module.training for module in self.agent.modules())
            or self._binding != self.agent._binding()
        ):
            raise RuntimeError("inference policy mode/configuration changed")

    def _forward_check(self, agent: object) -> None:
        self.check(agent)
        if torch.is_grad_enabled():
            raise RuntimeError("prepared inference observations cannot enter autograd")
        self.cache_programs()

    def cache_programs(self) -> None:
        """Prepare the fixed policy catalog, checking the serving lifetime first."""
        self.check(self.agent)
        if self.cache_catalog and not self.agent._cache_versions:
            self.agent.cache_programs()

    def __enter__(self) -> "PolicyInference":
        self.check(self.agent)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._closed = True
        self.agent.clear_cache()

    @torch.no_grad()
    def infer(
        self,
        obs: ObservationBuffers,
        *,
        span: Callable[[str], AbstractContextManager[object]] | None = None,
    ) -> InferenceResult:
        measure = span or (lambda name: nullcontext())
        with measure("input_admission_and_transfer_host"):
            prepared = self.prepare(obs)
        with measure("forward_host"):
            logits, values = self.agent(prepared)
        with measure("output_transfer_host"):
            return InferenceResult(
                to_numpy(logits), self.transfer_bytes, self.transfers
            )

    @torch.no_grad()
    def prepare(self, obs: ObservationBuffers) -> Prepared[torch.Tensor]:
        self.check(self.agent)
        if any(not isinstance(value, np.ndarray) for value in obs.values()):
            raise TypeError("inference preparation requires NumPy observation buffers")
        batch = self.agent.prepare(obs, inference=True)
        if self.cache_catalog:
            batch = replace(batch, programs=None)
        self.transfer_bytes = (
            0
            if self.device.type == "cpu"
            else sum(
                array.nbytes for array in _array_fields(batch, np.ndarray).values()
            )
        )
        self.transfers = (
            0
            if self.device.type == "cpu"
            else len(
                {array.dtype for array in _array_fields(batch, np.ndarray).values()}
            )
        )
        return replace(on_device(batch, self.device), guard=self._forward_check)


# -----------------------------------------------------------------------------
# Basic Building Blocks
# -----------------------------------------------------------------------------
class MeanPoolingLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class AttentionReadout(nn.Module):
    """One learned query cross-attending over final objects [B,N,D] -> [B,D].

    A Set Transformer pooling-by-attention seed (Lee et al., 2019, PMA with one
    seed and no feedforward block). Each copy of an object receives its own
    softmax weight, so, as in a mean, copies share in proportion to their count;
    unlike a mean, the learned query can weight informative objects above the
    rest. Duplicating every object leaves the output unchanged. Cost is one O(N)
    query per observation, outside the object-attention path.
    """

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embedding_dim) / embedding_dim**0.5)
        self.mha = nn.MultiheadAttention(
            embedding_dim, num_heads=num_heads, batch_first=True
        )

    def forward(
        self, objects: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        query = self.query.expand(objects.shape[0], -1, -1)
        pooled, _ = self.mha(
            query,
            objects,
            objects,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return pooled.squeeze(1)


class ProjectionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            layer_init(nn.Linear(input_dim, output_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(output_dim, output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class GameObjectAttention(nn.Module):
    """
    Processes the complete game state by:
      - Adding a learned perspective vector (with sign flip based on controller),
      - Applying multi-head attention,
      - And returning the context-rich output.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        *,
        ownership: bool = True,
        feedforward_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.logger = getLogger(__name__).getChild("attention")
        self.logger.info(f"Creating perspective vector of size {embedding_dim}")
        self.perspective = (
            nn.Parameter(torch.randn(embedding_dim) / embedding_dim**0.5)
            if ownership
            else None
        )
        self.mha = nn.MultiheadAttention(
            embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        expansion = (
            feedforward_dim
            if feedforward_dim is not None
            else num_heads * embedding_dim
        )
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(embedding_dim, expansion)),
            nn.ReLU(),
            layer_init(nn.Linear(expansion, embedding_dim)),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        objects: torch.Tensor,
        is_agent: torch.Tensor,
        key_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        owned_objects = objects
        if self.perspective is not None:
            perspective_scale = torch.where(is_agent.unsqueeze(-1), 1.0, -1.0)
            # A trailing value token has no owner; real-object indexes stay fixed.
            if objects.shape[1] > is_agent.shape[1]:
                perspective_scale = torch.cat(
                    (perspective_scale, torch.zeros_like(perspective_scale[:, :1])),
                    dim=1,
                )
            owned_objects = objects + perspective_scale * self.perspective

        attn_out, _ = self.mha(
            owned_objects,
            owned_objects,
            owned_objects,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = self.norm1(owned_objects + attn_out)
        mlp_out = self.mlp(x)
        post_norm = self.norm2(x + mlp_out)

        mask = (~key_padding_mask).unsqueeze(-1).float()
        return post_norm * mask

    def forward_live(
        self, packed: torch.Tensor, owners: torch.Tensor, tokens: LiveTokens
    ) -> torch.Tensor:
        """`forward` over packed live tokens [L,D]; `owners` [L] is +1 for the
        acting viewer's objects, -1 for the opponent's and 0 for a value token."""
        if self.perspective is not None:
            packed = packed + owners.unsqueeze(-1) * self.perspective
        x = self.norm1(packed + tokens.self_attention(self.mha, packed))
        return self.norm2(x + self.mlp(x))


def layer_init(
    layer: nn.Module, gain: float = 1.0, bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
