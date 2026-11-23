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

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from manabot.backends.torch._continuation import ContinuationEncoder
from manabot.backends.torch._decision_memory import CollectionMemory, DecisionMemory
from manabot.backends.torch._facts import FactEncoder, RecordVocabulary
from manabot.backends.torch._name_candidates import NameCandidateEncoder
from manabot.backends.torch._numerics import buffer_on
from manabot.backends.torch._recent_events import RecentEventEncoder
from manabot.backends.torch._semantic_cards import (
    SemanticCardEncoder,
)
from manabot.backends.torch._shared_effects import SharedEffectEncoder

# Local imports
from manabot.backends.torch.arrays import array_fields, device_of, on_device, to_numpy
from manabot.backends.torch.live_tokens import LiveTokens
from manabot.backends.torch.rl import NumericalError
from manabot.belief.range import BeliefError
from manabot.env.observation import (
    ObservationBuffers,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig
from manabot.infra.log import getLogger
from manabot.model.architecture import ArchitectureReceipt, architecture_receipt
from manabot.policy import admission
from manabot.policy.buckets import trim_to_needs
from manabot.policy.prepare import (
    BeliefRows,
    Layout,
    Prepared,
    Tables,
    object_ownership,
    prepare,
)
from manabot.policy.protocol import InferenceResult
from manabot.policy.readings import Readings, fresh_readings
import managym

if TYPE_CHECKING:
    from manabot.belief.encoding import BeliefEncodingSchema

if TYPE_CHECKING:
    from manabot.belief.encoding import BeliefCheckpointBinding


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
        readings: Readings | None = None,
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

        if (
            config.decision_memory != "none"
            and not observation_space.config.max_decision_records
        ):
            raise ValueError(
                "decision-memory model requires explicit observation capacities"
            )
        if config.decision_memory != "none" and not config.catalog_records():
            raise ValueError("decision memory requires a catalog world")
        if config.decision_memory != "none" and not observation_space.one_token:
            raise ValueError("decision memory requires policy input v11")
        if config.recent_events and (
            config.world is None or observation_space.config.policy_history_version != 1
        ):
            raise ValueError("recent events require world and policy_history_version=1")
        if (
            not config.recent_events
            and observation_space.config.policy_history_version != 0
        ):
            raise ValueError(
                "history observation contract requires recent_events ActorCritic"
            )
        # A model built without a checkpoint reads its inputs by kind.
        self.readings: Readings = (
            fresh_readings(observation_space) if readings is None else readings
        )
        columns = Layout(self.readings)
        player_dim = columns.numeric_width("agent_player")
        card_dim = columns.numeric_width("agent_cards")
        # Inputs v10/v11 append each seat's turn-state row to its player row and
        # each card row's turn-state value to that card row.
        self.turn_state = observation_space.turn_state
        self.stack_tokens = observation_space.one_token
        if self.turn_state:
            player_dim += columns.numeric_width("player_turn_state")
            card_dim += 1
        # Joins (definitions, links, action continuations) and validity
        # columns the mask carries are never numeric features.
        perm_dim = columns.numeric_width("agent_permanents")
        action_dim = columns.numeric_width("actions")
        self.max_focus_objects = observation_space.config.max_focus_objects
        embed_dim = config.hidden_dim

        self.semantic_cards = None
        self.card_definitions: nn.Embedding | None = None
        if config.world is not None:
            # Log-enabled controls reserve the branches' exact catalog rows,
            # so trunk copying never remaps a shared embedding or lookup.
            self.semantic_cards = SemanticCardEncoder(
                config.world,
                embed_dim,
                decklists=True,
                records=config.catalog_records(),
                decision_memory=bool(observation_space.config.max_decision_records),
                library=observation_space.one_token,
                stream_norm=config.stream_norm,
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
        self.action_embedding = ProjectionLayer(action_dim, embed_dim)

        self.logger.info(f"Player embedding: ({player_dim} -> {embed_dim})")
        self.logger.info(f"Card embedding: ({card_dim} -> {embed_dim})")
        self.logger.info(f"Permanent embedding: ({perm_dim} -> {embed_dim})")
        self.logger.info(f"Action embedding: ({action_dim} -> {embed_dim})")

        # Global game state processor via attention.
        if self.config.attention_on:
            num_heads = self.config.num_attention_heads
            self.attention = GameObjectAttention(
                embed_dim,
                num_heads=num_heads,
                feedforward_dim=config.attention_feedforward_dim,
                memory_gate_scale=config.decision_memory_gate_scale,
                pre_norm=config.stream_norm,
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
        ownership = object_ownership(
            observation_space.config.max_cards_per_player,
            observation_space.config.max_permanents_per_player,
        )
        self.register_buffer("_is_agent_template", torch.tensor(ownership[None]))

        # Construct optional parameters after the historical model to preserve
        # its initialization order and state-dict names.
        self.extra_attention = nn.ModuleList(
            GameObjectAttention(
                embed_dim,
                config.num_attention_heads,
                ownership=False,
                feedforward_dim=config.attention_feedforward_dim,
                memory_gate_scale=config.decision_memory_gate_scale,
                pre_norm=config.stream_norm,
            )
            for _ in range(config.attention_layers - 1)
        )
        self.value_token = (
            nn.Parameter(torch.randn(1, 1, embed_dim) / embed_dim**0.5)
            if config.value_aggregation == "value_token"
            else None
        )
        self.decision_memory = (
            DecisionMemory(
                config.decision_memory, config.decision_memory_summaries, embed_dim
            )
            if config.decision_memory != "none"
            else None
        )
        if self.decision_memory is not None:
            for layer in (self.attention, *self.extra_attention):
                layer.memory_gate = nn.Parameter(torch.zeros(()))
                layer.memory_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.memory_summaries = (
            config.decision_memory_summaries if self.decision_memory is not None else 0
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

        self.continuation = ContinuationEncoder(
            embed_dim, columns.numeric_width("relations")
        )

        # The policy input links permanents to other objects: an Aura or
        # Equipment to its host, a blocker to the attacker it blocks. Each
        # link exchanges pre-attention embeddings both ways: the linked
        # permanent reads its target and the target reads (summed) linked
        # permanents, so an Equipment's effect, an Honest Work host's set
        # characteristics and a blocking assignment reach both objects. Zero
        # weights start additive; skip_init draws no RNG.
        self.link_from_target = nn.ModuleList()
        self.link_to_target = nn.ModuleList()
        for _ in columns.link_columns:
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
            embed_dim,
            numeric=columns.fact_width(catalog_ids=record_vocabulary == "catalog"),
            vocabulary=record_vocabulary,
        )
        self.shared_effects = SharedEffectEncoder(
            embed_dim,
            numeric=columns.numeric_width("effects"),
            vocabulary=record_vocabulary,
        )
        self.name_candidates = NameCandidateEncoder(
            managym.naming_catalog(),
            embed_dim,
            self.semantic_cards.definition_rows
            if self.semantic_cards is not None
            else None,
            onto_definitions=config.definition_names,
        )

        # An execution choice of the training stage, never model state: no
        # parameter, buffer or configuration field records it.
        self.bucketed_shapes = bucketed_shapes
        self.live_tokens = live_tokens
        self.name_candidates.bucketed = self.bucketed_shapes

        # Input v11 stack rows, built after every v9/v10 parameter (except the
        # value readout, built last) so those checkpoints keep their
        # initialization and state-dict names. A row
        # projects its numeric columns and adds its definition's program, as a
        # permanent does; an ability's definition is its source's. Its source
        # link (one-based object index) adds the source's pre-attention
        # embedding through a zero-initialized layer, so the initial function
        # ignores it.
        self.stack_embedding: ProjectionLayer | None = None
        self.stack_source: nn.Linear | None = None
        if self.stack_tokens:
            stack_dim = columns.numeric_width("stack")
            self.stack_embedding = ProjectionLayer(stack_dim, embed_dim)
            self.stack_source = torch.nn.utils.skip_init(
                nn.Linear, embed_dim, embed_dim
            )
            nn.init.zeros_(self.stack_source.weight)
            nn.init.zeros_(self.stack_source.bias)
            self.logger.info(f"Stack embedding: ({stack_dim} -> {embed_dim})")

        # Built last: every other parameter keeps the masked-mean arm's draws
        # under the same seed, so the two readouts start from one shared model.
        self.value_readout = (
            AttentionReadout(embed_dim, config.num_attention_heads)
            if config.value_aggregation == "attention_readout"
            else None
        )

        # One normalized token stream (`ModelConfig.stream_norm`): every
        # stream summed into an object token is normalized here or starts at
        # zero (links, stack sources, facts, shared effects). Pre-normalized
        # attention leaves the residual stream unnormalized, so the heads read
        # it through `final_norm`. LayerNorms draw no RNG, so every parameter
        # above keeps its draws.
        self.stream_norms: nn.ModuleDict | None = None
        self.final_norm: nn.LayerNorm | None = None
        if config.stream_norm:
            self.final_norm = nn.LayerNorm(embed_dim)
            streams = ["player", "card", "permanent", "definition", "context"]
            if self.stack_tokens:
                streams.append("stack")
            self.stream_norms = nn.ModuleDict(
                {name: nn.LayerNorm(embed_dim) for name in streams}
            )

        # Focus-conditioned scoring (FiLM over the offer's focused objects):
        # the offer's features are scaled by 1 + W @ focused before
        # action_layer, so two offers naming one object differ by a function
        # of it at initialization. Ordinary initialization, unlike
        # policy_film: a zero start would leave that gap blind to the object
        # until W moved. Built last so every other parameter keeps the draws
        # of the model without it.
        self.focus_film: nn.Linear | None = None
        if config.focus_film:
            self.focus_film = layer_init(
                nn.Linear(self.max_focus_objects * embed_dim, embed_dim, bias=False)
            )

        # When True, forward() stashes legal logits at original coordinates on
        # self.last_raw_logits for offline diagnosis. Off by default: the
        # detach/cpu copy is measurable in the inference hot path.
        self.debug = False
        self.last_raw_logits: Optional[torch.Tensor] = None
        self._spec_cache: tuple[object, admission.Spec] | None = None
        self._program_cache: torch.Tensor | None = None
        self._cache_versions: tuple[tuple[int, int], ...] = ()

    def admit_belief(self, belief_schema: "BeliefEncodingSchema") -> None:
        """Admit a host belief vocabulary against this policy's input embeddings."""
        if self.belief_count_buckets != belief_schema.count_buckets:
            raise BeliefError("ActorCritic and belief encoding count buckets differ")
        if self.belief_card_embedding is None:
            raise BeliefError(
                "Manabot requires an ActorCritic with belief input enabled"
            )
        maximum_card_id = max(row.card_def_id for row in belief_schema.rows)
        if maximum_card_id >= self.belief_card_embedding.num_embeddings:
            raise BeliefError("belief schema exceeds the ActorCritic card vocabulary")
        if self.belief_owner_embedding is None:
            raise BeliefError("Manabot requires an ActorCritic owner-role vocabulary")
        maximum_owner_id = max(row.owner_role_id for row in belief_schema.rows)
        if maximum_owner_id >= self.belief_owner_embedding.num_embeddings:
            raise BeliefError(
                "belief schema exceeds the ActorCritic owner-role vocabulary"
            )
        if self.belief_zone_embedding is None:
            raise BeliefError("Manabot requires an ActorCritic hidden-zone vocabulary")
        maximum_zone_id = max(row.hidden_zone_id for row in belief_schema.rows)
        if maximum_zone_id >= self.belief_zone_embedding.num_embeddings:
            raise BeliefError(
                "belief schema exceeds the ActorCritic hidden-zone vocabulary"
            )

    def _binding(self) -> str:
        return admission.binding(self.config, self.observation_space, self.readings)

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
                Tables.from_buffers(buffers, Layout(self.readings)),
                order_lengths=_catalog_order,
            ),
            self.card_definitions.num_embeddings
            if self.card_definitions is not None
            else 0,
        )
        self._spec_cache = (key, spec)
        return spec

    def prepare(
        self,
        obs: ObservationBuffers,
        *,
        inference: bool = False,
        memory_roots: Mapping[str, int] | None = None,
    ) -> Prepared[np.ndarray]:
        """Validate host buffers and build the shared policy input exactly once."""
        self.check_observation(obs)
        return self.prepare_admitted(
            obs, inference=inference, memory_roots=memory_roots
        )

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
        memory_roots: Mapping[str, int] | None = None,
    ) -> Prepared[np.ndarray]:
        """Prepare unchanged admitted rows for learning or inference.

        `inference` selects compaction and catalog coverage independently of
        autograd mode; bucketed batches always use the full catalog schedule.
        """
        spec = self._admission_spec()
        capacity = obs["actions_valid"].shape[1]
        if self.bucketed_shapes:
            obs, needs = trim_to_needs(
                obs, self.observation_space, needs, objects=self.object_trimming()
            )
        return prepare(
            obs,
            spec.tables,
            inference=inference and device_of(self).type == "cpu",
            bucketed=self.bucketed_shapes,
            live_tokens=self.live_tokens and self.config.attention_on,
            value_token=self.value_token is not None,
            memory_summaries=self.memory_summaries,
            memory_fold=self.config.decision_memory == "gru",
            memory_roots=memory_roots,
            binding=spec.binding,
            action_capacity=capacity,
            bucket_needs=needs,
            select_programs=not inference and not self.bucketed_shapes,
        )

    def _evaluate_outputs(
        self, batch: Prepared[np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            return self(on_device(batch, device_of(self)))
        finally:
            for module, training in modes:
                module.training = training

    @torch.no_grad()
    def evaluate(self, batch: Prepared[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Run one admitted batch, preserving the caller's module modes."""
        logits, values = self._evaluate_outputs(batch)
        return to_numpy(logits), to_numpy(values)

    @torch.no_grad()
    def evaluate_probabilities(
        self, batch: Prepared[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize on the policy device before transferring outputs to the host."""
        logits, values = self._evaluate_outputs(batch)
        return to_numpy(torch.softmax(logits, dim=-1)), to_numpy(values)

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
        logits, _ = self(on_device(batch, device_of(self)))
        return self.sample_logits(
            logits,
            deterministic=deterministic,
            temperature=temperature,
            sampling=sampling,
        )

    @torch.no_grad()
    def sample_logits(
        self,
        logits: np.ndarray | torch.Tensor,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        sampling: Literal["categorical", "multinomial"] = "categorical",
    ) -> np.ndarray:
        """Sample once after bounded forwards, preserving backend RNG partition."""
        values = torch.as_tensor(logits, device=device_of(self))
        if deterministic:
            actions = values.argmax(-1)
        elif sampling == "categorical" and temperature == 1.0:
            actions = torch.distributions.Categorical(logits=values).sample()
        else:
            actions = torch.multinomial(
                torch.softmax(values / temperature, dim=-1), 1
            ).squeeze(-1)
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
        programs = None
        if not torch.is_grad_enabled() and self._program_cache is not None:
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
        summaries = None
        if self.decision_memory is not None:
            assert obs.memory is not None and self.semantic_cards is not None
            summaries = self.decision_memory(
                obs.memory,
                self.semantic_cards.token_embedding,
                self.name_candidates.public_definitions(state.programs),
                self.name_candidates.embedding,
            )
        encoded = self._attend_objects(
            state.objects,
            state.is_agent,
            state.validity,
            obs.attention_columns,
            live,
            summaries,
        )
        # The optional trailing token is a critic input, never an action focus.
        post_attention_objects = encoded[:, : state.validity.shape[1]]
        informed_actions = self._gather_informed_actions(
            obs, post_attention_objects, state
        )
        if self.policy_film is not None:
            valid = state.validity.unsqueeze(-1)
            pooled = (post_attention_objects * valid).sum(1) / valid.sum(1).clamp_min(1)
            informed_actions = informed_actions * (
                1 + self.policy_film(pooled)[batches]
            )
        # Autocast may compute the head in float16, which cannot represent
        # the public illegal-action sentinel. Keep that output boundary in
        # float32; the cast preserves gradients into the reduced-precision head.
        scores = self.policy_head(informed_actions).squeeze(-1).float()
        # Public logits retain original coordinates, including interior padding.
        # Only the legal rows entered the action networks; objects/programs were
        # encoded once for the batch, independent of each row's support size.
        logits = scores.new_full(capacity, -1e8)
        logits = logits.index_put((batches, slots), scores.to(logits.dtype))
        if self.debug:
            self.last_raw_logits = logits.detach().cpu()
        return logits, self._value_from_objects(encoded, state.validity)

    def encode_decision_memory(self, obs: Prepared[torch.Tensor]) -> torch.Tensor:
        """Differentiable summaries for offline probes, from admitted policy input.

        Uses the same definition, token and name embeddings as policy inference.
        No board attention or action/value head is evaluated.
        """
        if not isinstance(obs, Prepared) or not isinstance(obs.legal, torch.Tensor):
            raise TypeError("memory encoding requires device Prepared rows")
        if obs.guard is not None:
            obs.guard(self)
        if obs.binding != self._binding():
            raise ValueError(
                "observation was admitted for a different policy configuration"
            )
        if self.decision_memory is None or self.semantic_cards is None:
            raise ValueError("policy has no decision memory encoder")
        assert obs.memory is not None and obs.programs is not None
        return self.decision_memory(
            obs.memory,
            self.semantic_cards.token_embedding,
            self.name_candidates.public_definitions(
                self.semantic_cards.definitions(obs.programs)
            ),
            self.name_candidates.embedding,
        )

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

    def memory_gates(self) -> dict[str, float]:
        """Effective memory gates, read only at update reporting boundaries."""
        if self.decision_memory is None:
            return {}
        return {
            f"gate_{index}": float(
                torch.tanh(layer.memory_gate_scale * layer.memory_gate).detach()
            )
            for index, layer in enumerate((self.attention, *self.extra_attention))
            if layer.memory_gate is not None
        }

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
        summaries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend using host-prepared columns or live-token coordinates."""
        if self.live_tokens and self.config.attention_on:
            assert live is not None
            return self._attend_live(objects, is_agent, live, summaries)
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
                objects,
                is_agent,
                key_padding_mask=key_padding_mask,
                summaries=summaries,
            )
            for layer in self.extra_attention:
                objects = layer(
                    objects, is_agent, key_padding_mask, summaries=summaries
                )
        if self.final_norm is not None:
            # Padding stays zero, not the norm's bias.
            live_slots = (~key_padding_mask).unsqueeze(-1).to(objects.dtype)
            objects = self.final_norm(objects) * live_slots
        if columns is not None:
            # Restore the optional neutral value token in the same scatter,
            # at the slot immediately after the original object coordinates.
            extras = int(self.value_token is not None)
            if extras:
                columns = torch.cat(
                    (
                        columns,
                        torch.arange(
                            original_slots,
                            original_slots + extras,
                            device=columns.device,
                        ),
                    )
                )
                original_slots += extras
            objects = objects.new_zeros(
                (objects.shape[0], original_slots, objects.shape[2])
            ).index_copy(1, columns, objects)
        return objects

    def _attend_live(
        self,
        objects: torch.Tensor,
        is_agent: torch.Tensor,
        tokens: LiveTokens,
        summaries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend real objects and optional neutral value token, scattering zero padding."""
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
        packed = self.attention.forward_live(packed, owners, tokens, summaries)
        for layer in self.extra_attention:
            packed = layer.forward_live(packed, owners, tokens, summaries)
        if self.final_norm is not None:
            packed = self.final_norm(packed)
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
        if self.value_token is None:
            objects = objects[:, : validity.shape[1]]
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
        actions = actions + state.shared_context[obs.legal_batches]
        focused = post_attention_objects.flatten(0, 1)[obs.focus]
        focused = (focused * obs.focus_mask.unsqueeze(-1)).flatten(1)
        if self.focus_film is not None:
            # An absent focus is zero, so an unfocused offer is scaled by 1.
            actions = actions * (1 + self.focus_film(focused))
        return self.action_layer(torch.cat((actions, focused), dim=-1))

    def _stream(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Normalize one stream summed into object tokens (`stream_norm`)."""
        if self.stream_norms is None:
            return x
        return self.stream_norms[name](x)

    def _definition_table(self, table: torch.Tensor) -> torch.Tensor:
        """Normalize a definition table [definitions + 1, D] with row 0 kept
        zero, so a named card and an unnamed token enter at one scale."""
        if self.stream_norms is None:
            return table
        normed = self.stream_norms["definition"](table[1:])
        return torch.cat((table.new_zeros((1, table.shape[1])), normed))

    def _generic_definitions(
        self, rows: torch.Tensor, *, table: torch.Tensor | None
    ) -> torch.Tensor:
        """Identity rows of a world without card programs."""
        assert self.card_definitions is not None
        if table is not None:
            return table[rows]
        return self.card_definitions(rows)

    def _gather_object_embeddings(
        self, obs: Prepared[torch.Tensor], programs: torch.Tensor | None = None
    ) -> EmbeddedObservation:
        enc_agent_player = self._stream(
            "player", self.player_embedding(obs.agent_player)
        )
        enc_opp_player = self._stream(
            "player", self.player_embedding(obs.opponent_player)
        )
        enc_agent_cards = self._stream("card", self.card_embedding(obs.agent_cards))
        enc_opp_cards = self._stream("card", self.card_embedding(obs.opponent_cards))
        definitions = None
        if self.semantic_cards is not None:
            if programs is None:
                assert obs.programs is not None
                programs = self.semantic_cards.definitions(obs.programs)
            # Every object reader sees the normalized table; the raw
            # `programs` keep scoring naming candidates and decision memory.
            definitions = self._definition_table(
                self.name_candidates.public_definitions(programs)
            )
            cards, known = self.semantic_cards(obs, definitions)
            enc_agent_player = enc_agent_player + known[:, 0, None]
            enc_opp_player = enc_opp_player + known[:, 1, None]
            lookup = partial(self.semantic_cards.lookup, definitions)
        else:
            assert self.card_definitions is not None
            table = (
                self._definition_table(self.card_definitions.weight)
                if self.stream_norms is not None
                else None
            )
            lookup = partial(self._generic_definitions, table=table)
            cards = lookup(obs.card_rows)
        enc_agent_cards = enc_agent_cards + cards[:, 0]
        enc_opp_cards = enc_opp_cards + cards[:, 1]
        permanent_definitions = lookup(obs.permanent_rows)
        # Keep the per-seat contiguous projection layout after biases train.
        enc_agent_perms = (
            self._stream(
                "permanent", self.perm_embedding(obs.permanents[:, 0].contiguous())
            )
            + permanent_definitions[:, 0]
        )
        enc_opp_perms = (
            self._stream(
                "permanent", self.perm_embedding(obs.permanents[:, 1].contiguous())
            )
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
        if self.stack_tokens:
            assert self.stack_embedding is not None and obs.stack is not None
            assert obs.stack_rows is not None
            parts.append(
                self._stream("stack", self.stack_embedding(obs.stack))
                + lookup(obs.stack_rows)
            )
        if obs.belief is not None:
            rows, global_row = self._belief_rows(obs.belief)
            parts.extend((rows, global_row))
        objects = self._join_permanent_links(obs, torch.cat(parts, dim=1))
        if self.stack_tokens:
            objects = self._join_stack_sources(obs, objects)
        token_embedding = (
            self.semantic_cards.token_embedding
            if self.semantic_cards is not None and self.semantic_cards.records
            else None
        )
        objects = self.facts(obs.facts, objects, token_embedding, definitions)
        objects = self.shared_effects(
            obs.effects, objects, token_embedding, definitions
        )
        encoded_relations = self.continuation.encode_rows(
            obs, objects, lookup(obs.relation_rows)
        )
        relation_count = obs.relation_valid.sum(1, keepdim=True)
        shared_context = encoded_relations.sum(1) / relation_count.clamp_min(1)
        frame_context = self.continuation.frame_context(obs, encoded_relations)
        if self.recent_events is not None:
            assert definitions is not None and obs.events is not None
            shared_context = (
                self.recent_events(obs.events, objects, definitions) + shared_context
            )
        if self.stream_norms is not None:
            # Zero, not a unit-scale bias, for a decision with no context.
            present = relation_count.squeeze(-1) > 0
            if self.recent_events is not None:
                assert obs.events is not None
                present = present | obs.events.validity.any(-1)
            shared_context = self._stream(
                "context", shared_context
            ) * present.unsqueeze(-1).to(shared_context.dtype)
        objects = objects + shared_context.unsqueeze(1)
        return EmbeddedObservation(
            objects, obs.is_agent, obs.validity, shared_context, frame_context, programs
        )

    def _join_stack_sources(
        self, obs: Prepared[torch.Tensor], objects: torch.Tensor
    ) -> torch.Tensor:
        """Add source embeddings to the host-prepared v11 stack rows."""
        assert self.stack_source is not None and obs.stack is not None
        assert obs.stack_targets is not None and obs.stack_mask is not None
        start = 2 + 2 * obs.agent_cards.shape[1] + 2 * obs.permanents.shape[2]
        count = obs.stack.shape[1]
        sources = objects.flatten(0, 1)[obs.stack_targets].reshape(
            objects.shape[0], count, objects.shape[2]
        )
        delta = self.stack_source(sources).masked_fill(
            (obs.stack_mask == 0).unsqueeze(-1), 0
        )
        return torch.cat(
            (
                objects[:, :start],
                objects[:, start : start + count] + delta.to(objects.dtype),
                objects[:, start + count :],
            ),
            dim=1,
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
        self.memory = CollectionMemory()
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
            self.memory.clear()
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
        self.memory.clear()
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
            logits, _ = self.agent(prepared)
        with measure("output_transfer_host"):
            return InferenceResult(
                to_numpy(logits), self.transfer_bytes, self.transfers
            )

    @torch.no_grad()
    def prepare(self, obs: ObservationBuffers) -> Prepared[torch.Tensor]:
        self.check(self.agent)
        if any(not isinstance(value, np.ndarray) for value in obs.values()):
            raise TypeError("inference preparation requires NumPy observation buffers")
        batch = self.agent.prepare(
            obs,
            inference=True,
            memory_roots=self.memory.roots(obs)
            if self.agent.config.decision_memory == "gru"
            else None,
        )
        if self.cache_catalog:
            batch = replace(batch, programs=None)
        arrays = (
            []
            if self.device.type == "cpu"
            else array_fields(batch, np.ndarray).values()
        )
        self.transfer_bytes = sum(array.nbytes for array in arrays)
        self.transfers = len({array.dtype for array in arrays})
        return replace(
            self.memory.bind(on_device(batch, self.device), obs),
            guard=self._forward_check,
        )


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

    Post-normalization (`norm(x + f(x))`) is the historical layout. With
    `pre_norm` each block reads a normalized copy of the residual stream,
    `x + mha(norm1(x))` then `x + mlp(norm2(x))` (Xiong et al., 2020), so
    query/key size does not grow with the number of streams summed into a
    token; the caller then owns one final norm. Parameter names are shared.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        *,
        ownership: bool = True,
        feedforward_dim: int | None = None,
        memory_gate_scale: float = 1.0,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.memory_gate: nn.Parameter | None = None
        self.memory_projection: nn.Linear | None = None
        self.memory_gate_scale = memory_gate_scale
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
        summaries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.memory_gate is not None:
            assert summaries is not None and self.memory_projection is not None
            # Mean keeps K>1 summaries usable without extending the attention sequence.
            delta = self.memory_projection(summaries.mean(1).to(objects.dtype))
            delta = torch.tanh(self.memory_gate_scale * self.memory_gate) * delta
            board_mask = (~key_padding_mask).clone()
            board_mask[:, is_agent.shape[1] :] = False
            objects = objects + delta.unsqueeze(1) * board_mask.unsqueeze(-1)
        owned_objects = objects
        if self.perspective is not None:
            perspective_scale = torch.where(is_agent.unsqueeze(-1), 1.0, -1.0)
            # Appended value tokens are neutral; real-object indexes stay fixed.
            if objects.shape[1] > is_agent.shape[1]:
                perspective_scale = torch.cat(
                    (
                        perspective_scale,
                        perspective_scale.new_zeros(
                            (objects.shape[0], objects.shape[1] - is_agent.shape[1], 1)
                        ),
                    ),
                    dim=1,
                )
            owned_objects = objects + perspective_scale * self.perspective

        mask = (~key_padding_mask).unsqueeze(-1).float()
        attention_input = self.norm1(owned_objects) if self.pre_norm else owned_objects
        attended, _ = self.mha(
            attention_input,
            attention_input,
            attention_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self._after_attention(owned_objects + attended) * mask

    def _after_attention(self, residual: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            return residual + self.mlp(self.norm2(residual))
        x = self.norm1(residual)
        return self.norm2(x + self.mlp(x))

    def forward_live(
        self,
        packed: torch.Tensor,
        owners: torch.Tensor,
        tokens: LiveTokens,
        summaries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`forward` over packed live tokens [L,D]; `owners` [L] is +1 for the
        acting viewer's objects, -1 for the opponent's and 0 for a value token."""
        if self.memory_gate is not None:
            assert summaries is not None and self.memory_projection is not None
            delta = self.memory_projection(summaries.mean(1).to(packed.dtype))
            delta = torch.tanh(self.memory_gate_scale * self.memory_gate) * delta
            # Packed coordinates identify the observation; value tokens have owner zero.
            packed = packed + delta[tokens.index // tokens.slots] * (
                owners != 0
            ).unsqueeze(-1)
        if self.perspective is not None:
            packed = packed + owners.unsqueeze(-1) * self.perspective
        attention_input = self.norm1(packed) if self.pre_norm else packed
        return self._after_attention(
            packed + tokens.self_attention(self.mha, attention_input)
        )


def layer_init(
    layer: nn.Module, gain: float = 1.0, bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer
