"""
Shared viewer-observation encoder, legal-action policy and scalar or WDL critic.

ActorCritic.forward preserves the serving policy/value interface. Training uses
forward_distribution when it needs loss/draw/win logits rather than their mean.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

# Local imports
from manabot.env.observation import (
    FACT_KINDS,
    FACT_KINDS_V8,
    FACT_SOURCE_INDEX,
    FACT_VALID_INDEX,
    PERMANENT_DEFINITION_INDEX,
    PERMANENT_LINK_INDEXES,
    RELATION_VALID_INDEX,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig
from manabot.infra.log import getLogger
from manabot.model.continuation import ContinuationEncoder
from manabot.model.facts import FactEncoder, RecordVocabulary
from manabot.model.inference import PreparedObservation
from manabot.model.name_candidates import NameCandidateEncoder
from manabot.model.policy_distribution import legal_mask
from manabot.model.recent_events import RecentEventEncoder
from manabot.model.semantic_cards import (
    RecordEmbeddings,
    SemanticCardEncoder,
    admit_definition_ids,
)
from manabot.model.shared_effects import SharedEffectEncoder
from manabot.model.state_records import SharedFactEncoder, TypedContinuationEncoder
import managym


@dataclass
class EmbeddedObservation:
    """One encoding shared by object attention, action joins and the critic."""

    objects: torch.Tensor
    is_agent: torch.Tensor
    validity: torch.Tensor
    shared_context: torch.Tensor | None
    frame_context: torch.Tensor | None
    # Raw executable programs; object inputs separately add public name identity.
    programs: torch.Tensor | None


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
        self, observation_space: ObservationSpace, config: ModelConfig
    ) -> None:
        super().__init__()
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
        if config.state_encoder != "rows" and enc.config.policy_input_version not in (
            5,
            7,
            8,
        ):
            raise ValueError(
                "typed state experiment requires policy input v5, v7 or v8"
            )
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
        # Definition IDs and v5 permanent links are categorical joins, never
        # numeric permanent features.
        self._link_columns = PERMANENT_LINK_INDEXES if enc.shows_everything else ()
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
                decklists=enc.config.policy_input_version >= 5,
                records=config.catalog_records(enc.config.policy_input_version),
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

        self.continuation: ContinuationEncoder | TypedContinuationEncoder | None
        if config.state_encoder != "rows":
            self.continuation = TypedContinuationEncoder(
                embed_dim, config.state_encoder, version=enc.config.policy_input_version
            )
        else:
            self.continuation = (
                ContinuationEncoder(embed_dim, enc.relation_dim)
                if enc.config.policy_input_version >= 2
                else None
            )

        # Policy input v5 links permanents to other objects: an Aura or
        # Equipment to its host, a blocker to the attacker it blocks. Each
        # link exchanges pre-attention embeddings both ways: the linked
        # permanent reads its target and the target reads (summed) linked
        # permanents, so an Equipment's effect, an Honest Work host's set
        # characteristics and a blocking assignment reach both objects. Zero
        # weights start additive; skip_init draws no RNG.
        self.link_from_target = nn.ModuleList()
        self.link_to_target = nn.ModuleList()
        self._permanents_start = 2 + 2 * enc.cards_per_player
        for _ in self._link_columns:
            for layers in (self.link_from_target, self.link_to_target):
                layer = torch.nn.utils.skip_init(nn.Linear, embed_dim, embed_dim)
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                layers.append(layer)
        # Policy input v5 fact rows (counter kinds, added subtypes, set
        # characteristics, spent abilities, continuous-effect records) join
        # their objects. Built last so earlier parameters keep their draws.
        # Catalog ids share the card encoder's token and definition rows.
        record_vocabulary: RecordVocabulary = (
            "catalog"
            if config.catalog_records(enc.config.policy_input_version)
            else "hashed"
        )
        self.facts: FactEncoder | SharedFactEncoder | None
        if config.state_encoder != "rows" and enc.config.policy_input_version == 5:
            self.facts = SharedFactEncoder(embed_dim, config.state_encoder)
        else:
            self.facts = (
                FactEncoder(
                    embed_dim,
                    enc.config.fact_rows,
                    kinds=FACT_KINDS_V8
                    if enc.config.policy_input_version >= 8
                    else FACT_KINDS,
                    vocabulary=record_vocabulary,
                )
                if enc.shows_everything
                else None
            )

        self.shared_effects = (
            SharedEffectEncoder(
                embed_dim,
                "typed_gru" if config.state_encoder == "typed_gru" else "typed_pool",
                version=enc.config.policy_input_version,
                vocabulary=record_vocabulary,
            )
            if enc.config.policy_input_version >= 7
            else None
        )

        self.name_candidates = (
            NameCandidateEncoder(
                managym.naming_catalog(),
                embed_dim,
                self.semantic_cards.definition_rows
                if self.semantic_cards is not None
                else None,
            )
            if enc.config.policy_input_version >= 9
            else None
        )

        # When True, forward() stashes legal logits at original coordinates on
        # self.last_raw_logits for offline diagnosis. Off by default: the
        # detach/cpu copy is measurable in the inference hot path.
        self.debug = False
        self.last_raw_logits: Optional[torch.Tensor] = None

    def forward(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and raw critic output from one shared encoding.

        Categorical output [B,3] contains loss/draw/win logits from the acting
        viewer's perspective; scalar output [B,1] is an unconstrained value.
        Neither output is detached: categorical supervision can train the
        critic and shared encoder directly without passing through its mean.
        A distributional head alone does not select a categorical loss.
        """
        state = self._gather_object_embeddings(obs)

        encoded = self._attend_objects(
            state.objects,
            state.is_agent,
            state.validity,
            objects_checked=isinstance(obs, PreparedObservation)
            and obs.objects_checked,
        )
        # The optional trailing token is a critic input, never an action focus.
        post_attention_objects = encoded[:, : state.validity.shape[1]]

        batches, slots = legal_mask(obs["actions_valid"]).nonzero(as_tuple=True)
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
        logits = scores.new_full(obs["actions_valid"].shape, -1e8)
        logits = logits.index_put((batches, slots), scores)
        if self.debug:
            self.last_raw_logits = logits.detach().cpu()
        return logits, self._value_from_objects(encoded, state.validity)

    def _attend_objects(
        self,
        objects: torch.Tensor,
        is_agent: torch.Tensor,
        validity: torch.Tensor,
        *,
        objects_checked: bool = False,
    ) -> torch.Tensor:
        """Encode real objects and, when configured, a trailing neutral value token."""
        if (
            not objects_checked
            and self.config.value_aggregation != "historical_mean"
            and (validity.sum(1) == 0).any()
        ):
            raise ValueError("value aggregation requires a valid object")
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
        return objects

    def _value_from_objects(
        self, objects: torch.Tensor, validity: torch.Tensor
    ) -> torch.Tensor:
        """Read the critic while preserving historical value_head parameter names."""
        if self.config.value_aggregation == "historical_mean":
            return self.value_head(objects)
        if self.value_token is not None:
            objects = objects[:, -1]
        projected = self.value_head[1](self.value_head[0](objects))
        if self.config.value_aggregation == "masked_mean":
            projected = projected.masked_fill((validity == 0).unsqueeze(-1), 0)
            projected = projected.sum(1) / validity.sum(1, keepdim=True)
        # Indexing avoids constructing temporary Sequential modules per decision.
        return self.value_head[5](self.value_head[4](self.value_head[3](projected)))

    def _gather_informed_actions(
        self,
        obs: dict[str, torch.Tensor],
        post_attention_objects: torch.Tensor,
        state: EmbeddedObservation,
        batches: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """Embed selected legal rows [K,D], preserving their original joins.

        Coordinates select actions, continuation frames and focus objects from
        the same observation row.
        """
        action_rows = obs["actions"][batches, slots]
        actions = self.action_embedding(action_rows[..., :-2])
        indexes = action_rows[..., -2]
        if not ((indexes == indexes.floor()) & (indexes >= 0) & (indexes <= 64)).all():
            raise ValueError("invalid action continuation reference")
        if state.frame_context is not None:
            rows = obs["relations"]
            # Select the two join columns before gathering legal action rows;
            # duplicating complete relation payloads would undo the saving.
            present = (
                (indexes[:, None] == rows[..., 1][batches])
                & (rows[..., RELATION_VALID_INDEX][batches] == 1)
            ).any(-1)
            if ((indexes != 0) & ~present).any():
                raise ValueError("action continuation reference has no frame")
            linked = state.frame_context[batches, (indexes.long() - 1).clamp_min(0)]
            actions = actions + linked.masked_fill((indexes == 0).unsqueeze(-1), 0)
        elif (indexes != 0).any():
            raise ValueError("trigger ordering requires continuation policy input")
        if self.name_candidates is not None:
            candidates = obs["name_candidates"][batches, slots]
            actions = actions + self.name_candidates(candidates)
            if self.semantic_cards is not None and state.programs is not None:
                actions = actions + self.semantic_cards.embed_definitions(
                    candidates[..., 1], state.programs
                )
        if state.shared_context is not None:
            actions = actions + state.shared_context[batches]
        focus = obs["action_focus"][batches, slots]
        if not (
            (focus == focus.floor())
            & (focus >= -1)
            & (focus < post_attention_objects.shape[1])
        ).all():
            raise ValueError("invalid action focus reference")
        focused = post_attention_objects[batches[:, None], focus.clamp_min(0).long()]
        focused = focused * (focus != -1).unsqueeze(-1)
        return self.action_layer(torch.cat((actions, focused.flatten(1)), dim=-1))

    def _gather_object_embeddings(
        self, obs: Dict[str, torch.Tensor]
    ) -> EmbeddedObservation:
        if isinstance(obs, PreparedObservation):
            obs.scope.check(self)
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "prepared inference observations cannot enter autograd"
                )
        enc_agent_player = self.player_embedding(obs["agent_player"])
        enc_opp_player = self.player_embedding(obs["opponent_player"])
        enc_agent_cards = self.card_embedding(obs["agent_cards"])
        enc_opp_cards = self.card_embedding(obs["opponent_cards"])
        definitions = None
        programs = None
        references = obs.references if isinstance(obs, PreparedObservation) else None
        permanent_ids = torch.stack(
            [
                obs[key][..., PERMANENT_DEFINITION_INDEX]
                for key in ("agent_permanents", "opponent_permanents")
            ],
            dim=1,
        )
        if self.semantic_cards is not None:
            # Learning encodes only referenced definitions (same values within
            # float rounding, identical gradient support); without autograd the
            # full table keeps unprepared inference bitwise equal to serving.
            if isinstance(obs, PreparedObservation):
                programs = obs.scope.catalog()
            elif torch.is_grad_enabled():
                programs = self.semantic_cards.definitions(
                    self._referenced_definitions(obs, permanent_ids)
                )
            else:
                programs = self.semantic_cards.definitions()
            assert programs is not None
            definitions = (
                self.name_candidates.public_definitions(programs)
                if self.name_candidates is not None
                else programs
            )
            cards, known = self.semantic_cards(obs, definitions, references)
            enc_agent_cards = enc_agent_cards + cards[:, 0]
            enc_opp_cards = enc_opp_cards + cards[:, 1]
            enc_agent_player = enc_agent_player + known[:, 0, None]
            enc_opp_player = enc_opp_player + known[:, 1, None]
            permanent_definitions = (
                self.semantic_cards.embed_definitions(permanent_ids, definitions)
                if references is None
                else definitions[references.permanents]
            )
        else:
            cards = self._generic_definitions(
                obs["semantic_cards"], None if references is None else references.cards
            )
            enc_agent_cards = enc_agent_cards + cards[:, 0]
            enc_opp_cards = enc_opp_cards + cards[:, 1]
            permanent_definitions = self._generic_definitions(
                permanent_ids, None if references is None else references.permanents
            )
        permanent_features = [
            obs[key][..., self._numeric_permanent_columns]
            for key in ("agent_permanents", "opponent_permanents")
        ]
        enc_agent_perms = (
            self.perm_embedding(permanent_features[0]) + permanent_definitions[:, 0]
        )
        enc_opp_perms = (
            self.perm_embedding(permanent_features[1]) + permanent_definitions[:, 1]
        )
        object_parts = [
            enc_agent_player,
            enc_opp_player,
            enc_agent_cards,
            enc_opp_cards,
            enc_agent_perms,
            enc_opp_perms,
        ]
        validity_parts = [
            obs["agent_player_valid"],
            obs["opponent_player_valid"],
            obs["agent_cards_valid"],
            obs["opponent_cards_valid"],
            obs["agent_permanents_valid"],
            obs["opponent_permanents_valid"],
        ]

        belief_is_agent = None
        if self.belief_count_buckets > 0:
            belief_rows, belief_validity, belief_is_agent, belief_global = (
                self._belief_rows(obs)
            )
            object_parts.extend((belief_rows, belief_global))
            validity_parts.extend(
                (
                    belief_validity,
                    torch.ones(
                        (belief_global.shape[0], 1),
                        device=belief_global.device,
                    ),
                )
            )

        objects = torch.cat(object_parts, dim=1)
        if self._link_columns:
            objects = self._join_permanent_links(obs, objects)
        # Catalog fact and effect ids read the card encoder's rows.
        catalog = None
        if self.semantic_cards is not None and self.semantic_cards.records:
            assert definitions is not None
            catalog = RecordEmbeddings(self.semantic_cards, definitions)
        if isinstance(self.facts, FactEncoder):
            objects = self.facts(obs["facts"], objects, catalog)
        elif self.facts is not None:
            objects = self.facts(obs["facts"], objects)
        if self.shared_effects is not None:
            objects = self.shared_effects(
                obs["effects"],
                obs["effect_details"],
                obs["effect_members"],
                objects,
                catalog,
            )
        # History and current continuation both inform objects and focus-free actions.
        shared_context = None
        frame_context = None
        if self.recent_events is not None:
            assert self.semantic_cards is not None and definitions is not None
            shared_context = self.recent_events(
                obs, objects, definitions, self.semantic_cards.definition_rows
            )
        if self.continuation is not None:
            rows = obs["relations"]
            ids = rows[..., [5, 23]]
            relation_definitions = (
                self.semantic_cards.embed_definitions(ids, definitions)
                if self.semantic_cards is not None and definitions is not None
                else self._generic_definitions(ids)
            )
            if isinstance(self.continuation, TypedContinuationEncoder):
                continuation, frame_context = self.continuation(
                    rows, objects, relation_definitions
                )
            else:
                encoded_relations = self.continuation.encode_rows(
                    rows, objects, relation_definitions
                )
                continuation = encoded_relations.sum(1) / rows[
                    ..., RELATION_VALID_INDEX
                ].sum(1, keepdim=True).clamp_min(1)
                frame_context = self.continuation.frame_context(rows, encoded_relations)
            shared_context = (
                continuation
                if shared_context is None
                else shared_context + continuation
            )
        if shared_context is not None:
            objects = objects + shared_context.unsqueeze(1)

        # The visible object layout is static, so its ownership mask is a
        # registered buffer. Explicit belief rows extend it below.
        batch = objects.shape[0]
        is_agent = self._is_agent_template.expand(batch, -1)
        if belief_is_agent is not None:
            is_agent = torch.cat(
                (
                    is_agent,
                    belief_is_agent,
                    torch.zeros((batch, 1), dtype=torch.bool, device=objects.device),
                ),
                dim=1,
            )

        validity = torch.cat(validity_parts, dim=1)
        return EmbeddedObservation(
            objects, is_agent, validity, shared_context, frame_context, programs
        )

    def _join_permanent_links(
        self, obs: Dict[str, torch.Tensor], objects: torch.Tensor
    ) -> torch.Tensor:
        """Add each link target's embedding to its linked permanents and theirs to it.

        Links are object index + 1 (zero: none) into `objects` [B,N,D].
        """
        start = self._permanents_start
        delta = torch.zeros_like(objects)
        for column, from_target, to_target in zip(
            self._link_columns, self.link_from_target, self.link_to_target
        ):
            links = torch.cat(
                [
                    obs[key][..., column]
                    for key in ("agent_permanents", "opponent_permanents")
                ],
                dim=1,
            )
            if not (
                (links == links.floor()) & (links >= 0) & (links <= objects.shape[1])
            ).all():
                raise ValueError("invalid permanent link object reference")
            permanents = objects[:, start : start + links.shape[1]]
            linked = (links > 0).unsqueeze(-1)
            targets = (
                (links.long() - 1).clamp_min(0).unsqueeze(-1).expand_as(permanents)
            )
            delta = delta.scatter_add(
                1, targets, to_target(permanents).masked_fill(~linked, 0)
            )
            delta[:, start : start + links.shape[1]] += from_target(
                objects.gather(1, targets)
            ).masked_fill(~linked, 0)
        return objects + delta

    def _referenced_definitions(
        self, obs: Dict[str, torch.Tensor], permanent_ids: torch.Tensor
    ) -> torch.Tensor:
        """Unique catalog rows + 1 this batch can index: visible cards,
        permanents, known hands, decklists, public history, relations and names.

        A source added to the forward pass must be added here, or its rows
        read zero embeddings.
        """
        assert self.semantic_cards is not None
        ids = [obs["semantic_cards"], permanent_ids, obs["known_hand"][..., 0]]
        if self.semantic_cards.decklist_projection is not None:
            ids.append(obs["decklists"][..., 0])
        events, valid = obs.get("events"), obs.get("events_valid")
        if (
            self.recent_events is not None
            and events is not None
            and valid is not None
            and events.ndim == 3
            and events.shape[-1] == 12
            and valid.shape == events.shape[:2]
        ):
            # Malformed history is rejected by RecentEventEncoder itself.
            ids.extend(events[..., column][valid == 1] for column in (2, 7))
        if self.continuation is not None:
            ids.append(obs["relations"][..., [5, 23]])
        if self.semantic_cards.records:
            # Catalog fact and effect sources read their definitions' rows.
            if self.facts is not None:
                facts = obs["facts"]
                ids.append(
                    facts[..., FACT_SOURCE_INDEX][facts[..., FACT_VALID_INDEX] == 1]
                )
            if self.shared_effects is not None:
                effects = obs["effects"]
                ids.append(effects[..., 11][effects[..., 17] == 1])
        if self.name_candidates is not None:
            # Names share the same program table as objects. Only legal naming
            # rows widen the learner's support to the complete naming catalog.
            ids.append(obs["name_candidates"][..., 1][obs["actions_valid"].bool()])
        return self.semantic_cards.referenced_rows(ids)

    def _generic_definitions(
        self, ids: torch.Tensor, admitted: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Reject unbound/fractional IDs instead of silently aliasing definitions."""
        assert self.card_definitions is not None
        rows = (
            admit_definition_ids(ids, self.card_definitions.num_embeddings)
            if admitted is None
            else admitted
        )
        return self.card_definitions(rows)

    def _belief_rows(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build explicit semantic belief rows and one global diagnostic row."""

        required = {
            "belief_card_def_ids",
            "belief_owner_role_ids",
            "belief_hidden_zone_ids",
            "belief_count_probabilities",
            "belief_validity",
            "belief_globals",
        }
        missing = sorted(required.difference(obs))
        if missing:
            raise ValueError(f"belief-enabled ActorCritic is missing inputs: {missing}")
        card_ids = obs["belief_card_def_ids"].long()
        owner_ids = obs["belief_owner_role_ids"].long()
        zone_ids = obs["belief_hidden_zone_ids"].long()
        probabilities = obs["belief_count_probabilities"].float()
        validity = obs["belief_validity"].float()
        globals_ = obs["belief_globals"].float()
        if card_ids.ndim != 2:
            raise ValueError("belief ids must have shape [batch, rows]")
        expected_ids = card_ids.shape
        if owner_ids.shape != expected_ids or zone_ids.shape != expected_ids:
            raise ValueError("belief semantic ids must share shape [batch, rows]")
        if probabilities.shape != (*expected_ids, self.belief_count_buckets):
            raise ValueError(
                "belief count probabilities must have shape [batch, rows, buckets]"
            )
        if validity.shape != expected_ids:
            raise ValueError("belief validity must have shape [batch, rows]")
        if globals_.shape != (expected_ids[0], 2):
            raise ValueError("belief globals must have shape [batch, 2]")
        if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("belief probabilities must be finite and non-negative")
        if not torch.isfinite(globals_).all():
            raise ValueError("belief globals must be finite")
        valid_rows = validity > 0
        probability_totals = probabilities.sum(dim=-1)
        if valid_rows.any() and not torch.allclose(
            probability_totals[valid_rows],
            torch.ones_like(probability_totals[valid_rows]),
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError("valid belief rows must be normalized")
        for ids, embedding, label in (
            (card_ids, self.belief_card_embedding, "card definition"),
            (owner_ids, self.belief_owner_embedding, "owner role"),
            (zone_ids, self.belief_zone_embedding, "hidden zone"),
        ):
            if (ids < 0).any() or (ids >= embedding.num_embeddings).any():
                raise ValueError(f"belief {label} id is outside the model vocabulary")

        rows = (
            self.belief_card_embedding(card_ids)
            + self.belief_owner_embedding(owner_ids)
            + self.belief_zone_embedding(zone_ids)
            + self.belief_count_embedding(probabilities)
        )
        rows = rows * validity.unsqueeze(-1)
        global_row = self.belief_global_embedding(globals_).unsqueeze(1)
        return rows, validity, owner_ids == 0, global_row

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        if (obs["actions_valid"].sum(dim=-1) == 0).any():
            raise ValueError("No valid actions available")
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            _, value = self.forward(obs)
            return value


# -----------------------------------------------------------------------------
# Basic Building Blocks
# -----------------------------------------------------------------------------
class MeanPoolingLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


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


def layer_init(
    layer: nn.Module, gain: float = 1.0, bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
