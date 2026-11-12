"""
Shared viewer-observation encoder, legal-action policy and scalar or WDL critic.

ActorCritic.forward preserves the serving policy/value interface. Training uses
forward_distribution when it needs loss/draw/win logits rather than their mean.
Both read only observations admitted by `manabot.model.admission`, which owns
every input check; the forward does no validation host reads.

Object attention runs over the fixed object tables. With `bucketed_shapes=True`
(`manabot.model.buckets`), each forward trims its variable axes to power-of-two
buckets and returns logits at full capacity.
With `live_tokens=True` (`manabot.model.live_tokens`), table attention runs
over the batch's valid objects packed into one sequence; either setting
composes with the other and neither changes the model.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

# Local imports
from manabot.env.observation import (
    FACT_SOURCE_INDEX,
    FACT_VALID_INDEX,
    PERMANENT_DEFINITION_INDEX,
    PERMANENT_LINK_INDEXES,
    RELATION_VALID_INDEX,
    TURN_STATE_DIM,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig
from manabot.infra.log import getLogger
from manabot.model.admission import AdmittedObservation, admission_binding
from manabot.model.buckets import (
    BucketSizes,
    bucket_sizes,
    object_columns,
)
from manabot.model.continuation import ContinuationEncoder
from manabot.model.facts import FactEncoder, RecordVocabulary
from manabot.model.inference import PreparedObservation, trim_admitted, trim_prepared
from manabot.model.live_tokens import LiveTokens
from manabot.model.name_candidates import NameCandidateEncoder
from manabot.model.recent_events import RecentEventEncoder
from manabot.model.semantic_cards import (
    RecordEmbeddings,
    SemanticCardEncoder,
    definition_rows,
)
from manabot.model.shared_effects import SharedEffectEncoder
import managym

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

    def forward(self, obs: AdmittedObservation) -> tuple[torch.Tensor, torch.Tensor]:
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
        self, obs: AdmittedObservation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and raw critic output from one shared encoding.

        Categorical output [B,3] contains loss/draw/win logits from the acting
        viewer's perspective; scalar output [B,1] is an unconstrained value.
        Neither output is detached: categorical supervision can train the
        critic and shared encoder directly without passing through its mean.
        A distributional head alone does not select a categorical loss.
        """
        capacity = obs["actions_valid"].shape
        # Trimming keeps every valid object, so the count holds for buckets too.
        live = obs.live_token_count if isinstance(obs, PreparedObservation) else None
        buckets = None
        if self.bucketed_shapes:
            obs, buckets = self._bucketed(obs)
        state = self._gather_object_embeddings(obs, buckets)
        batches, slots = obs["actions_valid"].bool().nonzero(as_tuple=True)

        encoded = self._attend_objects(
            state.objects, state.is_agent, state.validity, live
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

    def live_token_count(self, obs: Dict[str, torch.Tensor]) -> int | None:
        """Host count of the tokens live-token attention packs for CPU `obs`.

        Valid table objects, belief rows and their global row, and one value
        token per observation; None unless this policy packs live tokens.
        """
        if not self.live_tokens or not self.config.attention_on:
            return None
        batch = obs["actions_valid"].shape[0]
        count = sum(int(obs[key].count_nonzero()) for key in OBJECT_VALIDITY_KEYS)
        if self.belief_count_buckets > 0:
            # Belief rows plus one global belief row per observation.
            count += int(obs["belief_validity"].count_nonzero()) + batch
        if self.value_token is not None:
            count += batch
        return count

    def object_trimming(self) -> bool:
        """Whether buckets may trim object axes: padding objects must be inert."""
        return (
            self.config.value_aggregation != "historical_mean"
            and self.belief_count_buckets == 0
        )

    def _bucketed(
        self, obs: AdmittedObservation
    ) -> tuple[AdmittedObservation, BucketSizes | None]:
        """Trim `obs` to its buckets; prepared inputs were measured on the host.

        Without buckets (a layout admission did not judge), `obs` is returned
        unchanged.
        """
        self._require_admitted(obs)
        if isinstance(obs, PreparedObservation):
            return trim_prepared(obs), obs.buckets
        buckets = bucket_sizes(
            obs, self.observation_space.encoder, objects=self.object_trimming()
        )
        if buckets is None:
            return obs, None
        return trim_admitted(obs, buckets), buckets

    def _require_admitted(self, obs: Dict[str, torch.Tensor]) -> None:
        if not isinstance(obs, AdmittedObservation):
            raise TypeError(
                "policy forward requires an observation from "
                "manabot.model.admission.admit_observation"
            )
        if obs.binding != admission_binding(self):
            raise ValueError(
                "observation was admitted for a different policy configuration"
            )

    def _attend_objects(
        self,
        objects: torch.Tensor,
        is_agent: torch.Tensor,
        validity: torch.Tensor,
        live: int | None = None,
    ) -> torch.Tensor:
        """Encode real objects and, when configured, a trailing neutral value token.

        `live` is the host-measured token count for live-token attention.
        """
        if self.live_tokens and self.config.attention_on:
            return self._attend_live(objects, is_agent, validity, live)
        original_slots = objects.shape[1]
        columns = None
        if (
            self.config.attention_on
            and objects.device.type == "cpu"
            and not torch.is_grad_enabled()
        ):
            # Padding is never an attention key and every layer zeros its query
            # output. Omit columns unused by the entire inference batch, then
            # restore the original coordinates for focus joins and historical
            # mean pooling (whose denominator includes padding). Learning keeps
            # fixed shapes and its existing autograd path; device inference
            # avoids a host synchronization to discover the compact shape.
            used = (validity != 0).any(0).nonzero().flatten()
            if 0 < used.numel() < original_slots:
                columns = used
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
        validity: torch.Tensor,
        live: int | None,
    ) -> torch.Tensor:
        """`_attend_objects` over valid objects only, returned at [B,N(+1),D].

        The value token is one more live token per observation, ownerless as
        on the padded path. Padding slots come back zero, as every padded
        layer leaves them, so focus joins and every value aggregation read the
        same values.
        """
        batch = objects.shape[0]
        owner = torch.where(is_agent, 1.0, -1.0).to(objects.dtype)
        present = validity != 0
        if self.value_token is not None:
            objects = torch.cat(
                (objects, self.value_token.expand(batch, -1, -1).to(objects.dtype)),
                dim=1,
            )
            owner = torch.cat((owner, owner.new_zeros((batch, 1))), dim=1)
            present = torch.cat((present, present.new_ones((batch, 1))), dim=1)
        tokens = LiveTokens.pack(present, live)
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
        linked = state.frame_context[batches, (indexes.long() - 1).clamp_min(0)]
        actions = actions + linked.masked_fill((indexes == 0).unsqueeze(-1), 0)
        candidates = obs["name_candidates"][batches, slots]
        actions = actions + self.name_candidates(candidates)
        if self.semantic_cards is not None and state.programs is not None:
            actions = actions + self.semantic_cards.embed_definitions(
                candidates[..., 1], state.programs
            )
        actions = actions + state.shared_context[batches]
        focus = obs["action_focus"][batches, slots]
        focused = post_attention_objects[batches[:, None], focus.clamp_min(0).long()]
        focused = focused * (focus != -1).unsqueeze(-1)
        return self.action_layer(torch.cat((actions, focused.flatten(1)), dim=-1))

    def _gather_object_embeddings(
        self, obs: Dict[str, torch.Tensor], buckets: BucketSizes | None = None
    ) -> EmbeddedObservation:
        self._require_admitted(obs)
        assert isinstance(obs, AdmittedObservation)
        if isinstance(obs, PreparedObservation):
            obs.scope.check(self)
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "prepared inference observations cannot enter autograd"
                )
        agent_player, opp_player = obs["agent_player"], obs["opponent_player"]
        agent_cards, opp_cards = obs["agent_cards"], obs["opponent_cards"]
        if self.turn_state:
            # [B, 2, TURN_STATE_DIM] and [B, 2, cards]: agent seat first.
            players, cards = obs["player_turn_state"], obs["card_turn_state"]
            agent_player = torch.cat((agent_player, players[:, 0:1]), dim=-1)
            opp_player = torch.cat((opp_player, players[:, 1:2]), dim=-1)
            agent_cards = torch.cat((agent_cards, cards[:, 0, :, None]), dim=-1)
            opp_cards = torch.cat((opp_cards, cards[:, 1, :, None]), dim=-1)
        enc_agent_player = self.player_embedding(agent_player)
        enc_opp_player = self.player_embedding(opp_player)
        enc_agent_cards = self.card_embedding(agent_cards)
        enc_opp_cards = self.card_embedding(opp_cards)
        definitions = None
        programs = None
        references = obs.references
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
            # Bucketed shapes encode the complete catalog in learning too: a
            # fixed recurrent layout instead of a per-batch referenced subset.
            if isinstance(obs, PreparedObservation):
                programs = obs.scope.catalog()
            elif torch.is_grad_enabled() and not self.bucketed_shapes:
                programs = self.semantic_cards.definitions(
                    self._referenced_definitions(obs, permanent_ids)
                )
            else:
                programs = self.semantic_cards.definitions()
            assert programs is not None
            definitions = self.name_candidates.public_definitions(programs)
            cards, known = self.semantic_cards(obs, definitions, references)
            enc_agent_cards = enc_agent_cards + cards[:, 0]
            enc_opp_cards = enc_opp_cards + cards[:, 1]
            enc_agent_player = enc_agent_player + known[:, 0, None]
            enc_opp_player = enc_opp_player + known[:, 1, None]
            permanent_definitions = definitions[references.permanents]
        else:
            assert self.card_definitions is not None
            cards = self.card_definitions(references.cards)
            enc_agent_cards = enc_agent_cards + cards[:, 0]
            enc_opp_cards = enc_opp_cards + cards[:, 1]
            permanent_definitions = self.card_definitions(references.permanents)
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
        validity_parts = [obs[key] for key in OBJECT_VALIDITY_KEYS]

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

        objects = self._join_permanent_links(obs, torch.cat(object_parts, dim=1))
        # Catalog fact and effect ids read the card encoder's rows.
        catalog = None
        if self.semantic_cards is not None and self.semantic_cards.records:
            assert definitions is not None
            catalog = RecordEmbeddings(self.semantic_cards, definitions)
        objects = self.facts(obs["facts"], objects, catalog)
        objects = self.shared_effects(
            obs["effects"],
            obs["effect_details"],
            obs["effect_members"],
            objects,
            catalog,
        )
        # History and current continuation both inform objects and focus-free actions.
        rows = obs["relations"]
        ids = rows[..., [5, 23]]
        relation_definitions = (
            self.semantic_cards.embed_definitions(ids, definitions)
            if self.semantic_cards is not None and definitions is not None
            else self._generic_definitions(ids)
        )
        encoded_relations = self.continuation.encode_rows(
            rows, objects, relation_definitions
        )
        shared_context = encoded_relations.sum(1) / rows[..., RELATION_VALID_INDEX].sum(
            1, keepdim=True
        ).clamp_min(1)
        frame_context = self.continuation.frame_context(rows, encoded_relations)
        if self.recent_events is not None:
            assert self.semantic_cards is not None and definitions is not None
            shared_context = (
                self.recent_events(
                    obs, objects, definitions, self.semantic_cards.definition_rows
                )
                + shared_context
            )
        objects = objects + shared_context.unsqueeze(1)

        # The visible object layout is static, so its ownership mask is a
        # registered buffer. Explicit belief rows extend it below.
        batch = objects.shape[0]
        is_agent = self._ownership(buckets).expand(batch, -1)
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
            objects,
            is_agent,
            validity,
            shared_context,
            frame_context,
            programs,
        )

    def _ownership(self, buckets: BucketSizes | None) -> torch.Tensor:
        """The static ownership mask [1,N] at this batch's object coordinates."""
        template = self._is_agent_template
        if buckets is None or not buckets.trims_objects:
            return template
        return torch.cat([template[:, part] for part in object_columns(buckets)], 1)

    def _join_permanent_links(
        self, obs: Dict[str, torch.Tensor], objects: torch.Tensor
    ) -> torch.Tensor:
        """Add each link target's embedding to its linked permanents and theirs to it.

        Links are object index + 1 (zero: none) into `objects` [B,N,D].
        """
        start = 2 + 2 * obs["agent_cards"].shape[1]
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
            permanents = objects[:, start : start + links.shape[1]]
            linked = (links > 0).unsqueeze(-1)
            targets = (
                (links.long() - 1).clamp_min(0).unsqueeze(-1).expand_as(permanents)
            )
            # Autocast leaves scatter's operands in their own dtypes; float32
            # runs convert nothing.
            delta = delta.scatter_add(
                1,
                targets,
                to_target(permanents).masked_fill(~linked, 0).to(delta.dtype),
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
        if self.recent_events is not None:
            events, valid = obs["events"], obs["events_valid"]
            ids.extend(events[..., column][valid == 1] for column in (2, 7))
        ids.append(obs["relations"][..., [5, 23]])
        if self.semantic_cards.records:
            # Catalog fact and effect sources read their definitions' rows.
            facts = obs["facts"]
            ids.append(facts[..., FACT_SOURCE_INDEX][facts[..., FACT_VALID_INDEX] == 1])
            effects = obs["effects"]
            ids.append(effects[..., 11][effects[..., 17] == 1])
        # Names share the same program table as objects. Only legal naming
        # rows widen the learner's support to the complete naming catalog.
        ids.append(obs["name_candidates"][..., 1][obs["actions_valid"].bool()])
        return self.semantic_cards.referenced_rows(ids)

    def _generic_definitions(self, ids: torch.Tensor) -> torch.Tensor:
        """Embed admitted card definition IDs of a world without a catalog."""
        assert self.card_definitions is not None
        return self.card_definitions(definition_rows(ids))

    def _belief_rows(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build explicit semantic belief rows and one global diagnostic row."""
        card_ids = obs["belief_card_def_ids"].long()
        owner_ids = obs["belief_owner_role_ids"].long()
        zone_ids = obs["belief_hidden_zone_ids"].long()
        probabilities = obs["belief_count_probabilities"].float()
        validity = obs["belief_validity"].float()
        globals_ = obs["belief_globals"].float()
        assert (
            self.belief_card_embedding is not None
            and self.belief_owner_embedding is not None
            and self.belief_zone_embedding is not None
            and self.belief_count_embedding is not None
            and self.belief_global_embedding is not None
        )
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
        obs: AdmittedObservation,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        if not obs.legal:
            raise ValueError("No valid actions available")
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs: AdmittedObservation) -> torch.Tensor:
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
