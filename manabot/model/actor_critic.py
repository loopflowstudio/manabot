"""
Shared viewer-observation encoder, legal-action policy and scalar or WDL critic.

ActorCritic.forward preserves the serving policy/value interface. Training uses
forward_distribution when it needs loss/draw/win logits rather than their mean.
"""

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

# Local imports
from manabot.env.observation import (
    PERMANENT_DEFINITION_INDEX,
    RELATION_VALID_INDEX,
    ObservationSpace,
)
from manabot.infra.config import ModelConfig
from manabot.infra.log import getLogger
from manabot.model.compound import CompoundDecoder, CompoundOutput
from manabot.model.continuation import ContinuationEncoder
from manabot.model.inference import PreparedObservation
from manabot.model.recent_events import RecentEventEncoder
from manabot.model.semantic_cards import admit_definition_ids
from manabot.sim.action.structured import RaggedOfferBatch
import managym


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
        # means and compound prefix critics have architectural signed units.
        self.value_semantic: Literal["signed_outcome", "win_logit"] | None = (
            "signed_outcome"
            if config.value_kind == "categorical_wdl" or config.compound_decisions
            else None
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
        # Definition IDs are categorical joins, never numeric permanent features.
        perm_dim = enc.permanent_dim - 1
        action_dim = enc.action_dim
        self.max_focus_objects = enc.max_focus_objects
        embed_dim = config.hidden_dim
        self.compound_decoder = (
            CompoundDecoder(embed_dim) if config.compound_decisions else None
        )

        self.semantic_cards = None
        self.card_definitions: nn.Embedding | None = None
        if config.world is not None:
            from .semantic_cards import SemanticCardEncoder

            self.semantic_cards = SemanticCardEncoder(config.world, embed_dim)
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

        self.continuation = (
            ContinuationEncoder(embed_dim, enc.relation_dim)
            if enc.config.policy_input_version >= 2
            else None
        )

        # When True, forward() stashes raw pre-mask logits on
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
        if self.compound_decoder is not None:
            raise ValueError(
                "compound checkpoint requires authoritative offers; use ActorCritic.compound"
            )
        objects, is_agent, validity, shared_context, frame_context = (
            self._gather_object_embeddings(obs)
        )

        encoded = self._attend_objects(
            objects,
            is_agent,
            validity,
            objects_checked=isinstance(obs, PreparedObservation)
            and obs.objects_checked,
        )
        # The optional trailing token is a critic input, never an action focus.
        post_attention_objects = encoded[:, : validity.shape[1]]

        informed_actions = self._gather_informed_actions(
            obs, post_attention_objects, shared_context, frame_context
        )
        if self.policy_film is not None:
            informed_actions = self._condition_actions(
                informed_actions, post_attention_objects, validity
            )

        logits_before_mask = self.policy_head(informed_actions).squeeze(-1)
        if self.debug:
            # Save raw logits for offline diagnosis (scripts/diagnose_*).
            self.last_raw_logits = logits_before_mask.detach().cpu()
        logits = logits_before_mask.masked_fill(obs["actions_valid"] == 0, -1e8)
        return logits, self._value_from_objects(encoded, validity)

    def _condition_actions(
        self, actions: torch.Tensor, objects: torch.Tensor, validity: torch.Tensor
    ) -> torch.Tensor:
        """Scale actions [B,A,D] by a state code from valid objects [B,N,D].

        The mean is invariant to object and offer order, so the policy's
        permutation contracts are unchanged.
        """
        assert self.policy_film is not None
        valid = validity.unsqueeze(-1)
        state = (objects * valid).sum(1) / valid.sum(1).clamp_min(1)
        return actions * (1 + self.policy_film(state).unsqueeze(1))

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

    def compound(
        self,
        obs: dict[str, torch.Tensor],
        batch: RaggedOfferBatch,
        *,
        tokens: tuple[int, ...] | None = None,
        prefix: tuple[int, ...] = (),
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> CompoundOutput:
        """One viewer-safe root, one ragged action; no intermediate observation."""
        if self.compound_decoder is None:
            raise ValueError("checkpoint has no compound policy")
        objects, ownership, valid, shared_context, frame_context = (
            self._gather_object_embeddings(obs)
        )
        if objects.shape[0] != 1:
            raise ValueError("compound decoding requires one root")
        if self.config.attention_on:
            objects = self.attention(objects, ownership, key_padding_mask=valid == 0)
        context = (objects * valid.unsqueeze(-1)).sum(1) / valid.sum(
            1, keepdim=True
        ).clamp_min(1)
        # Complete priority/fallback offers preserve the native action ordering.
        # Complete declarations/payments instead have one set-valued offer;
        # its first row must not inherit an unrelated microchoice embedding.
        features = None
        if not any(
            offer["verb"] in {"declare_attackers", "declare_blockers", "pay_waterbend"}
            for offer in batch.offers
        ):
            features = self._gather_informed_actions(
                obs, objects, shared_context, frame_context
            )[0, : len(batch.offers)]
        return self.compound_decoder(
            context[0],
            batch,
            offer_features=features,
            tokens=tokens,
            prefix=prefix,
            generator=generator,
            deterministic=deterministic,
        )

    def _add_focus(
        self,
        actions: torch.Tensor,
        post_attention_objects: torch.Tensor,
        action_focus: torch.Tensor,
    ) -> torch.Tensor:
        B, max_actions, embed_dim = actions.shape

        valid_mask = (action_focus != -1).unsqueeze(-1).float()
        action_focus = action_focus.clamp(min=0)

        focus_indices = (
            action_focus.unsqueeze(-1)
            .expand(B, max_actions, self.max_focus_objects, embed_dim)
            .long()
        )
        post_attention_focus_objects = post_attention_objects.unsqueeze(1).expand(
            -1, max_actions, -1, -1
        )
        focus_embeds = torch.gather(post_attention_focus_objects, 2, focus_indices)
        focus_embeds = focus_embeds * valid_mask
        focus_flat = focus_embeds.reshape(B, max_actions, -1)
        return torch.cat([actions, focus_flat], dim=-1)

    def _gather_informed_actions(
        self,
        obs: Dict[str, torch.Tensor],
        post_attention_objects: torch.Tensor,
        shared_context: torch.Tensor | None,
        frame_context: torch.Tensor | None,
    ) -> torch.Tensor:
        # The continuation selector is a join, never a numeric strategic feature.
        actions = self.action_embedding(obs["actions"][..., :-2])
        indexes = torch.where(obs["actions_valid"].bool(), obs["actions"][..., -2], 0)
        if not ((indexes == indexes.floor()) & (indexes >= 0) & (indexes <= 64)).all():
            raise ValueError("invalid action continuation reference")
        if frame_context is not None:
            rows = obs["relations"]
            present = (
                (indexes.unsqueeze(-1) == rows[:, None, :, 1])
                & (rows[:, None, :, RELATION_VALID_INDEX] == 1)
            ).any(-1)
            if ((indexes != 0) & ~present).any():
                raise ValueError("action continuation reference has no frame")
            linked = frame_context.gather(
                1,
                (indexes.long() - 1)
                .clamp_min(0)
                .unsqueeze(-1)
                .expand(-1, -1, actions.shape[-1]),
            )
            actions = actions + linked.masked_fill((indexes == 0).unsqueeze(-1), 0)
        elif (indexes != 0).any():
            raise ValueError("trigger ordering requires continuation policy input")
        if shared_context is not None:
            # Focus-free actions also receive the same context without attention.
            actions = actions + shared_context.unsqueeze(1)
        valid_mask = obs["actions_valid"].unsqueeze(-1)
        actions = actions * valid_mask
        actions_with_focus = self._add_focus(
            actions, post_attention_objects, obs["action_focus"]
        )
        return self.action_layer(actions_with_focus)

    def _gather_object_embeddings(
        self, obs: Dict[str, torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
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
        references = obs.references if isinstance(obs, PreparedObservation) else None
        permanent_ids = torch.stack(
            [
                obs[key][..., PERMANENT_DEFINITION_INDEX]
                for key in ("agent_permanents", "opponent_permanents")
            ],
            dim=1,
        )
        if self.semantic_cards is not None:
            definitions = (
                obs.scope.catalog()
                if isinstance(obs, PreparedObservation)
                else self.semantic_cards.definitions()
            )
            assert definitions is not None
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
            torch.cat(
                (
                    obs[key][..., :PERMANENT_DEFINITION_INDEX],
                    obs[key][..., PERMANENT_DEFINITION_INDEX + 1 :],
                ),
                dim=-1,
            )
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
        return objects, is_agent, validity, shared_context, frame_context

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
