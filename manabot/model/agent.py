"""
agent.py
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# Local imports
from manabot.env import ObservationSpace
from manabot.infra import AgentHypers
from manabot.infra.log import getLogger


class Agent(nn.Module):
    """
    Agent that:
      1. Encodes game objects with typed embeddings.
      2. Uses an attention mechanism to allow information exchange.
      3. Augments action representations with focus object information.
      4. Computes policy logits and a value estimate.

    Debug logging uses explicit category tags to help diagnose issues,
    such as why the model appears to always select the default action.
    """

    def __init__(self, observation_space: ObservationSpace, hypers: AgentHypers):
        super().__init__()
        self.observation_space = observation_space
        self.hypers = hypers
        self.logger = getLogger(__name__)

        # Extract dimensions from the observation encoder.
        enc = observation_space.encoder
        player_dim = enc.player_dim
        card_dim = enc.card_dim
        perm_dim = (
            enc.permanent_dim
        )  # fixed: use permanent_dim as defined in ObservationEncoder
        action_dim = enc.action_dim  # e.g., 6 (5 action types + validity flag)
        self.max_focus_objects = enc.max_focus_objects
        embed_dim = hypers.hidden_dim

        # Set up typed object embeddings.
        self.player_embedding = ProjectionLayer(player_dim, embed_dim)
        self.card_embedding = ProjectionLayer(card_dim, embed_dim)
        self.perm_embedding = ProjectionLayer(perm_dim, embed_dim)
        # Use only the first (action_dim - 1) features for actions.
        self.action_embedding = ProjectionLayer(action_dim - 1, embed_dim)

        self.logger.info(f"Player embedding: ({player_dim} -> {embed_dim})")
        self.logger.info(f"Card embedding: ({card_dim} -> {embed_dim})")
        self.logger.info(f"Permanent embedding: ({perm_dim} -> {embed_dim})")
        self.logger.info(f"Action embedding: ({action_dim - 1} -> {embed_dim})")

        # Global game state processor via attention.
        if self.hypers.attention_on:
            num_heads = self.hypers.num_attention_heads
            self.attention = GameObjectAttention(embed_dim, num_heads=num_heads)
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
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.ReLU(),
            MeanPoolingLayer(dim=1),
            layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, 1)),
        )
        self.logger.info(f"Policy head: ({embed_dim} -> 1)")
        self.logger.info(f"Value head: ({embed_dim} -> 1)")

        # Belief-conditioning side-input channel (INT-14). When
        # max_conditions > 0, one neutral object row is appended to the
        # attention sequence, built from a per-row condition_index and
        # condition_weight provided as a side input by a conditional shard.
        # No belief head is added; the policy and scalar value heads above
        # are unchanged. index 0 is reserved for the neutral True/uniform
        # condition used when the obs dict omits the condition keys (arena
        # inference), so the uninformative prior is the inference default.
        self.max_conditions = int(hypers.max_conditions)
        if self.max_conditions > 0:
            self.condition_index_embedding = nn.Embedding(
                self.max_conditions, embed_dim
            )
            self.condition_embedding = ProjectionLayer(embed_dim + 1, embed_dim)
            self.logger.info(
                f"Condition channel: on (max_conditions={self.max_conditions}); "
                "one neutral object row, no belief head"
            )
        else:
            self.condition_index_embedding = None
            self.condition_embedding = None

        # Static ownership mask over the concatenated object sequence
        # (agent player, opponent player, agent cards, opponent cards,
        # agent permanents, opponent permanents). When the condition channel
        # is on, one neutral (non-agent) slot is appended so the perspective
        # sign-flip in GameObjectAttention does not mis-attribute the
        # condition tag as agent-owned.
        is_agent_parts = [
            torch.ones(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            torch.ones(enc.cards_per_player, dtype=torch.bool),
            torch.zeros(enc.cards_per_player, dtype=torch.bool),
            torch.ones(enc.perms_per_player, dtype=torch.bool),
            torch.zeros(enc.perms_per_player, dtype=torch.bool),
        ]
        if self.max_conditions > 0:
            is_agent_parts.append(torch.zeros(1, dtype=torch.bool))
        is_agent_template = torch.cat(is_agent_parts).unsqueeze(0)
        self.register_buffer("_is_agent_template", is_agent_template)

        # When True, forward() stashes raw pre-mask logits on
        # self.last_raw_logits for offline diagnosis. Off by default: the
        # detach/cpu copy is measurable in the inference hot path.
        self.debug = False
        self.last_raw_logits: Optional[torch.Tensor] = None

    def forward(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        objects, is_agent, validity = self._gather_object_embeddings(obs)

        key_padding_mask = validity == 0

        if self.hypers.attention_on:
            post_attention_objects = self.attention(
                objects, is_agent, key_padding_mask=key_padding_mask
            )
        else:
            post_attention_objects = objects

        informed_actions = self._gather_informed_actions(obs, post_attention_objects)

        logits_before_mask = self.policy_head(informed_actions).squeeze(-1)
        if self.debug:
            # Save raw logits for offline diagnosis (scripts/diagnose_*).
            self.last_raw_logits = logits_before_mask.detach().cpu()
        logits = logits_before_mask.masked_fill(obs["actions_valid"] == 0, -1e8)
        value = self.value_head(post_attention_objects).squeeze(-1)
        return logits, value

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
        self, obs: Dict[str, torch.Tensor], post_attention_objects: torch.Tensor
    ) -> torch.Tensor:
        actions = self.action_embedding(obs["actions"][..., :-1])
        valid_mask = obs["actions_valid"].unsqueeze(-1)
        actions = actions * valid_mask
        actions_with_focus = self._add_focus(
            actions, post_attention_objects, obs["action_focus"]
        )
        return self.action_layer(actions_with_focus)

    def _gather_object_embeddings(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_agent_player = self.player_embedding(obs["agent_player"])
        enc_opp_player = self.player_embedding(obs["opponent_player"])
        enc_agent_cards = self.card_embedding(obs["agent_cards"])
        enc_opp_cards = self.card_embedding(obs["opponent_cards"])
        enc_agent_perms = self.perm_embedding(obs["agent_permanents"])
        enc_opp_perms = self.perm_embedding(obs["opponent_permanents"])
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

        if self.max_conditions > 0:
            object_parts.append(self._condition_row(obs))
            batch = object_parts[0].shape[0]
            validity_parts.append(torch.ones((batch, 1), device=object_parts[0].device))

        objects = torch.cat(object_parts, dim=1)

        # The object layout is static, so the ownership mask is a registered
        # buffer expanded to the batch instead of six per-call allocations.
        batch = objects.shape[0]
        is_agent = self._is_agent_template.expand(batch, -1)

        validity = torch.cat(validity_parts, dim=1)
        return objects, is_agent, validity

    def _condition_row(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build the one-row belief-conditioning side input.

        ``condition_index`` (long, in [0, max_conditions)) and
        ``condition_weight`` (float) are provided per row by a conditional
        shard. When the obs dict omits them (arena inference via the standard
        ObservationSpace), the neutral True/uniform condition (index 0,
        weight 1.0) is used — the uninformative prior that the ~0 strength
        prediction is built on.
        """

        if self.condition_index_embedding is None or self.condition_embedding is None:
            raise RuntimeError("condition channel is not enabled on this Agent")
        index_embedding = self.condition_index_embedding
        condition_embedding = self.condition_embedding
        device = obs["agent_player"].device
        batch = obs["agent_player"].shape[0]
        if "condition_index" in obs and "condition_weight" in obs:
            index = obs["condition_index"].long().to(device)
            weight = obs["condition_weight"].float().to(device).unsqueeze(-1)
        else:
            index = torch.zeros((batch,), dtype=torch.long, device=device)
            weight = torch.ones((batch, 1), dtype=torch.float32, device=device)
        index_embed = index_embedding(index)
        row = condition_embedding(torch.cat([index_embed, weight], dim=-1))
        return row.unsqueeze(1)

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
class MaxPoolingLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = torch.max(x, dim=self.dim)
        return pooled


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

    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.logger = getLogger(__name__).getChild("attention")
        self.logger.info(f"Creating perspective vector of size {embedding_dim}")
        self.perspective = nn.Parameter(torch.randn(embedding_dim) / embedding_dim**0.5)
        self.mha = nn.MultiheadAttention(
            embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(embedding_dim, num_heads * embedding_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(num_heads * embedding_dim, embedding_dim)),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        objects: torch.Tensor,
        is_agent: torch.Tensor,
        key_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        perspective_scale = torch.where(is_agent.unsqueeze(-1), 1.0, -1.0)
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
