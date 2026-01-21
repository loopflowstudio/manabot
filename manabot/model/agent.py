"""
agent.py
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from manabot.env import ObservationSpace
from manabot.infra.log import getLogger
from manabot.infra import AgentHypers
from manabot.env.observation import ActionEnum

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
        perm_dim = enc.permanent_dim  # fixed: use permanent_dim as defined in ObservationEncoder
        action_dim = enc.action_dim   # e.g., 6 (5 action types + validity flag)
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
            self.logger.info(f"Attention: {embed_dim} -> {embed_dim} with {num_heads} heads")
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
            layer_init(nn.Linear(embed_dim, 1))
        )
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(embed_dim, embed_dim)),
            MaxPoolingLayer(dim=1),
            layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim, 1))
        )
        self.logger.info(f"Policy head: ({embed_dim} -> 1)")
        self.logger.debug(f"[INIT] Policy head first layer weight norm: {self.policy_head[0].weight.norm(2).item():.4f}")
        self.logger.info(f"Value head: ({embed_dim} -> 1)")
        self.logger.debug(f"[INIT] Value head first layer weight norm: {self.value_head[0].weight.norm(2).item():.4f}")

    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        log = self.logger.getChild("forward")
        objects, is_agent, validity = self._gather_object_embeddings(obs)
        log.debug(f"[SHAPES] Objects shape: {objects.shape}")
        log.debug(f"[SHAPES] Is agent shape: {is_agent.shape}")
        log.debug(f"[SHAPES] Validity shape: {validity.shape}")

        key_padding_mask = (validity == 0)
        log.debug(f"[SHAPES] Key padding mask shape: {key_padding_mask.shape}")
        
        if self.hypers.attention_on:
            post_attention_objects = self.attention(objects, is_agent, key_padding_mask=key_padding_mask)
        else:
            post_attention_objects = objects
        log.debug(f"[SHAPES] Post attention objects shape: {post_attention_objects.shape}")

        informed_actions = self._gather_informed_actions(obs, post_attention_objects)
        log.debug(f"[SHAPES] Informed actions shape: {informed_actions.shape}")

        logits_before_mask = self.policy_head(informed_actions).squeeze(-1)
        # Save raw logits for later debugging if needed (e.g., for attack logging)
        self.last_raw_logits = logits_before_mask.detach().cpu()
        log.debug(f"[LOGITS] Raw logits: mean={logits_before_mask.mean().item():.4f}, std={logits_before_mask.std().item():.4f}")
        logits = logits_before_mask.masked_fill(obs["actions_valid"] == 0, -1e8)
        log.debug(f"[LOGITS] Masked logits: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
        value = self.value_head(post_attention_objects).squeeze(-1)
        log.debug(f"[SHAPES] Value output shape: {value.shape}")
        return logits, value
    
    def _add_focus(self, actions: torch.Tensor, post_attention_objects: torch.Tensor, action_focus: torch.Tensor) -> torch.Tensor:
        log = self.logger.getChild("add_focus")
        B, max_actions, embed_dim = actions.shape
        log.debug(f"[SHAPES] Batch: {B}, max_actions: {max_actions}, embed_dim: {embed_dim}")
        log.debug(f"[SHAPES] post_attention_objects shape: {post_attention_objects.shape}")
        log.debug(f"[SHAPES] action_focus shape: {action_focus.shape}")
        log.debug(f"[ATTACK DEBUG] Raw action_focus indices: {action_focus.tolist()}")

        valid_mask = (action_focus != -1).unsqueeze(-1).float()
        log.debug(f"[ATTACK DEBUG] Valid mask: {valid_mask.squeeze(-1).tolist()}")
        action_focus = action_focus.clone()
        action_focus[action_focus == -1] = 0
        
        focus_indices = action_focus.unsqueeze(-1).expand(B, max_actions, self.max_focus_objects, embed_dim).long()
        post_attention_focus_objects = post_attention_objects.unsqueeze(1).expand(-1, max_actions, -1, -1)
        focus_embeds = torch.gather(post_attention_focus_objects, 2, focus_indices)
        focus_embeds = focus_embeds * valid_mask
        focus_flat = focus_embeds.view(B, max_actions, -1)
        actions_with_focus = torch.cat([actions, focus_flat], dim=-1)
        log.debug(f"[SHAPES] Actions with focus shape: {actions_with_focus.shape}")
        return actions_with_focus
    
    def _gather_informed_actions(self, obs: Dict[str, torch.Tensor], post_attention_objects: torch.Tensor) -> torch.Tensor:
        log = self.logger.getChild("gather_informed_actions")
        actions = self.action_embedding(obs["actions"][..., :-1])
        valid_mask = obs["actions_valid"].unsqueeze(-1)
        actions = actions * valid_mask
        actions_with_focus = self._add_focus(actions, post_attention_objects, obs["action_focus"])
        log.debug(f"[SHAPES] Action embedding output shape: {actions.shape}")
        log.debug(f"[SHAPES] Actions with focus shape: {actions_with_focus.shape}")
        return self.action_layer(actions_with_focus)
    
    def _gather_object_embeddings(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log = self.logger.getChild("gather_object_embeddings")
        device = obs["agent_player"].device
        log.debug(f"[SHAPES] Device: {device}")
        log.debug(f"[SHAPES] Obs['agent_player'] shape: {obs['agent_player'].shape}")
        log.debug(f"[SHAPES] Obs['opponent_player'] shape: {obs['opponent_player'].shape}")
        log.debug(f"[SHAPES] Obs['agent_cards'] shape: {obs['agent_cards'].shape}")
        log.debug(f"[SHAPES] Obs['opponent_cards'] shape: {obs['opponent_cards'].shape}")
        
        enc_agent_player = self.player_embedding(obs["agent_player"])
        enc_opp_player = self.player_embedding(obs["opponent_player"])
        enc_agent_cards = self.card_embedding(obs["agent_cards"])
        enc_opp_cards = self.card_embedding(obs["opponent_cards"])
        enc_agent_perms = self.perm_embedding(obs["agent_permanents"])
        enc_opp_perms = self.perm_embedding(obs["opponent_permanents"])
        objects = torch.cat([
            enc_agent_player, enc_opp_player,
            enc_agent_cards, enc_opp_cards,
            enc_agent_perms, enc_opp_perms
        ], dim=1)
        
        agent_player_is_agent = torch.ones(enc_agent_player.shape[0], enc_agent_player.shape[1], dtype=torch.bool, device=device)
        opponent_player_is_agent = torch.zeros(enc_opp_player.shape[0], enc_opp_player.shape[1], dtype=torch.bool, device=device)
        agent_cards_is_agent = torch.ones(enc_agent_cards.shape[0], enc_agent_cards.shape[1], dtype=torch.bool, device=device)
        opp_cards_is_agent = torch.zeros(enc_opp_cards.shape[0], enc_opp_cards.shape[1], dtype=torch.bool, device=device)
        agent_perms_is_agent = torch.ones(enc_agent_perms.shape[0], enc_agent_perms.shape[1], dtype=torch.bool, device=device)
        opp_perms_is_agent = torch.zeros(enc_opp_perms.shape[0], enc_opp_perms.shape[1], dtype=torch.bool, device=device)
        is_agent = torch.cat([
            agent_player_is_agent,
            opponent_player_is_agent,
            agent_cards_is_agent,
            opp_cards_is_agent,
            agent_perms_is_agent,
            opp_perms_is_agent
        ], dim=1)
        
        validity = torch.cat([
            obs["agent_player_valid"], obs["opponent_player_valid"],
            obs["agent_cards_valid"], obs["opponent_cards_valid"],
            obs["agent_permanents_valid"], obs["opponent_permanents_valid"]
        ], dim=1)
        
        log.debug(f"[SHAPES] Combined objects shape: {objects.shape}")
        log.debug(f"[SHAPES] Combined is_agent shape: {is_agent.shape}")
        log.debug(f"[SHAPES] Combined validity shape: {validity.shape}")
        return objects, is_agent, validity
    
    def get_action_and_value(self, obs: Dict[str, torch.Tensor], action: Optional[torch.Tensor] = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log = self.logger.getChild("get_action_and_value")
        logits, value = self.forward(obs)
        if (obs["actions_valid"].sum(dim=-1) == 0).any():
            raise ValueError("No valid actions available")
        probs = torch.softmax(logits, dim=-1)
        self.logger.debug(f"Action probabilities: mean={probs.mean().item():.4f}, std={probs.std().item():.4f}")
        dist = torch.distributions.Categorical(probs=probs)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
        # For each sample, if the chosen action equals DECLARE_ATTACKER (attack), log extra details
        for i, act in enumerate(action):
            if act.item() == ActionEnum.DECLARE_ATTACKER:
                log.debug(f"[ATTACK DEBUG] Sample {i} attack selected. "
                          f"[ATTACK DEBUG] Raw logits: {self.last_raw_logits[i].tolist()}, "
                          f"[ATTACK DEBUG] Masked logits: {logits[i].detach().cpu().tolist()}, "
                          f"[ATTACK DEBUG] Valid actions: {obs['actions_valid'][i].detach().cpu().tolist()}")
        log.debug(f"[SHAPES] Selected action shape: {action.shape}")
        log.debug(f"[SHAPES] Log prob shape: {dist.log_prob(action).shape}")
        log.debug(f"[SHAPES] Entropy shape: {dist.entropy().shape}")
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
        log = getLogger(__name__).getChild("MaxPoolingLayer")
        log.debug(f"[SHAPES] Input to MaxPoolingLayer: {x.shape}")
        pooled, _ = torch.max(x, dim=self.dim)
        log.debug(f"[SHAPES] Output of MaxPoolingLayer: {pooled.shape}")
        return pooled

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            layer_init(nn.Linear(input_dim, output_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(output_dim, output_dim))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log = getLogger(__name__).getChild("ProjectionLayer")
        log.debug(f"[SHAPES] Input to ProjectionLayer: {x.shape}")
        output = self.projection(x)
        log.debug(f"[SHAPES] Output of ProjectionLayer: {output.shape}")
        return output

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
        self.mha = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(embedding_dim, num_heads * embedding_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(num_heads * embedding_dim, embedding_dim))
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, objects: torch.Tensor, is_agent: torch.Tensor, key_padding_mask: torch.BoolTensor) -> torch.Tensor:
        log = self.logger.getChild("forward")
        log.debug(f"[SHAPES] Objects: {objects.shape}")
        log.debug(f"[SHAPES] Is agent: {is_agent.shape}")
        log.debug(f"[SHAPES] Key padding mask: {key_padding_mask.shape}")
        
        perspective_scale = torch.where(is_agent.unsqueeze(-1), 1.0, -1.0)
        log.debug(f"[SHAPES] Perspective vector shape: {self.perspective.shape}")
        owned_objects = objects + perspective_scale * self.perspective
        log.debug(f"[SHAPES] Owned objects shape: {owned_objects.shape}")

        attn_out, _ = self.mha(owned_objects, owned_objects, owned_objects, key_padding_mask=key_padding_mask)
        log.debug(f"[SHAPES] Attention output shape: {attn_out.shape}")
        
        x = self.norm1(owned_objects + attn_out)
        log.debug(f"[SHAPES] After first normalization: {x.shape}")

        mlp_out = self.mlp(x)
        log.debug(f"[SHAPES] MLP output shape: {mlp_out.shape}")

        post_norm = self.norm2(x + mlp_out)
        log.debug(f"[SHAPES] After second normalization: {post_norm.shape}")

        mask = (~key_padding_mask).unsqueeze(-1).float()  # shape: [B, 302, 1]
        post_norm = post_norm * mask  # broadcasting over last dim works correctly

        # Use expanded mask for the assertion to ensure the masked positions are truly zero.
        if torch.any(key_padding_mask):
            if not torch.all(post_norm[mask.expand_as(post_norm) == 0] == 0):
                self.logger.error("Attention mask failure: nonzero values in masked positions")
        return post_norm


def layer_init(layer: nn.Module, gain: int = 1, bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, gain)
    torch.nn.init.constant_(layer.bias, bias_const)
    getLogger(__name__).debug(f"[INIT] Initialized layer {layer} with weight norm {layer.weight.norm(2).item():.4f}")
    return layer
