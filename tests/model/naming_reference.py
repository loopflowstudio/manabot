"""Dense action reference frozen from 9db184b9 for numerical equivalence tests.

Only the action computation is frozen; shared object encoding uses the current
model so independent representation improvements do not obscure this comparison.
Never imported by product code.
"""

from typing import Dict

import torch

from manabot.env.observation import RELATION_VALID_INDEX
from manabot.model.actor_critic import ActorCritic, EmbeddedObservation
from manabot.model.inference import PreparedObservation


class DenseActionReference(ActorCritic):
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

        action_count = obs["actions"].shape[1]
        if self.name_candidates is not None:
            # Padding grows with the vocabulary, but ordinary policy computation
            # scales with the actual support. Retain sparse positions, not just
            # a count: permutation proofs may put padding between legal rows.
            used = obs["actions_valid"].bool().any(0).nonzero()
            action_count = int(used[-1].item()) + 1 if used.numel() else 1
        informed_actions = self._gather_informed_actions(
            obs, post_attention_objects, state, action_count=action_count
        )
        if self.policy_film is not None:
            informed_actions = self._condition_actions(
                informed_actions, post_attention_objects, state.validity
            )

        logits_before_mask = self.policy_head(informed_actions).squeeze(-1)
        if self.debug:
            # Save raw logits for offline diagnosis (scripts/diagnose_*).
            self.last_raw_logits = logits_before_mask.detach().cpu()
        logits = logits_before_mask.masked_fill(
            obs["actions_valid"][:, :action_count] == 0, -1e8
        )
        if action_count < obs["actions"].shape[1]:
            logits = torch.nn.functional.pad(
                logits, (0, obs["actions"].shape[1] - action_count), value=-1e8
            )
        return logits, self._value_from_objects(encoded, state.validity)

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
        state: EmbeddedObservation,
        *,
        action_count: int | None = None,
    ) -> torch.Tensor:
        action_rows = obs["actions"][:, :action_count]
        valid = obs["actions_valid"][:, :action_count]
        # The continuation selector is a join, never a numeric strategic feature.
        actions = self.action_embedding(action_rows[..., :-2])
        indexes = torch.where(valid.bool(), action_rows[..., -2], 0)
        if not ((indexes == indexes.floor()) & (indexes >= 0) & (indexes <= 64)).all():
            raise ValueError("invalid action continuation reference")
        if state.frame_context is not None:
            rows = obs["relations"]
            present = (
                (indexes.unsqueeze(-1) == rows[:, None, :, 1])
                & (rows[:, None, :, RELATION_VALID_INDEX] == 1)
            ).any(-1)
            if ((indexes != 0) & ~present).any():
                raise ValueError("action continuation reference has no frame")
            linked = state.frame_context.gather(
                1,
                (indexes.long() - 1)
                .clamp_min(0)
                .unsqueeze(-1)
                .expand(-1, -1, actions.shape[-1]),
            )
            actions = actions + linked.masked_fill((indexes == 0).unsqueeze(-1), 0)
        elif (indexes != 0).any():
            raise ValueError("trigger ordering requires continuation policy input")
        if self.name_candidates is not None:
            candidates = torch.where(
                valid.bool().unsqueeze(-1), obs["name_candidates"][:, :action_count], 0
            )
            actions = actions + self.name_candidates(candidates)
            ids = candidates[..., 1]
            if self.semantic_cards is not None and state.definitions is not None:
                actions = actions + self.semantic_cards.embed_definitions(
                    ids, state.definitions
                )
        if state.shared_context is not None:
            # Focus-free actions also receive the same context without attention.
            actions = actions + state.shared_context.unsqueeze(1)
        actions = actions * valid.unsqueeze(-1)
        actions_with_focus = self._add_focus(
            actions, post_attention_objects, obs["action_focus"][:, :action_count]
        )
        return self.action_layer(actions_with_focus)
