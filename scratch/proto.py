"""Prototype variants of the policy readout, monkeypatched onto Agent."""
import torch, torch.nn as nn
from manabot.model import agent as A

VARIANT = None

def install(variant: str) -> None:
    global VARIANT
    VARIANT = variant
    orig_init = A.Agent.__init__
    def __init__(self, space, hypers):
        orig_init(self, space, hypers)
        d = hypers.hidden_dim
        if variant == "bilinear":
            self.policy_context = A.layer_init(nn.Linear(d, d), gain=1.0)
        elif variant == "film":
            self.policy_context = A.layer_init(nn.Linear(d, d), gain=1.0)
        elif variant in ("film0", "bilinear0"):
            self.policy_context = nn.Linear(d, d)
            nn.init.zeros_(self.policy_context.weight); nn.init.zeros_(self.policy_context.bias)
    A.Agent.__init__ = __init__

    def forward_distribution(self, obs):
        objects, is_agent, validity, history = self._gather_object_embeddings(obs)
        encoded = self._attend_objects(objects, is_agent, validity)
        post = encoded[:, : validity.shape[1]]
        informed = self._gather_informed_actions(obs, post, history)
        v = validity.unsqueeze(-1)
        context = (post * v).sum(1) / v.sum(1).clamp_min(1)
        if variant in ("bilinear", "bilinear0"):
            q = self.policy_context(context).unsqueeze(1)
            logits = self.policy_head(informed).squeeze(-1) + (informed * q).sum(-1) / informed.shape[-1] ** 0.5
        elif variant in ("film", "film0"):
            g = self.policy_context(context).unsqueeze(1)
            logits = self.policy_head(informed * (1 + g)).squeeze(-1)
        logits = logits.masked_fill(obs["actions_valid"] == 0, -1e8)
        return logits, self._value_from_objects(encoded, validity)
    A.Agent.forward_distribution = forward_distribution
