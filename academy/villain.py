"""
villain.py
Pluggable villain policies for the GUI server.

A villain policy is a callable ``(env, obs) -> action_index`` where ``env`` is
the raw ``managym.Env`` the game is running in (search policies use it for
``flat_mc_scores``) and ``obs`` is the current ``managym.Observation`` from the
villain's perspective.

Supported villain types (see ``build_villain_policy``):
  - "passive":    pass priority whenever possible (debugging baseline)
  - "random":     uniform random over legal actions
  - "search":     manabot's FlatSearchPolicy, flat determinized Monte Carlo
                  (strength dial: sims per legal action, default 64)
  - "checkpoint": a trained ActorCritic loaded from a .pt training checkpoint,
                  acting stochastically or greedily (deterministic flag)
"""

from __future__ import annotations

from collections.abc import Sequence
import random
from typing import TYPE_CHECKING, Callable

from managym.enums import ActionEnum

from manabot.sim.flat_search import DEFAULT_SEARCH_SIMS, FlatSearchPolicy

# Local imports
import managym

from .bot_decision import BotDecision, PolicyFactor

if TYPE_CHECKING:
    from .trace import GameConfig

VillainPolicy = Callable[[managym.Env, managym.Observation], int]

MAX_SEARCH_SIMS = 4096


def passive_policy(env: managym.Env, obs: managym.Observation) -> int:
    """Pass priority when possible, otherwise pick the first available action."""
    del env
    actions = obs.action_space.actions
    if not actions:
        raise ValueError("No available actions for villain policy.")

    pass_priority = int(ActionEnum.PRIORITY_PASS_PRIORITY)
    for index, action in enumerate(actions):
        if int(action.action_type) == pass_priority:
            return index

    return 0


class RandomVillain:
    """Select a uniformly random legal action."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def __call__(self, env: managym.Env, obs: managym.Observation) -> int:
        del env
        actions = obs.action_space.actions
        if not actions:
            raise ValueError("No available actions for villain policy.")
        return self._rng.randrange(len(actions))


class CheckpointVillain:
    """A trained ActorCritic loaded from a training checkpoint (.pt).

    Torch and the model stack are imported lazily so that search/random games
    never pay the import cost.
    """

    def __init__(
        self,
        path: str,
        *,
        deterministic: bool = False,
        player_configs: Sequence[managym.PlayerConfig] | None = None,
    ) -> None:
        from manabot.model.world import validate_agent_setup

        # Keep the optional Torch/model stack behind checkpoint construction.
        from manabot.sim.compound import CompoundPolicy
        from manabot.sim.flat_mc import load_checkpoint_agent

        self.agent, self.obs_space = load_checkpoint_agent(path)
        if player_configs is not None:
            validate_agent_setup(self.agent, player_configs)
        self.deterministic = deterministic
        self.last_decision: BotDecision | None = None
        self.compound = (
            CompoundPolicy(self.agent, deterministic=deterministic)
            if self.agent.config.compound_decisions
            else None
        )

    def __call__(self, env: managym.Env, obs: managym.Observation) -> int:
        self.last_decision = None
        if self.compound is not None:
            action = self.compound.act(env, obs)
            decision = self.compound.last_decision
            assert decision is not None
            output = decision.output
            self.last_decision = BotDecision(
                action,
                value=float(output.values[0].item()),
                value_semantic="signed_outcome at compound root",
                factors=[
                    PolicyFactor(token, probabilities.tolist(), float(value.item()))
                    for token, probabilities, value in zip(
                        output.tokens, output.probabilities, output.values, strict=True
                    )
                ],
                unavailable="Compound policy: conditional root-sequence factors are shown below; native microchoice marginals are unavailable.",
            )
            return action
        # Policy weights score each action independently of the padded action
        # count. At inference preserve the entire rules offer list, including
        # decisions wider than the training batch's storage capacity.
        space = self.obs_space
        if len(obs.action_space.actions) > space.encoder.max_actions:
            from manabot.env import ObservationSpace

            space = ObservationSpace(
                space.encoder.config.model_copy(
                    update={
                        "max_actions": len(obs.action_space.actions),
                    }
                )
            )
        encoded = space.encode(obs)
        import torch

        from manabot.verify.util import _agent_device, _tensor_observation

        with torch.no_grad():
            logits, value = self.agent(
                _tensor_observation(encoded, device=_agent_device(self.agent))
            )
            distribution = torch.distributions.Categorical(logits=logits)
            action = int(
                (
                    logits.argmax(dim=-1)
                    if self.deterministic
                    else distribution.sample()
                ).item()
            )
            self.last_decision = BotDecision(
                action=action,
                probabilities=distribution.probs[
                    0, : len(obs.action_space.actions)
                ].tolist(),
                value=float(value.item()),
                value_semantic=self.agent.value_semantic or "undeclared critic units",
            )
        return action


def choose_bot(
    policy: VillainPolicy, env: managym.Env, obs: managym.Observation
) -> BotDecision:
    """Choose once and retain only outputs that actually fed this decision."""
    action = policy(env, obs)
    count = len(obs.action_space.actions)
    if not 0 <= action < count:
        raise ValueError("Bot selected an action outside the authority offers")
    if isinstance(policy, CheckpointVillain):
        assert policy.last_decision is not None
        return policy.last_decision
    probabilities = (
        [1.0 / count] * count
        if isinstance(policy, RandomVillain)
        else [float(index == action) for index in range(count)]
    )
    return BotDecision(
        action=action,
        probabilities=probabilities,
        search_scores=policy.last_scores
        if isinstance(policy, FlatSearchPolicy)
        else [],
        unavailable="This anchor has no value head.",
    )


def build_villain_policy(config: "GameConfig") -> VillainPolicy:
    """Build a villain policy from a parsed GameConfig."""
    seed = config.seed if config.seed is not None else 0
    villain_type = config.villain_type

    if villain_type == "passive":
        return passive_policy
    if villain_type == "random":
        return RandomVillain(seed=seed)
    if villain_type == "search":
        sims = config.villain_sims or DEFAULT_SEARCH_SIMS
        return FlatSearchPolicy(sims=sims, seed=seed)
    if villain_type == "checkpoint":
        if not config.villain_checkpoint:
            raise ValueError("checkpoint villain requires villain_checkpoint path.")
        return CheckpointVillain(
            config.villain_checkpoint,
            deterministic=config.villain_deterministic,
            player_configs=config.to_rust(),
        )
    raise ValueError(f"Unsupported villain_type: {villain_type}")
