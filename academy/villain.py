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
  - "checkpoint": a trained Agent loaded from a .pt training checkpoint,
                  acting stochastically or greedily (deterministic flag)
"""

from __future__ import annotations

from collections.abc import Sequence
import random
from typing import TYPE_CHECKING, Callable

from manabot.sim.flat_search import DEFAULT_SEARCH_SIMS, FlatSearchPolicy

# Local imports
import managym
from managym.enums import ActionEnum

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

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def __call__(self, env: managym.Env, obs: managym.Observation) -> int:
        del env
        actions = obs.action_space.actions
        if not actions:
            raise ValueError("No available actions for villain policy.")
        return self._rng.randrange(len(actions))


class CheckpointVillain:
    """A trained Agent loaded from a training checkpoint (.pt).

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
        from manabot.verify.util import _select_agent_action

        self._select_action = _select_agent_action
        self.agent, self.obs_space = load_checkpoint_agent(path)
        if player_configs is not None:
            validate_agent_setup(self.agent, player_configs)
        self.deterministic = deterministic
        self.compound = (
            CompoundPolicy(self.agent, deterministic=deterministic)
            if self.agent.hypers.compound_decisions
            else None
        )

    def __call__(self, env: managym.Env, obs: managym.Observation) -> int:
        if self.compound is not None:
            return self.compound.act(env, obs)
        # Policy weights score each action independently of the padded action
        # count. At inference preserve the entire rules offer list, including
        # decisions wider than the training batch's storage capacity.
        space = self.obs_space
        if len(obs.action_space.actions) > space.encoder.max_actions:
            from manabot.env import ObservationSpace

            space = ObservationSpace(
                space.encoder.hypers.model_copy(
                    update={
                        "max_actions": len(obs.action_space.actions),
                    }
                )
            )
        encoded = space.encode(obs)
        return int(
            self._select_action(self.agent, encoded, deterministic=self.deterministic)
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
