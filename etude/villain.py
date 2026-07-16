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
  - "search":     flat determinized Monte Carlo via managym.Env.flat_mc_scores
                  (strength dial: sims per legal action, default 64)
  - "checkpoint": a trained Agent loaded from a .pt training checkpoint,
                  acting stochastically or greedily (deterministic flag)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable

# Local imports
import managym

from .enums import ActionEnum

if TYPE_CHECKING:
    from .trace import GameConfig

VillainPolicy = Callable[[managym.Env, managym.Observation], int]

DEFAULT_SEARCH_SIMS = 64
DEFAULT_ROLLOUTS_PER_WORLD = 4
DEFAULT_MAX_PLAYOUT_STEPS = 2000
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


class SearchVillain:
    """Flat determinized Monte Carlo (see manabot/sim/flat_mc.py).

    ``sims`` is the strength dial N: simulations per legal action, split as
    W worlds x R rollouts. Determinization never peeks at the hero's hand:
    the engine replaces the *opponent-of-the-searcher*'s hand with a uniform
    redraw from their unseen pool, so search plays by the same hidden-info
    rules as a human.
    """

    def __init__(
        self,
        sims: int = DEFAULT_SEARCH_SIMS,
        *,
        rollouts_per_world: int = DEFAULT_ROLLOUTS_PER_WORLD,
        max_steps: int = DEFAULT_MAX_PLAYOUT_STEPS,
        seed: int = 0,
    ):
        if sims < 1:
            raise ValueError("search sims must be >= 1")
        self.sims = sims
        self.rollouts = max(1, min(rollouts_per_world, sims))
        self.worlds = max(1, sims // self.rollouts)
        self.max_steps = max_steps
        self._seed = seed
        self._calls = 0

    def __call__(self, env: managym.Env, obs: managym.Observation) -> int:
        actions = obs.action_space.actions
        if not actions:
            raise ValueError("No available actions for villain policy.")
        if len(actions) == 1:
            return 0

        self._calls += 1
        call_seed = (self._seed * 1_000_003 + self._calls) & 0xFFFFFFFFFFFFFFFF
        scores, _, _ = env.flat_mc_scores(
            self.worlds,
            self.rollouts,
            call_seed,
            self.max_steps,
        )
        best_index = 0
        best_score = float("-inf")
        for index, score in enumerate(scores):
            if float(score) > best_score:
                best_index = index
                best_score = float(score)
        return best_index


class CheckpointVillain:
    """A trained Agent loaded from a training checkpoint (.pt).

    Torch and the model stack are imported lazily so that search/random games
    never pay the import cost.
    """

    def __init__(self, path: str, *, deterministic: bool = False):
        from manabot.sim.flat_mc import load_checkpoint_agent
        from manabot.verify.util import _select_agent_action

        self._select_action = _select_agent_action
        self.agent, self.obs_space = load_checkpoint_agent(path)
        self.deterministic = deterministic

    def __call__(self, env: managym.Env, obs: managym.Observation) -> int:
        del env
        encoded = self.obs_space.encode(obs)
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
        return SearchVillain(sims=sims, seed=seed)
    if villain_type == "checkpoint":
        if not config.villain_checkpoint:
            raise ValueError("checkpoint villain requires villain_checkpoint path.")
        return CheckpointVillain(
            config.villain_checkpoint,
            deterministic=config.villain_deterministic,
        )
    raise ValueError(f"Unsupported villain_type: {villain_type}")
