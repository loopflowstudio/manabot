"""Dependency-light flat determinized Monte Carlo policy over a native Env.

``FlatSearchPolicy`` scores every legal action with
``managym.Env.flat_mc_scores`` and plays the best one. It imports neither
NumPy nor Torch, so the Academy play runtime can use it as its search
opponent; the arena and experiments use the same class to reproduce that
opponent exactly. ``manabot.sim.flat_mc`` holds the training-side players
built on the same engine search.
"""

from __future__ import annotations

from manabot.sim.search_runtime import DEFAULT_MAX_PLAYOUT_STEPS
import managym

DEFAULT_SEARCH_SIMS = 64
DEFAULT_ROLLOUTS_PER_WORLD = 4


class FlatSearchPolicy:
    """Flat determinized Monte Carlo (see manabot/sim/flat_mc.py).

    ``sims`` is the strength dial N: simulations per legal action, split as
    W worlds x R rollouts. Determinization never peeks at the opponent's hand:
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
    ) -> None:
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
            raise ValueError("No available actions for search policy.")
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
