"""Native bot policies and the outputs retained at choice time.

These policies need only the engine. Training extensions provide checkpoint
policies through the same callable and optional ``last_decision`` contract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import random
from typing import Literal

import managym
from managym.enums import ActionEnum

BotPolicy = Callable[[managym.Env, managym.Observation], int]
DEFAULT_SEARCH_SIMS = 64
DEFAULT_ROLLOUTS_PER_WORLD = 4
DEFAULT_MAX_PLAYOUT_STEPS = 2000


@dataclass(frozen=True)
class PolicyFactor:
    chosen: int
    probabilities: list[float]
    value: float


@dataclass(frozen=True)
class BeliefSummary:
    model: str
    digest: str
    condition_mass: float
    sampled_worlds: int


@dataclass(frozen=True)
class BotDecision:
    action: int
    probabilities: list[float] = field(default_factory=list)
    value: float | None = None
    value_semantic: str | None = None
    search_scores: list[float | None] = field(default_factory=list)
    unavailable: str | None = None
    factors: list[PolicyFactor] = field(default_factory=list)
    belief: BeliefSummary | None = None


class FlatSearchPolicy:
    """Flat determinized Monte Carlo through native Env.flat_mc_scores.

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
        self.last_scores: list[float] = []

    def __call__(self, env: managym.Env, obs: managym.Observation) -> int:
        self.last_scores = []
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
        self.last_scores = [float(score) for score in scores]
        best_index = 0
        best_score = float("-inf")
        for index, score in enumerate(scores):
            if float(score) > best_score:
                best_index = index
                best_score = float(score)
        return best_index


def passive_policy(env: managym.Env, obs: managym.Observation) -> int:
    """Pass priority when possible, otherwise pick the first available action."""
    del env
    actions = obs.action_space.actions
    if not actions:
        raise ValueError("No available actions for bot policy.")

    pass_priority = int(ActionEnum.PRIORITY_PASS_PRIORITY)
    for index, action in enumerate(actions):
        if int(action.action_type) == pass_priority:
            return index

    return 0


class RandomPolicy:
    """Select a uniformly random legal action."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def __call__(self, env: managym.Env, obs: managym.Observation) -> int:
        del env
        actions = obs.action_space.actions
        if not actions:
            raise ValueError("No available actions for bot policy.")
        return self._rng.randrange(len(actions))


def choose_bot(
    policy: BotPolicy, env: managym.Env, obs: managym.Observation
) -> BotDecision:
    """Choose once and retain only outputs that actually fed this decision."""
    action = policy(env, obs)
    count = len(obs.action_space.actions)
    if not 0 <= action < count:
        raise ValueError("Bot selected an action outside the authority offers")
    retained = getattr(policy, "last_decision", None)
    if retained is not None:
        if not isinstance(retained, BotDecision) or retained.action != action:
            raise ValueError("Bot outputs do not describe the selected action")
        return retained
    probabilities = (
        [1.0 / count] * count
        if isinstance(policy, RandomPolicy)
        else [float(index == action) for index in range(count)]
    )
    return BotDecision(
        action=action,
        probabilities=probabilities,
        search_scores=policy.last_scores
        if isinstance(policy, FlatSearchPolicy)
        else [],
        unavailable="This bot policy has no value head.",
    )


class BotArtifactError(ValueError):
    def __init__(
        self,
        reason: Literal["advisor_artifact_unavailable", "advisor_artifact_mismatch"],
    ) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class VerifiedArena:
    """Verified native replay root and retained policy outputs in record order."""

    env: managym.Env
    observation: managym.Observation
    configs: Sequence[managym.PlayerConfig]
    decisions: list[BotDecision]
