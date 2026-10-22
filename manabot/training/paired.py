"""Train two live policies against each other with independent learning state.

The collector freezes both policies for a complete-game batch. Each player's
optimizer sees only that player's own behavior rows. Callers own checkpoint
exports, stopping rules and experiment receipts, just as with update_iteration.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from manabot.sim.net_opponent import SeatRoutedCollector
from manabot.training.learner import Learner
from manabot.training.models import DampedPolicyLearning, Learning
from manabot.training.objectives import update_iteration
from manabot.training.selection import UpdateDiagnostics


@dataclass
class LearningPlayer:
    """One policy's optimizer, learning rule and minibatch random stream."""

    learner: Learner
    learning: Learning | DampedPolicyLearning
    rng: np.random.Generator
    transitions: int = 0


@dataclass
class PairedUpdate:
    first: UpdateDiagnostics
    second: UpdateDiagnostics
    environment_decisions: int
    games: int


class PairedLearners:
    """One collection boundary and two independent updates per iteration.

    Use an ``opponent_mode="learning"`` collector;
    its opponent_agent must be the second learner's model. EMA behavior is not
    implicit here: both models collect under their current raw weights.
    """

    def __init__(
        self,
        collector: SeatRoutedCollector,
        first: LearningPlayer,
        second: LearningPlayer,
    ) -> None:
        if (
            collector.opponent_mode != "learning"
            or collector.opponent_agent is not second.learner.agent
        ):
            raise ValueError("paired learners require their shared learning collector")
        if any(
            player.learner.minibatch_rows != collector.num_envs
            for player in (first, second)
        ):
            raise ValueError("damped minibatch width must match native stream count")
        first_parameters = {id(p) for p in first.learner.agent.parameters()}
        if first_parameters.intersection(
            id(p) for p in second.learner.agent.parameters()
        ):
            raise ValueError("paired models must have independent parameters")
        if first.learner.optimizer is second.learner.optimizer:
            raise ValueError("paired models must have independent optimizers")
        if first.learning.ema is not None or second.learning.ema is not None:
            raise ValueError("paired learners use raw behavior; EMA is not supported")
        self.first = first
        self.second = second
        self.collector: SeatRoutedCollector = collector
        self.iteration = 0

    def step(
        self,
        transitions: int,
        *,
        progress: float,
        deadline_monotonic: float | None = None,
        check: Callable[[], None] | None = None,
    ) -> PairedUpdate:
        """Collect fresh rows, then update both policies before the next game batch.

        ``transitions`` is a total-decision floor per native stream, shared by
        both players; an explicit games_per_stream overrides it. Counts report
        the actual complete games, including quota overshoot.
        """
        before_decisions = self.collector.stats.micro_steps
        before_games = self.collector.stats.games
        first_batch, second_batch = self.collector.collect_pair(
            self.first.learner.agent,
            transitions,
            deadline_monotonic=deadline_monotonic,
            check=check,
        )
        if check is not None:
            check()
        iteration = self.iteration + 1
        first = update_iteration(
            self.first.learner,
            first_batch,
            self.first.learning,
            progress,
            self.first.rng,
            iteration=iteration,
        )
        second = update_iteration(
            self.second.learner,
            second_batch,
            self.second.learning,
            progress,
            self.second.rng,
            iteration=iteration,
        )
        self.first.transitions += first_batch.actions.size
        self.second.transitions += second_batch.actions.size
        self.iteration = iteration
        return PairedUpdate(
            first,
            second,
            self.collector.stats.micro_steps - before_decisions,
            self.collector.stats.games - before_games,
        )
