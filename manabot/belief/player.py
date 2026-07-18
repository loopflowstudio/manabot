"""Matched belief and compatible-prior manabot players."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import resource
import sys
import time
from typing import Any

import numpy as np

from manabot.belief.tracker import ActionLikelihood, BeliefTracker
from manabot.env import Env
from manabot.sim.flat_mc import (
    DEFAULT_MAX_PLAYOUT_STEPS,
    DEFAULT_ROLLOUTS_PER_WORLD,
    SearchStats,
)
from managym.decision import Command, DecisionFrame, SemanticTransition
from managym.possible_worlds import PossibleWorldSpace


class RangeSampling(str, Enum):
    BELIEF = "belief"
    COMPATIBLE_PRIOR = "compatible_prior"


@dataclass(slots=True)
class BeliefPlayerStats:
    search: SearchStats = field(default_factory=SearchStats)
    end_to_end_seconds: float = 0.0
    end_to_end_decision_seconds: list[float] = field(default_factory=list)
    update_seconds: list[float] = field(default_factory=list)
    command_seconds: list[float] = field(default_factory=list)
    sampled_worlds: int = 0
    unique_sampled_worlds: int = 0
    materialization_failures: int = 0
    commands_emitted: int = 0
    range_updates: int = 0
    action_updates: int = 0
    hidden_draws: int = 0
    known_exits: int = 0
    known_returns: int = 0
    likelihood_seconds: float = 0.0
    peak_range_bytes: int = 0
    peak_support_size: int = 0


class ExactRangePlayer:
    """Stateful canonical-world player with a matched prior sampling arm."""

    def __init__(
        self,
        sims: int,
        *,
        likelihood: ActionLikelihood,
        sampling: RangeSampling | str = RangeSampling.BELIEF,
        epsilon: float = 0.05,
        rollouts_per_world: int = DEFAULT_ROLLOUTS_PER_WORLD,
        max_steps: int = DEFAULT_MAX_PLAYOUT_STEPS,
        seed: int = 0,
    ) -> None:
        if sims < 1:
            raise ValueError("sims must be positive")
        self.sampling = RangeSampling(sampling)
        self.likelihood = likelihood
        self.epsilon = epsilon
        self.sims = sims
        self.rollouts = max(1, min(rollouts_per_world, sims))
        self.worlds = max(1, sims // self.rollouts)
        self.max_steps = max_steps
        self.seed = seed
        self.calls = 0
        self.seat: int | None = None
        self.tracker: BeliefTracker | None = None
        self.stats = BeliefPlayerStats()
        self.last_scores: np.ndarray | None = None
        self.last_command: Command | None = None
        self._pending_likelihood_root: Any | None = None
        self._decision_started: float | None = None
        self.completed_replays: list[dict[str, Any]] = []

    @property
    def search_stats(self) -> SearchStats:
        return self.stats.search

    def start_game(self, env: Env, seat: int) -> None:
        self.seat = seat
        self.tracker = BeliefTracker.from_engine(
            env._engine,
            viewer=seat,
            likelihood=self.likelihood,
            epsilon=self.epsilon,
        )
        self._pending_likelihood_root = None
        self._decision_started = None

    def finish_game(self, *, game_index: int, seed: int) -> None:
        if self.tracker is None:
            raise RuntimeError("start_game must run before finish_game")
        receipt = self.tracker.replay_receipt()
        receipt["game_index"] = game_index
        receipt["seed"] = seed
        self.completed_replays.append(receipt)

    def prepare_step(self, env: Env, acting: int, action: int) -> None:
        del action
        if self.seat is None:
            raise RuntimeError("start_game must run before prepare_step")
        self._pending_likelihood_root = (
            env._engine.clone_env() if acting != self.seat else None
        )

    def observe_step(
        self, env: Env, acting: int, transition: SemanticTransition
    ) -> None:
        observe_started = time.perf_counter()
        if self.tracker is None or self.seat is None:
            raise RuntimeError("start_game must run before observe_step")
        tracker_before = (
            self.tracker.stats.action_updates,
            self.tracker.stats.hidden_draws,
            self.tracker.stats.known_exits,
            self.tracker.stats.known_returns,
            self.tracker.stats.likelihood_seconds,
        )
        self.tracker.observe(
            env._engine,
            acting=acting,
            transition=transition,
            likelihood_root=self._pending_likelihood_root,
        )
        tracker_after = self.tracker.stats
        self.stats.range_updates += 1
        self.stats.action_updates += tracker_after.action_updates - tracker_before[0]
        self.stats.hidden_draws += tracker_after.hidden_draws - tracker_before[1]
        self.stats.known_exits += tracker_after.known_exits - tracker_before[2]
        self.stats.known_returns += tracker_after.known_returns - tracker_before[3]
        self.stats.likelihood_seconds += (
            tracker_after.likelihood_seconds - tracker_before[4]
        )
        self.stats.peak_range_bytes = max(
            self.stats.peak_range_bytes, tracker_after.peak_range_bytes
        )
        self.stats.peak_support_size = max(
            self.stats.peak_support_size, tracker_after.peak_support_size
        )
        self._pending_likelihood_root = None
        self.stats.update_seconds.append(time.perf_counter() - observe_started)

    def act(self, env: Env, obs: dict[str, np.ndarray]) -> int:
        del obs
        if self.tracker is None or self.seat is None:
            raise RuntimeError("start_game must run before act")
        self._decision_started = time.perf_counter()
        self.calls += 1
        call_seed = (self.seed * 1_000_003 + self.calls) & 0xFFFFFFFFFFFFFFFF
        belief = (
            self.tracker.posterior
            if self.sampling is RangeSampling.BELIEF
            else self.tracker.prior
        )
        current_space = PossibleWorldSpace.from_engine(env._engine, self.seat)
        if current_space.identity != belief.space.identity:
            raise RuntimeError("player BeliefState is stale for the current decision")
        sampled_indexes = belief.sample_indexes(self.worlds, seed=call_seed)
        seed_rng = np.random.default_rng(call_seed ^ 0x9E3779B97F4A7C15)
        world_seeds = [
            int(value)
            for value in seed_rng.integers(
                0, np.iinfo(np.uint64).max, size=self.worlds, dtype=np.uint64
            )
        ]
        search_started = time.perf_counter()
        try:
            scores, simulations, cap_hits = current_space.flat_mc_scores(
                sampled_indexes,
                world_seeds,
                rollouts=self.rollouts,
                max_steps=self.max_steps,
            )
        except Exception:
            self.stats.materialization_failures += 1
            raise
        search_elapsed = time.perf_counter() - search_started
        self.stats.search.decisions += 1
        self.stats.search.seconds += search_elapsed
        self.stats.search.simulations += simulations
        self.stats.search.cap_hits += cap_hits
        self.stats.search.decision_seconds.append(search_elapsed)
        self.stats.sampled_worlds += len(sampled_indexes)
        self.stats.unique_sampled_worlds += len(set(sampled_indexes))
        self.last_scores = np.asarray(scores, dtype=np.float32)
        return int(np.argmax(scores))

    def command_for_action(
        self, engine: Any, action: int, *, command_id: str
    ) -> Command:
        command_started = time.perf_counter()
        frame = DecisionFrame.from_json(engine.semantic_decision_frame_json())
        if action < 0 or action >= len(frame.offers):
            raise RuntimeError("action is outside the canonical DecisionFrame")
        command = Command(
            command_id=command_id,
            expected_revision=frame.revision,
            offer_id=int(frame.offers[action]["id"]),
        )
        self.last_command = command
        self.stats.commands_emitted += 1
        self.stats.command_seconds.append(time.perf_counter() - command_started)
        if self._decision_started is not None:
            elapsed = time.perf_counter() - self._decision_started
            self.stats.end_to_end_seconds += elapsed
            self.stats.end_to_end_decision_seconds.append(elapsed)
            self._decision_started = None
        return command

    def evidence_stats(self) -> dict[str, Any]:
        decision_seconds = np.asarray(
            self.stats.search.decision_seconds, dtype=np.float64
        )
        end_to_end_seconds = np.asarray(
            self.stats.end_to_end_decision_seconds, dtype=np.float64
        )
        update_seconds = np.asarray(self.stats.update_seconds, dtype=np.float64)
        command_seconds = np.asarray(self.stats.command_seconds, dtype=np.float64)
        diagnostics = self.tracker.diagnostics() if self.tracker is not None else {}
        return {
            "sampling": self.sampling.value,
            "decisions": self.stats.search.decisions,
            "simulations": self.stats.search.simulations,
            "cap_hits": self.stats.search.cap_hits,
            "search_seconds": self.stats.search.seconds,
            "end_to_end_seconds": self.stats.end_to_end_seconds,
            "p50_search_latency_ms": _quantile_ms(decision_seconds, 0.50),
            "p95_search_latency_ms": _quantile_ms(decision_seconds, 0.95),
            "p50_end_to_end_latency_ms": _quantile_ms(end_to_end_seconds, 0.50),
            "p95_end_to_end_latency_ms": _quantile_ms(end_to_end_seconds, 0.95),
            "p50_belief_update_latency_ms": _quantile_ms(update_seconds, 0.50),
            "p95_belief_update_latency_ms": _quantile_ms(update_seconds, 0.95),
            "p50_command_latency_ms": _quantile_ms(command_seconds, 0.50),
            "p95_command_latency_ms": _quantile_ms(command_seconds, 0.95),
            "rollouts_per_second": (
                self.stats.search.simulations / self.stats.search.seconds
                if self.stats.search.seconds
                else None
            ),
            "sampled_worlds": self.stats.sampled_worlds,
            "unique_sampled_worlds": self.stats.unique_sampled_worlds,
            "sample_collision_rate": (
                1.0 - self.stats.unique_sampled_worlds / self.stats.sampled_worlds
                if self.stats.sampled_worlds
                else 0.0
            ),
            "materialization_failures": self.stats.materialization_failures,
            "commands_emitted": self.stats.commands_emitted,
            "range_updates": self.stats.range_updates,
            "action_updates": self.stats.action_updates,
            "hidden_draws": self.stats.hidden_draws,
            "known_exits": self.stats.known_exits,
            "known_returns": self.stats.known_returns,
            "likelihood_seconds": self.stats.likelihood_seconds,
            "peak_range_bytes": self.stats.peak_range_bytes,
            "peak_support_size": self.stats.peak_support_size,
            "peak_rss_bytes": _peak_rss_bytes(),
            "replay_games": len(self.completed_replays),
            "current_belief": diagnostics,
        }

    def replay_receipts(self) -> list[dict[str, Any]]:
        return list(self.completed_replays)


class UniformRangePlayer(ExactRangePlayer):
    def __init__(
        self, sims: int, *, likelihood: ActionLikelihood, **kwargs: Any
    ) -> None:
        super().__init__(
            sims,
            likelihood=likelihood,
            sampling=RangeSampling.COMPATIBLE_PRIOR,
            **kwargs,
        )


def _quantile_ms(values: np.ndarray, quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) * 1000 if values.size else None


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024
