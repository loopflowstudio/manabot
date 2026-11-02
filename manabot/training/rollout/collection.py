"""Seat-routed on-policy collection for shared RL learning rules.

``SeatRoutedCollector`` owns native game streams and frozen behavior scopes.
One-seat collection alternates learner seats by stream and pauses at an exact
learner-decision quota. This remains the control and the random/frozen-opponent
path. Default both-seat self-play collection keeps a separate trajectory for each seat,
collects at least the requested total decisions per native stream, then finishes
its current game. Complete trajectories are packed in one column, separated by
terminal flags: GAE cannot cross players or games, and no behavior sample or
pending action survives an update. Batch row counts may exceed the quota.

Self-play uses one live model for both seats. ``collect_pair`` instead returns
one batch per learning model, with seats balanced across streams; each model's
own policy, critic and distribution generated its rows. Frozen and random
opponent decisions remain excluded. Terminal reward is computed per seat from
``winner_index``; the native reward stream is ignored.

Collection never updates model weights; the regime executor owns learning.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
import time
from typing import Any, Literal, TypedDict

import numpy as np
import torch

from manabot.env.failures import CapturedVector
from manabot.env.match import Match, Reward
from manabot.env.observation import ACTION_TYPE_DIM, ActionEnum, ObservationSpace
from manabot.infra.profiler import Profiler
from manabot.model.actor_critic import ActorCritic
from manabot.model.inference import PolicyInference
from manabot.model.policy_distribution import (
    NumericalError,
    legal_mask,
    policy_logs,
    require,
)
from manabot.sim.rollout import (
    OBS_KEYS,
    BatchedSampler,
    RandomBatchController,
    _allocate_buffers,
)
from manabot.training.decks import UniformDecks, pair_match
from manabot.training.rollout.matchups import (
    Curriculum,
    elemental_roster,
    stream_matches,
)
from manabot.training.rollout.producers import ShardedVectorEnv
from manabot.training.rollout.self_play import SelfPlayCounter
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorLimit,
    GameErrorPolicy,
    GameErrorState,
    GameOutcome,
)
from managym import WORLD_SETUP, Deck

OPPONENT_MODES = ("random", "frozen", "self", "learning")

# -----------------------------------------------------------------------------
# Seat-routed rollout collection
# -----------------------------------------------------------------------------


@dataclass
class _Pending:
    """A decision awaiting its reward at the same seat's next turn or game end."""

    obs: dict[str, np.ndarray]
    action: int
    logprob: float
    value: float
    probabilities: np.ndarray
    log_probabilities: np.ndarray
    outcome_probabilities: np.ndarray | None = None


@dataclass
class _Transition:
    """Finalized learner row with its collection-time outcome distribution."""

    pending: _Pending
    reward: float
    done: bool


@dataclass
class RolloutBatch:
    """On-policy rows [time, trajectory]; complete games may be packed [rows, 1]."""

    obs: dict[str, np.ndarray]
    actions: np.ndarray
    logprobs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    next_obs: dict[str, np.ndarray]
    next_done: np.ndarray
    probabilities: np.ndarray
    log_probabilities: np.ndarray
    # Collection-time loss/draw/win probabilities [T,E,3], absent for scalar.
    outcome_probabilities: np.ndarray | None = None


ExposureKind = Literal[
    "game_roots",
    "environment_decisions",
    "learner_transitions",
    "terminal_games",
    "truncated_games",
]


class ExposureCounts(TypedDict):
    game_roots: int
    environment_decisions: int
    learner_transitions: int
    terminal_games: int
    truncated_games: int


@dataclass
class CollectorStats:
    """Cumulative accounting, including the opponent fingerprint."""

    micro_steps: int = 0
    learner_transitions: int = 0
    opponent_decisions: int = 0
    games: int = 0
    learner_wins: int = 0
    self_play: SelfPlayCounter = field(default_factory=SelfPlayCounter)
    truncations: int = 0
    seconds: float = 0.0
    opponent_action_types: dict[str, int] = field(default_factory=dict)
    learner_action_types: dict[str, int] = field(default_factory=dict)
    games_by_stream: dict[int, int] = field(default_factory=dict)
    # Ordered deck identities plus learning seats; counts are realized, not quotas.
    exposure: dict[str, ExposureCounts] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "micro_steps": self.micro_steps,
            "environment_decisions": self.micro_steps,
            "learner_transitions": self.learner_transitions,
            "opponent_decisions": self.opponent_decisions,
            "games": self.games,
            "learner_wins": self.learner_wins,
            "learner_win_rate": (self.learner_wins / self.games if self.games else 0.0),
            "truncations": self.truncations,
            "seconds": self.seconds,
            "opponent_action_types": dict(self.opponent_action_types),
            "learner_action_types": dict(self.learner_action_types),
            "games_by_stream": dict(self.games_by_stream),
            "exposure": deepcopy(self.exposure),
        }


@dataclass
class CurrentGames:
    """Native reset seeds and only the unfinished games' legal action prefixes."""

    seeds: list[int]
    actions: list[list[int]]


@dataclass
class CollectorSnapshot:
    """Private training state. Replay is native execution, not policy inference.

    Journal rows contain [action per stream] and [active per stream]. The
    native constructor seed plus every vector step also restores auto-reset RNG.
    Only empty transition buffers at update boundaries may be snapshotted.
    """

    journal: list[tuple[list[int], list[bool]]]
    buffers: dict[str, np.ndarray]
    sampling_rng: torch.Tensor
    stats: CollectorStats
    current_games: CurrentGames | None = None
    decks_digest: str | None = None


_ACTION_TYPE_NAMES = {int(member): member.name.lower() for member in ActionEnum}


class SeatRoutedCollector:
    """Collect on-policy trajectories against random, frozen or learning policies.

    ``learning_seats="both"`` is the default for ``self``
    (one shared model) or ``learning`` (two distinct models). ``one`` retains
    the exact-quota, learner-only control, including in self-play.
    """

    def __init__(
        self,
        observation_space: ObservationSpace,
        match: Match,
        reward: Reward,
        *,
        num_envs: int,
        seed: int = 0,
        opponent_mode: str = "random",
        opponent_agent: ActorCritic | None = None,
        learning_seats: Literal["one", "both"] | None = None,
        games_per_stream: int | None = None,
        device: str = "cpu",
        recovery_max_microsteps: int | None = None,
        recovery_max_game_microsteps: int | None = None,
        root: str | None = None,
        matchup_curriculum: Curriculum = "fixed",
        deck_source: UniformDecks | None = None,
        workers: int = 1,
        game_errors: GameErrorPolicy | None = None,
        record_attempt: Callable[[GameAttempt], GameErrorState] | None = None,
    ) -> None:
        # The source's identity is fixed for the collector; every snapshot and
        # game attempt records it, so hash it once.
        self._decks_digest = deck_source.digest if deck_source is not None else None
        if deck_source is not None:
            if matchup_curriculum != "fixed":
                raise ValueError("a deck source cannot be combined with a curriculum")
            if match.config.world != WORLD_SETUP:
                raise ValueError("deck-source games play in the world")
            self._uniform_roster = deck_source.players()
        elif matchup_curriculum == "elemental-uniform":
            self._uniform_roster = elemental_roster(match)
        else:
            self._uniform_roster = []
        if self._uniform_roster and (
            workers != 1 or root is not None or opponent_mode != "self"
        ):
            raise ValueError(
                "per-game roster sampling requires current-self, one producer and ordinary roots"
            )
        if workers not in (1, 2, 4) or workers > num_envs:
            raise ValueError("workers must be 1, 2 or 4 and cannot exceed streams")
        if workers != 1 and root is not None:
            raise ValueError("diagnostic roots require one producer")
        if opponent_mode not in OPPONENT_MODES:
            raise ValueError(
                f"opponent_mode must be one of {OPPONENT_MODES}, got {opponent_mode!r}"
            )
        if learning_seats is None:
            learning_seats = "both" if opponent_mode in ("self", "learning") else "one"
        if games_per_stream is not None and games_per_stream < 1:
            raise ValueError("games_per_stream must be positive")
        if learning_seats not in ("one", "both"):
            raise ValueError("learning_seats must be one or both")
        if learning_seats == "both" and opponent_mode not in ("self", "learning"):
            raise ValueError("both learning seats require self or learning opponents")
        if opponent_mode == "learning" and learning_seats != "both":
            raise ValueError("learning opponents require both learning seats")
        if opponent_mode in ("frozen", "learning") and opponent_agent is None:
            raise ValueError(f"opponent_mode={opponent_mode!r} requires opponent_agent")

        self.observation_space = observation_space
        self.match = match
        self.reward = reward
        self.num_envs = num_envs
        self.opponent_mode = opponent_mode
        self.learning_seats = learning_seats
        self.games_per_stream = games_per_stream
        self.opponent_agent = opponent_agent
        self.device = torch.device(device)
        self.stats = CollectorStats()
        if game_errors is not None and (
            record_attempt is None
            or root is not None
            or recovery_max_microsteps is not None
        ):
            raise ValueError(
                "game error recovery needs durable receipts, native games and current-game snapshots"
            )
        self.game_errors = game_errors
        self.game_error_state = GameErrorState()
        self._record_attempt = record_attempt
        self._game_started = [time.perf_counter()] * num_envs
        # The calling training stage owns enablement and resets this timer.
        # Only the collector thread records host intervals; workers never share it.
        self.profiler = Profiler(enabled=False)
        if (
            recovery_max_microsteps is not None
            or recovery_max_game_microsteps is not None
        ) and opponent_mode != "self":
            raise ValueError("recovery supports current-self opponents only")
        self._recovery_limit = recovery_max_microsteps
        self._game_recovery_limit = recovery_max_game_microsteps
        self._game_actions: list[list[int]] = [[] for _ in range(num_envs)]
        self._journal: list[tuple[list[int], list[bool]]] = []

        if root is None:
            self._env = (
                ShardedVectorEnv(num_envs, seed, workers)
                if workers > 1
                else CapturedVector(
                    num_envs=num_envs,
                    seed=seed,
                    skip_trivial=True,
                    opponent_policy="none",
                )
            )
        else:
            from manabot.env.target_practice import TargetPracticeVector

            if (
                root != "lethal-target-v1"
                or recovery_max_microsteps is not None
                or recovery_max_game_microsteps is not None
            ):
                raise ValueError("unsupported training root or root recovery")
            self._env = TargetPracticeVector(observation_space, match, num_envs, seed)
        try:
            self._buffers = _allocate_buffers(observation_space, num_envs)
            self._env.set_buffers(self._buffers)
            self.stream_matches = (
                []
                if self._uniform_roster
                else stream_matches(match, num_envs, matchup_curriculum)
            )
            if deck_source is not None:
                self._env.set_training_decks(self._uniform_roster, deck_source.digest)
            elif self._uniform_roster:
                self._env.set_uniform_roster(self._uniform_roster)
            if matchup_curriculum in ("fixed", "elemental-uniform"):
                self._env.reset_all_into_buffers(match.to_rust())
            else:
                self._env.reset_each_into_buffers(
                    [m.to_rust() for m in self.stream_matches]
                )

            self._refresh_roots()

            #: Learner seat per stream: stream s seats the learner at s % 2, so
            #: half the streams have the learner on the play — seat-balanced.
            self.learner_seat = np.arange(num_envs, dtype=np.int64) % 2

            slots = num_envs * (2 if learning_seats == "both" else 1)
            self._streams: list[list[_Transition]] = [[] for _ in range(slots)]
            self._pending: list[_Pending | None] = [None] * slots
            self._game_start_rows = [0] * slots

            if opponent_mode == "random":
                self._opponent: Any = RandomBatchController(seed=seed + 1)
            elif opponent_mode == "frozen":
                self._opponent = BatchedSampler(
                    opponent_agent, deterministic=False, seed=seed + 1, device=device
                )
            else:  # self-play: live learner weights, bound at collect() time
                self._opponent = None
            self._self_rng = torch.Generator(device=self.device).manual_seed(seed + 1)

            self._win_reward = float(reward.config.win_reward)
            self._lose_reward = float(reward.config.lose_reward)
            for row in range(num_envs):
                self._expose(row, "game_roots")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Join owned CPU producers; ordinary single-vector collectors own no pool."""
        if isinstance(self._env, ShardedVectorEnv):
            self._env.close()

    def snapshot(self) -> CollectorSnapshot:
        """Capture a completed update boundary, including native RNG replay."""
        if self._recovery_limit is None and self._game_recovery_limit is None:
            raise ValueError("collector recovery is disabled")
        if any(self._streams) or any(p is not None for p in self._pending):
            raise ValueError("collector snapshot requires an update boundary")
        return CollectorSnapshot(
            deepcopy(self._journal),
            {key: value.copy() for key, value in self._buffers.items()},
            self._self_rng.get_state().clone(),
            deepcopy(self.stats),
            CurrentGames(self._env.current_game_seeds(), deepcopy(self._game_actions))
            if self._game_recovery_limit is not None
            else None,
            self._decks_digest,
        )

    def restore(self, state: CollectorSnapshot, check: Callable[[], None]) -> None:
        """Replay into a fresh collector; reject divergence before any learning."""
        if getattr(state, "decks_digest", None) != self._decks_digest:
            raise ValueError("collector deck source identity mismatch")
        if self.stats.micro_steps or self._journal:
            raise ValueError("restore requires a fresh collector")
        if state.current_games is not None:
            games = state.current_games
            if (
                self._game_recovery_limit is None
                or len(games.seeds) != self.num_envs
                or len(games.actions) != self.num_envs
                or sum(map(len, games.actions)) > self._game_recovery_limit
            ):
                raise ValueError("current-game replay exceeds recovery bound")
            self._env.reset_seeds_into_buffers(games.seeds)
            for index in range(max(map(len, games.actions), default=0)):
                check()
                active = [index < len(row) for row in games.actions]
                actions = [
                    row[index] if enabled else 0
                    for row, enabled in zip(games.actions, active, strict=True)
                ]
                self._env.step_into_buffers(actions, active)
                if (
                    self._buffers["terminated"].any()
                    or self._buffers["truncated"].any()
                ):
                    raise ValueError("current-game replay crossed a terminal boundary")
            self._game_actions = deepcopy(games.actions)
        else:
            if (
                self._recovery_limit is None
                or state.stats.micro_steps > self._recovery_limit
            ):
                raise ValueError("collector replay exceeds recovery bound")
            for actions, active in state.journal:
                check()
                self._env.step_into_buffers(actions, active)
        for key, expected in state.buffers.items():
            # At an update boundary the terminal flags describe the preceding
            # step; the encoded observation already belongs to its auto-reset root.
            if state.current_games is not None and key in {
                "rewards",
                "terminated",
                "truncated",
            }:
                self._buffers[key][...] = expected
                continue
            if not np.array_equal(self._buffers[key], expected):
                raise ValueError(f"collector replay diverged: {key}")
        self._journal = deepcopy(state.journal)
        self._self_rng.set_state(state.sampling_rng)
        self.stats = deepcopy(state.stats)
        self._refresh_roots()
        # Pre-instrumentation snapshots have no outcome counters to recover.
        if not hasattr(self.stats, "exposure"):
            self.stats.exposure = {}
        if not hasattr(self.stats, "self_play"):
            self.stats.self_play = SelfPlayCounter()

    def _refresh_roots(self) -> None:
        """Read native sampling after resets, before counting the next game's rows."""
        if self._uniform_roster:
            # Actual native setups, not indices: they remain exact for any deck
            # the run submitted, even one outside the supply it began with.
            self.stream_matches = [
                Match(pair_match(*players))
                for players in self._env.current_player_configs()
            ]
        self._self_play_decks = [
            (
                Deck(m.hero_deck, m.hero_sideboard).identity,
                Deck(m.villain_deck, m.villain_sideboard).identity,
            )
            for m in self.stream_matches
        ]

    def _expose(self, row: int, kind: ExposureKind, count: int = 1) -> None:
        if not self._uniform_roster:
            return
        a, b = self._self_play_decks[row]
        seat = (
            "both"
            if self.learning_seats == "both"
            else str(int(self.learner_seat[row]))
        )
        key = f"{a}/{b}/learner-{seat}"
        counts = self.stats.exposure.setdefault(
            key,
            ExposureCounts(
                game_roots=0,
                environment_decisions=0,
                learner_transitions=0,
                terminal_games=0,
                truncated_games=0,
            ),
        )
        counts[kind] += count

    # -- opponent routing -----------------------------------------------------

    def _opponent_actions(
        self, agent: ActorCritic, rows: np.ndarray, inference: PolicyInference
    ) -> np.ndarray:
        if self.opponent_mode == "self":
            obs = self._slice_obs_tensors(rows, inference)
            with torch.inference_mode():
                logits, _ = agent.forward(obs)
                probs = self._policy_logs(logits, obs).exp()
                actions = torch.multinomial(probs, 1, generator=self._self_rng).squeeze(
                    -1
                )
            return actions.cpu().numpy().astype(np.int64)
        return self._opponent.select(self._buffers, rows)

    @staticmethod
    def _policy_logs(
        logits: torch.Tensor, obs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        try:
            return policy_logs(logits, legal_mask(obs["actions_valid"]))
        except NumericalError as error:
            error.tensors.update(
                {f"observation/{key}": value.detach() for key, value in obs.items()}
            )
            raise

    def _slice_obs_tensors(
        self, rows: np.ndarray, inference: PolicyInference | None = None
    ) -> dict[str, torch.Tensor]:
        obs = {}
        for key in OBS_KEYS:
            tensor = torch.from_numpy(self._buffers[key][rows])
            if key == "action_focus":
                # Training convention (VectorEnv): action_focus as float32.
                tensor = tensor.float()
            obs[key] = tensor
        if inference is not None:
            if inference.device.type != self.device.type or (
                self.device.index is not None
                and inference.device.index != self.device.index
            ):
                raise ValueError("collector and inference policy devices differ")
            return inference.prepare(obs)
        return {key: tensor.to(self.device) for key, tensor in obs.items()}

    def _count_action_types(
        self, counter: dict[str, int], rows: np.ndarray, actions: np.ndarray
    ) -> None:
        chosen = self._buffers["actions"][rows, actions, :ACTION_TYPE_DIM]
        type_indices = chosen.argmax(axis=1)
        typed = chosen.max(axis=1) > 0
        for idx, has_type in zip(type_indices, typed):
            name = (
                _ACTION_TYPE_NAMES.get(int(idx), "unknown") if has_type else "unknown"
            )
            counter[name] = counter.get(name, 0) + 1

    # -- collection loop ------------------------------------------------------

    def collect(
        self,
        agent: ActorCritic,
        num_steps: int,
        *,
        deadline_monotonic: float | None = None,
        check: Callable[[], None] | None = None,
    ) -> RolloutBatch:
        """Collect exact one-seat quotas or complete-game two-seat batches.

        With games_per_stream, both controls stop after that many whole games
        in each stream. Otherwise two-seat num_steps is a total-decision floor.
        No sampled action crosses the update boundary.
        """

        if self.opponent_mode == "learning":
            raise ValueError("use collect_pair for two learning models")
        # No cache or prepared observation survives learning, snapshot or restore.
        # Both seats in self-play use this same unchanged behavior-policy scope.
        with self.profiler.track("Collection"), PolicyInference(agent) as inference:
            batch = self._collect(
                agent,
                num_steps,
                inference,
                deadline_monotonic=deadline_monotonic,
                check=check,
            )
            assert isinstance(batch, RolloutBatch)
            return batch

    def collect_pair(
        self,
        agent: ActorCritic,
        num_steps: int,
        *,
        deadline_monotonic: float | None = None,
        check: Callable[[], None] | None = None,
    ) -> tuple[RolloutBatch, RolloutBatch]:
        """Freeze both behavior scopes; return rows owned by each learning model."""
        if self.opponent_mode != "learning" or self.opponent_agent is None:
            raise ValueError("collect_pair requires a learning opponent")
        if agent is self.opponent_agent:
            raise ValueError("use self mode for a shared learning model")
        with (
            self.profiler.track("Collection"),
            PolicyInference(agent) as inference,
            PolicyInference(self.opponent_agent) as opponent_inference,
        ):
            batches = self._collect(
                agent,
                num_steps,
                inference,
                deadline_monotonic=deadline_monotonic,
                check=check,
                opponent_inference=opponent_inference,
            )
        assert isinstance(batches, tuple)
        return batches

    def _collect(
        self,
        agent: ActorCritic,
        num_steps: int,
        inference: PolicyInference,
        *,
        deadline_monotonic: float | None,
        check: Callable[[], None] | None,
        opponent_inference: PolicyInference | None = None,
    ) -> RolloutBatch | tuple[RolloutBatch, RolloutBatch]:

        start = time.perf_counter()
        buffers = self._buffers
        env = self._env
        num_envs = self.num_envs

        both = self.learning_seats == "both"
        complete_games = (
            both or self.games_per_stream is not None or self.game_errors is not None
        )
        if self.game_error_state.stop_reason:
            raise GameErrorLimit(self.game_error_state.stop_reason)
        completed = np.zeros(num_envs, dtype=bool)
        games = np.zeros(num_envs, dtype=np.int64)
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        while True:
            try:
                if check is not None:
                    check()
                if (
                    deadline_monotonic is not None
                    and time.perf_counter() >= deadline_monotonic
                ):
                    raise TimeoutError("collector exceeded training deadline")
            except Exception as error:
                # The run stopped, not the games: deadlines, cancellation and
                # supervision limits settle in-progress games as abandoned
                # attempts outside the game error window.
                deadline = isinstance(error, TimeoutError)
                if (
                    deadline
                    and not self.game_errors
                    and hasattr(env, "capture_failure")
                ):
                    env.capture_failure(error)
                if self.game_errors:
                    seeds = env.current_game_seeds()
                    seats = env.current_agent_indices()
                    for row, prefix in enumerate(self._game_actions):
                        if prefix:
                            self._settle_attempt(
                                row,
                                seeds[row],
                                seats[row],
                                "deadline" if deadline else "abandoned",
                                error_class=type(error).__name__,
                                error=str(error),
                            )
                raise
            acting = np.asarray(env.current_agent_indices(), dtype=np.int64)
            for row in range(num_envs):
                slot = 2 * row + int(acting[row]) if both else row
                if (both or acting[row] == self.learner_seat[row]) and self._pending[
                    slot
                ] is not None:
                    self._finalize(slot, reward=0.0, done=False)
            active = (
                ~completed
                if complete_games
                else np.array([len(stream) < num_steps for stream in self._streams])
            )
            if not active.any():
                break
            actions = np.zeros(num_envs, dtype=np.int64)
            learner_rows = np.flatnonzero(active & (acting == self.learner_seat))
            opp_rows = np.flatnonzero(active & (acting != self.learner_seat))

            if both and self.opponent_mode == "self":
                learner_rows = np.flatnonzero(active)
                opp_rows = np.array([], dtype=np.int64)
            if len(learner_rows):
                actions[learner_rows] = self._sample_learning_rows(
                    agent, learner_rows, acting, inference
                )
                self._count_action_types(
                    self.stats.learner_action_types, learner_rows, actions[learner_rows]
                )

            if len(opp_rows):
                with self.profiler.track("Opponent policy"):
                    if self.opponent_mode == "learning":
                        assert (
                            self.opponent_agent is not None
                            and opponent_inference is not None
                        )
                        opp_acts = self._sample_learning_rows(
                            self.opponent_agent, opp_rows, acting, opponent_inference
                        )
                    else:
                        opp_acts = self._opponent_actions(agent, opp_rows, inference)
                actions[opp_rows] = opp_acts
                if not both:
                    self.stats.opponent_decisions += len(opp_rows)
                self._count_action_types(
                    self.stats.opponent_action_types, opp_rows, opp_acts
                )

            if self._recovery_limit is not None:
                if self.stats.micro_steps + int(active.sum()) > self._recovery_limit:
                    raise RuntimeError("collector recovery journal limit exceeded")
                self._journal.append((actions.tolist(), active.tolist()))
            for row in np.flatnonzero(active):
                self._expose(int(row), "environment_decisions")
            seeds = env.current_game_seeds() if self.game_errors else []
            with self.profiler.track("Environment step and encoding"):
                if self.game_errors:
                    env.step_isolated_into_buffers(actions.tolist(), active.tolist())
                else:
                    env.step_into_buffers(actions.tolist(), active.tolist())
            self.stats.micro_steps += int(active.sum())

            done = (buffers["terminated"] > 0) | (buffers["truncated"] > 0)
            infos = env.get_last_info() if self.game_errors or done.any() else []
            replace = np.zeros(num_envs, dtype=bool)
            failed = np.zeros(num_envs, dtype=bool)
            if self.game_errors:
                for row in np.flatnonzero(active):
                    self._game_actions[row].append(int(actions[row]))
                    outcome: GameOutcome
                    if "game_error" in infos[row]:
                        outcome = "engine_error"
                    elif buffers["truncated"][row]:
                        outcome = "truncated"
                    elif (
                        not done[row]
                        and len(self._game_actions[row])
                        >= self.game_errors.max_game_actions
                    ):
                        outcome = "action_limit"
                    else:
                        continue
                    self.stats.truncations += int(outcome == "truncated")
                    self._settle_attempt(
                        int(row),
                        seeds[row],
                        int(acting[row]),
                        outcome,
                        error_class=infos[row].get("game_error_class", outcome),
                        error=infos[row].get("game_error", outcome),
                        frame_json=infos[row].get("game_error_frame"),
                        native_trace=infos[row].get("game_error_trace"),
                    )
                    failed[row] = True
                    done[row] = False
                    # Native truncation already reset this stream.
                    replace[row] = not bool(buffers["truncated"][row])
            elif self._game_recovery_limit is not None:
                for row in np.flatnonzero(active):
                    if done[row]:
                        self._game_actions[row].clear()
                    else:
                        self._game_actions[row].append(int(actions[row]))
                if sum(map(len, self._game_actions)) > self._game_recovery_limit:
                    raise RuntimeError("current-game recovery journal limit exceeded")
            if done.any():
                for row in np.flatnonzero(done):
                    winner = infos[row].get("winner_index")
                    winner = int(winner) if winner is not None else None
                    self.stats.games += 1
                    self.stats.games_by_stream[int(row)] = (
                        self.stats.games_by_stream.get(int(row), 0) + 1
                    )
                    if buffers["truncated"][row]:
                        self.stats.truncations += 1
                        self._expose(int(row), "truncated_games")
                        raise RuntimeError(
                            "collector game truncated; no terminal target available"
                        )
                    if self.opponent_mode in ("self", "learning"):
                        self.stats.self_play.record(
                            self._self_play_decks[int(row)], winner
                        )
                    if winner is None:
                        reward = 0.0
                    elif winner == int(self.learner_seat[row]):
                        self.stats.learner_wins += 1
                        reward = self._win_reward
                    else:
                        reward = self._lose_reward
                    self._expose(int(row), "terminal_games")
                    if both:
                        for seat in (0, 1):
                            slot = 2 * int(row) + seat
                            seat_reward = (
                                0.0
                                if winner is None
                                else self._win_reward
                                if winner == seat
                                else self._lose_reward
                            )
                            if self._pending[slot] is not None:
                                self._finalize(slot, reward=seat_reward, done=True)
                    elif self._pending[row] is not None:
                        self._finalize(int(row), reward=reward, done=True)
                    if self.game_errors:
                        self._settle_attempt(
                            int(row), seeds[row], int(acting[row]), "completed"
                        )
                    if complete_games:
                        games[row] += 1
                        completed[row] = (
                            games[row] >= self.games_per_stream
                            if self.games_per_stream is not None
                            else sum(
                                len(self._streams[slot])
                                for slot in self._game_slots(int(row))
                            )
                            >= num_steps
                        )

            if self.game_error_state.stop_reason:
                raise GameErrorLimit(self.game_error_state.stop_reason)
            if replace.any():
                env.replace_failed_into_buffers(replace.tolist())
            if (done | failed).any():
                self._refresh_roots()
                for row in np.flatnonzero(done | failed):
                    self._expose(int(row), "game_roots")

        with self.profiler.track("Batch assembly"):
            if complete_games:
                if self.opponent_mode == "learning":
                    # Each owner keeps only its own seat in each stream. No row
                    # may be trained by a policy that did not generate it.
                    first = [
                        2 * row + int(self.learner_seat[row]) for row in range(num_envs)
                    ]
                    second = [
                        2 * row + 1 - int(self.learner_seat[row])
                        for row in range(num_envs)
                    ]
                    batch = (
                        self._build_complete_batch(first),
                        self._build_complete_batch(second),
                    )
                else:
                    batch = self._build_complete_batch(list(range(len(self._streams))))
            else:
                batch = self._build_batch(num_steps)
        self.stats.seconds += time.perf_counter() - start
        self._game_start_rows = [0] * len(self._streams)
        return batch

    def _game_slots(self, row: int) -> range:
        seats = 2 if self.learning_seats == "both" else 1
        return range(seats * row, seats * (row + 1))

    def _settle_attempt(
        self,
        row: int,
        seed: int,
        seat: int,
        outcome: GameOutcome,
        *,
        error_class: str | None = None,
        error: str | None = None,
        frame_json: str | None = None,
        native_trace: str | None = None,
    ) -> None:
        """Settle this game's rows and persist its receipt before clearing replay inputs.

        Failed games lose only their suffix and pending decisions; complete
        earlier games stay valid. The ledger owns allowance across snapshots.
        """
        assert self._record_attempt is not None
        completed = outcome == "completed"
        valid = 0
        discarded = 0
        for slot in self._game_slots(row):
            count = len(self._streams[slot]) - self._game_start_rows[slot]
            if completed:
                valid += count
            else:
                discarded += count + int(self._pending[slot] is not None)
                self.stats.learner_transitions -= count
                self._expose(row, "learner_transitions", -count)
                del self._streams[slot][self._game_start_rows[slot] :]
                self._pending[slot] = None
            self._game_start_rows[slot] = len(self._streams[slot])
        self.game_error_state = self._record_attempt(
            GameAttempt(
                stream=row,
                seed=seed,
                seat=seat,
                matchup=self.stream_matches[row].config,
                decks_digest=self._decks_digest,
                outcome=outcome,
                actions=[] if completed else list(self._game_actions[row]),
                error_class=error_class,
                error=error,
                frame_json=frame_json,
                native_trace=native_trace,
                discarded_samples=discarded,
                valid_samples=valid,
                seconds=time.perf_counter() - self._game_started[row],
            )
        )
        self._game_actions[row].clear()
        self._game_started[row] = time.perf_counter()

    def _sample_learning_rows(
        self,
        agent: ActorCritic,
        rows: np.ndarray,
        acting: np.ndarray,
        inference: PolicyInference,
    ) -> np.ndarray:
        with self.profiler.track("Observation preparation"):
            obs_t = self._slice_obs_tensors(rows, inference)
        with self.profiler.track("Policy inference and action selection"):
            with torch.no_grad():
                outcomes: np.ndarray | None = None
                if agent.config.value_kind == "categorical_wdl":
                    logits, outcome_logits = agent.forward_distribution(obs_t)
                    outcome_t = outcome_logits.softmax(dim=-1)
                    value_t = outcome_t[:, 2] - outcome_t[:, 0]
                    outcomes = outcome_t.cpu().numpy()
                else:
                    logits, value_t = agent(obs_t)
                logs = self._policy_logs(logits, obs_t)
                require(
                    torch.isfinite(value_t),
                    "nonfinite_collection_values",
                    values=value_t,
                    valid=obs_t["actions_valid"] > 0,
                )
                weights = logs.exp()
                action_t = torch.multinomial(
                    weights, 1, generator=self._self_rng
                ).squeeze(-1)
                logprob_t = logs.gather(-1, action_t[:, None]).squeeze(-1)
                probabilities = weights.cpu().numpy()
                log_probabilities = logs.cpu().numpy()
            acts = action_t.cpu().numpy().astype(np.int64)
            logprobs = logprob_t.cpu().numpy()
            values = value_t.view(-1).cpu().numpy()
        with self.profiler.track("Rollout recording"):
            # One copied slice per key; per-row dicts view into it.
            obs_rows = {key: self._buffers[key][rows].copy() for key in OBS_KEYS}
            for j, row in enumerate(rows):
                slot = (
                    2 * int(row) + int(acting[row])
                    if self.learning_seats == "both"
                    else int(row)
                )
                self._pending[slot] = _Pending(
                    obs={key: obs_rows[key][j] for key in OBS_KEYS},
                    action=int(acts[j]),
                    logprob=float(logprobs[j]),
                    value=float(values[j]),
                    probabilities=probabilities[j].copy(),
                    log_probabilities=log_probabilities[j].copy(),
                    outcome_probabilities=(
                        None if outcomes is None else outcomes[j].copy()
                    ),
                )
        return acts

    def _finalize(self, row: int, *, reward: float, done: bool) -> None:
        pending = self._pending[row]
        assert pending is not None
        self._streams[row].append(_Transition(pending, reward, done))
        self._pending[row] = None
        self.stats.learner_transitions += 1
        stream = row // 2 if self.learning_seats == "both" else row
        self._expose(stream, "learner_transitions")

    def _build_complete_batch(self, slots: list[int]) -> RolloutBatch:
        streams = [[transition for slot in slots for transition in self._streams[slot]]]
        if not streams[0]:
            raise RuntimeError("learning model made no decisions in collected games")
        assert streams[0][-1].done
        for slot in slots:
            self._streams[slot].clear()
            assert self._pending[slot] is None
        return self._build_batch(len(streams[0]), streams=streams)

    def _build_batch(
        self, num_steps: int, *, streams: list[list[_Transition]] | None = None
    ) -> RolloutBatch:
        packed = streams is not None
        streams = self._streams if streams is None else streams
        num_envs = len(streams)
        shapes = self.observation_space.encoder.allocate(1)
        obs = {
            key: np.zeros((num_steps, num_envs) + value.shape[1:], dtype=value.dtype)
            for key, value in shapes.items()
        }
        actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        logprobs = np.zeros((num_steps, num_envs), dtype=np.float32)
        rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        dones = np.zeros((num_steps, num_envs), dtype=bool)
        values = np.zeros((num_steps, num_envs), dtype=np.float32)
        next_obs = {
            key: np.zeros((num_envs,) + value.shape[1:], dtype=value.dtype)
            for key, value in shapes.items()
        }
        next_done = np.zeros((num_envs,), dtype=bool)
        probabilities = np.zeros_like(obs["actions_valid"])
        log_probabilities = np.full_like(probabilities, -np.inf)
        outcome_probabilities = (
            np.zeros((num_steps, num_envs, 3), dtype=np.float32)
            if streams[0][0].pending.outcome_probabilities is not None
            else None
        )

        for env_index in range(num_envs):
            stream = streams[env_index]
            for step in range(num_steps):
                transition = stream[step]
                pending = transition.pending
                probabilities[step, env_index] = pending.probabilities
                log_probabilities[step, env_index] = pending.log_probabilities
                if outcome_probabilities is not None:
                    assert pending.outcome_probabilities is not None
                    outcome_probabilities[step, env_index] = (
                        pending.outcome_probabilities
                    )
                for key in OBS_KEYS:
                    obs[key][step, env_index] = pending.obs[key]
                actions[step, env_index] = pending.action
                logprobs[step, env_index] = pending.logprob
                values[step, env_index] = pending.value
                rewards[step, env_index] = transition.reward
                dones[step, env_index] = transition.done
            del stream[:num_steps]
            assert not stream
            if not packed:
                assert self._pending[env_index] is None
            for key in OBS_KEYS:
                next_obs[key][env_index] = self._buffers[key][env_index]
            next_done[env_index] = dones[num_steps - 1, env_index]

        return RolloutBatch(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=dones,
            values=values,
            next_obs=next_obs,
            next_done=next_done,
            probabilities=probabilities,
            log_probabilities=log_probabilities,
            outcome_probabilities=outcome_probabilities,
        )
