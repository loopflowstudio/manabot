"""Seat-routed on-policy collection for shared RL learning rules.

A ``Collector`` is the state of some native game streams: ``open_collector``
starts them, ``collect`` (or ``collect_pair``) steps them under a frozen
behavior scope and returns a ``RolloutBatch``, and ``close_collector`` releases
them. A collector is changed in place and cannot be saved or restored; a
resumed loop opens fresh streams. Who learns and when a batch closes are separate declarations. One-seat collection
alternates learner seats by stream; default both-seat self-play keeps a separate
trajectory for each seat. Complete-game closure collects at least the requested
decisions per native stream, then finishes its current game. Complete trajectories
are packed in one column, separated by terminal flags: GAE cannot cross players or
games, and no behavior sample or pending action survives an update. Batch row
counts may exceed the quota. Exact-quota closure, one seat only, pauses each stream
at the learner-decision quota and carries the game across the update. A game that
then fails loses only the rows not yet learned from; its stream refills to the quota.

Self-play uses one live model for both seats. ``collect_pair`` instead returns
one batch per learning model, with seats balanced across streams; each model's
own policy, critic and distribution generated its rows. Frozen and random
opponent decisions remain excluded. Terminal reward is computed per seat from
``winner_index``; the native reward stream is ignored.

Collection never updates model weights; the calling loop owns learning.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import time
from typing import Any, Literal, TypedDict

import numpy as np
import torch

from manabot.backends.torch.learner import forward_precision
from manabot.backends.torch.policy import ActorCritic
from manabot.backends.torch.rl import NumericalError, legal_mask, policy_logs, require
from manabot.env.failures import CapturedVector
from manabot.env.match import Match, Reward
from manabot.env.observation import ACTION_TYPE_DIM, ActionEnum, ObservationSpace
from manabot.infra.config import MatchConfig
from manabot.infra.profiler import Profiler
from manabot.model.admission import AdmittedObservation
from manabot.model.inference import PolicyInference
from manabot.policy.buckets import AXES
from manabot.sim.rollout import (
    BatchedSampler,
    RandomBatchController,
    _allocate_buffers,
)
from manabot.training.decks import UniformDecks, pair_match
from manabot.training.regime import BatchClosure, Precision
from manabot.training.rollout.actions import ACTION_FIELDS, ActionRows, PackedActions
from manabot.training.rollout.adaptive import (
    AdaptiveConfig,
    AdaptiveDiagnostics,
    AdaptiveSampler,
    AllocationRow,
    Assignment,
    ErrorChunk,
    ExposureRow,
    GameResult,
)
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
    RecordAttempt,
)
from managym import WORLD_SETUP, Deck

OPPONENT_MODES = ("random", "frozen", "self", "learning")

# -----------------------------------------------------------------------------
# Seat-routed rollout collection
# -----------------------------------------------------------------------------


class _ObservationRows:
    """One update's recorded non-action observation rows, each copied once.

    Recording copies a step's sampled rows from the native buffers straight into
    `[capacity, *shape]` arrays; a pending decision keeps only its row number, and
    the batch indexes these arrays instead of re-stacking every row. Each update
    allocates a fresh store because a returned batch may outlive collection
    (incidents) while the next update records; appends never write
    rows a batch already views. Capacity starts a quarter above the larger of
    the previous update's rows and the closure's row floor; growth doubles and
    copies used rows once. Beside the observation fields it keeps each row's
    bucket needs as admission measured them for the acting policy
    (`manabot.policy.buckets.row_needs`; -1 where none were measured), so the
    learner regroups them instead of measuring the rows again.
    """

    def __init__(
        self, buffers: dict[str, np.ndarray], keys: list[str], capacity: int
    ) -> None:
        self._buffers = buffers
        self._keys = keys
        self.rows = 0
        self.capacity = max(capacity, 1)
        self._arrays = self._allocate(self.capacity)
        self._needs = np.full((self.capacity, len(AXES)), -1, np.int64)

    def _allocate(self, capacity: int) -> dict[str, np.ndarray]:
        # Untouched np.empty pages are not resident, so spare capacity is cheap.
        return {
            key: np.empty(
                (capacity, *self._buffers[key].shape[1:]),
                dtype=self._buffers[key].dtype,
            )
            for key in self._keys
        }

    def append(self, rows: np.ndarray, needs: np.ndarray | None = None) -> int:
        """Copy buffer `rows` (and their bucket `needs`, when measured) to the end
        of the store; return the first store row."""
        start, stop = self.rows, self.rows + len(rows)
        if stop > self.capacity:
            self.capacity = max(2 * self.capacity, stop)
            grown = self._allocate(self.capacity)
            for key, array in self._arrays.items():
                grown[key][:start] = array[:start]
            self._arrays = grown
            grown_needs = np.full((self.capacity, len(AXES)), -1, np.int64)
            grown_needs[:start] = self._needs[:start]
            self._needs = grown_needs
        self._needs[start:stop] = -1 if needs is None else needs
        for key, array in self._arrays.items():
            # `clip` writes `out` directly; `raise` would buffer. Rows are in range.
            np.take(
                self._buffers[key], rows, axis=0, out=array[start:stop], mode="clip"
            )
        self.rows = stop
        return start

    def arrays(self) -> dict[str, np.ndarray]:
        """Used rows as views; rows of discarded failed games stay unreferenced."""
        return {key: array[: self.rows] for key, array in self._arrays.items()}

    def needs(self) -> np.ndarray:
        """Used rows' bucket needs [rows, len(AXES)], -1 where unmeasured."""
        return self._needs[: self.rows]


@dataclass
class _Pending:
    """A decision awaiting its reward at the same seat's next turn or game end."""

    obs_row: int  # Row in this update's _ObservationRows.
    action: int
    logprob: float
    value: float
    action_data: PackedActions
    outcome_probabilities: np.ndarray | None = None
    assignment: tuple[int, int, int, int] | None = None


@dataclass
class _Transition:
    """Finalized learner row with its collection-time outcome distribution."""

    pending: _Pending
    reward: float
    done: bool


@dataclass(frozen=True)
class RecordedObservations:
    """Non-action observation fields of one batch, each sampled row stored once.

    `rows` holds recorded rows [N, ...] in recording order, which is neither time-
    nor trajectory-major and may include rows of discarded failed games. `index`
    [time * trajectory] names the recorded row of each time-major rollout row, so
    readers address observations only through rollout indexes.
    """

    rows: dict[str, np.ndarray]
    index: np.ndarray
    # Recorded rows' bucket needs [N, len(AXES)], -1 where unmeasured.
    needs: np.ndarray | None = None

    def gather(self, indexes: np.ndarray) -> dict[str, np.ndarray]:
        """Rows for flat time-major rollout `indexes`, as [batch, ...]."""
        recorded = self.index[indexes]
        return {key: value[recorded] for key, value in self.rows.items()}

    def gather_needs(self, indexes: np.ndarray) -> np.ndarray | None:
        """Bucket needs of rollout `indexes` measured at collection, or None if
        any of those rows has none."""
        if self.needs is None:
            return None
        needs = self.needs[self.index[indexes]]
        return needs if bool((needs >= 0).all()) else None

    def ordered(self, shape: tuple[int, ...]) -> dict[str, np.ndarray]:
        """Every rollout row gathered into `shape` (time, trajectory) order."""
        return {
            key: value.reshape(*shape, *value.shape[1:])
            for key, value in self.gather(np.arange(len(self.index))).items()
        }


@dataclass
class RolloutBatch:
    """On-policy rows [time, trajectory]; complete games may be packed [rows, 1].

    `recorded` owns non-action fields, addressed by time-major rollout index.
    `action_data` owns legal action rows and behavior distributions in time-major
    order. `observations` reunites them.
    """

    recorded: RecordedObservations
    actions: np.ndarray
    logprobs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    next_obs: dict[str, np.ndarray]
    next_done: np.ndarray
    action_data: PackedActions
    # Collection-time loss/draw/win probabilities [T,E,3], absent for scalar.
    outcome_probabilities: np.ndarray | None = None
    assignments: np.ndarray | None = None
    target_residuals: np.ndarray | None = None

    def observations(self, indexes: np.ndarray | None = None) -> ActionRows:
        """Materialize only requested flat rollout rows on CPU.

        Explicit indexes return [batch,...]; omitting them restores
        [time,trajectory,...] for bounded inspection. Learners always select a
        minibatch. Object data and compact actions are the only retained owners.
        """
        trajectory_shape = self.actions.shape if indexes is None else None
        if indexes is None:
            indexes = np.arange(self.actions.size)
        rows = self.action_data.unpack(indexes)
        rows.observation.update(self.recorded.gather(indexes))
        if trajectory_shape is not None:
            rows.observation = {
                key: value.reshape(*trajectory_shape, *value.shape[1:])
                for key, value in rows.observation.items()
            }
            rows.probabilities = rows.probabilities.reshape(*trajectory_shape, -1)
            rows.log_probabilities = rows.log_probabilities.reshape(
                *trajectory_shape, -1
            )
        return rows

    def ordered_obs(self) -> dict[str, np.ndarray]:
        """Non-action fields gathered into [time, trajectory, ...] order."""
        return self.recorded.ordered(self.actions.shape)


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
    # Stream slots left unstepped while another stream finished its batch share.
    idle_stream_steps: int = 0
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
    # Per slot, rows of the game in progress that earlier updates already learned
    # from. Only exact-quota games span updates.
    learned_game_rows: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "micro_steps": self.micro_steps,
            "environment_decisions": self.micro_steps,
            "idle_stream_steps": self.idle_stream_steps,
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


_ACTION_TYPE_NAMES = {int(member): member.name.lower() for member in ActionEnum}


@dataclass
class Collector:
    """Native game streams and the rows they have produced so far.

    State, not behavior: `open_collector` builds one, `collect` and
    `collect_pair` step its streams and change it in place (the native vector,
    its observation buffers and the sampling generator are mutable handles),
    and `close_collector` releases it. Underscored fields belong to this
    module. Slots index pending decisions and finished rows: one per stream,
    or `2 * stream + seat` when both seats learn.
    """

    observation_space: ObservationSpace
    match: Match
    reward: Reward
    num_envs: int
    opponent_mode: str
    opponent_agent: ActorCritic | None
    learning_seats: Literal["one", "both"]
    batch_closure: BatchClosure
    games_per_stream: int | None
    device: torch.device
    precision: Precision
    game_errors: GameErrorPolicy | None
    #: Learner seat per stream: stream s seats the learner at s % 2, so
    #: half the streams have the learner on the play — seat-balanced.
    learner_seat: np.ndarray
    # Sampled rosters replace these with each root's setup in _refresh_roots.
    stream_matches: list[Match]
    stats: CollectorStats
    game_error_state: GameErrorState
    # The calling loop owns enablement and resets this timer. Only the
    # collecting thread records host intervals; workers never share it.
    profiler: Profiler
    # ShardedVectorEnv, CapturedVector or a diagnostic root's Python vector.
    _env: Any
    _buffers: dict[str, np.ndarray]
    _observation_keys: list[str]
    _observations: _ObservationRows
    _self_rng: torch.Generator
    _record_attempt: RecordAttempt | None
    # The deck source's identity is fixed; every game attempt records it.
    _decks_digest: str | None
    _uniform_roster: list[Any]
    _win_reward: float
    _lose_reward: float
    _streams: list[list[_Transition]]
    _pending: list[_Pending | None]
    _game_start_rows: list[int]
    _game_started: list[float]
    # Legal action prefix of each stream's game in progress, for failure receipts.
    _game_actions: list[list[int]]
    _self_play_decks: list[tuple[str, str]]
    # RandomBatchController or BatchedSampler; None when the live model plays both seats.
    _opponent: Any = None
    _sampler: AdaptiveSampler | None = None
    _assignments: list[Assignment] = field(default_factory=list)
    _ordinals: list[int] = field(default_factory=list)
    _results: list[GameResult] = field(default_factory=list)
    _allocations: list[AllocationRow] = field(default_factory=list)
    _uncommitted_batch: RolloutBatch | None = None
    _needs_assignment: set[int] = field(default_factory=set)
    _reverse_cells: list[int] = field(default_factory=list)


def _cell_identity(config: MatchConfig) -> tuple[object, ...]:
    return (
        config.world,
        tuple(sorted(config.hero_deck.items())),
        tuple(sorted(config.hero_sideboard.items())),
        tuple(sorted(config.villain_deck.items())),
        tuple(sorted(config.villain_sideboard.items())),
    )


def _reverse_cells(adaptive: AdaptiveConfig) -> list[int]:
    """Index of each catalogue cell's seat-swapped cell; every cell must have one."""
    indexes = {_cell_identity(cell): i for i, cell in enumerate(adaptive.cells)}
    reverse = []
    for cell in adaptive.cells:
        reversed_cell = Match(cell).swapped().config
        if _cell_identity(reversed_cell) not in indexes:
            raise ValueError("both-seat catalogue must include every reverse cell")
        reverse.append(indexes[_cell_identity(reversed_cell)])
    return reverse


def open_collector(
    observation_space: ObservationSpace,
    match: Match,
    reward: Reward,
    *,
    num_envs: int,
    seed: int = 0,
    opponent_mode: str = "random",
    opponent_agent: ActorCritic | None = None,
    learning_seats: Literal["one", "both"] | None = None,
    batch_closure: BatchClosure = "complete-games",
    games_per_stream: int | None = None,
    device: str = "cpu",
    root: str | None = None,
    matchup_curriculum: Curriculum = "fixed",
    deck_source: UniformDecks | None = None,
    workers: int = 1,
    game_errors: GameErrorPolicy | None = None,
    record_attempt: RecordAttempt | None = None,
    precision: Precision = "float32",
    adaptive: AdaptiveConfig | None = None,
) -> Collector:
    """Start `num_envs` native streams at their first roots.

    ``learning_seats="both"`` is the default for ``self`` (one shared model)
    or ``learning`` (two distinct models). ``one`` retains the learner-only
    control, including in self-play. ``batch_closure`` says separately when a
    batch ends: after whole games, or at an exact one-seat quota that
    bootstraps its mid-game tail from the critic. `record_attempt` persists
    each settled game when `game_errors` is set. The caller closes the
    collector with `close_collector`.
    """
    if adaptive is not None and (
        opponent_mode != "self"
        or learning_seats == "one"
        or workers != 1
        or root is not None
        or matchup_curriculum != "fixed"
    ):
        raise ValueError(
            "adaptive collection requires both-seat current-self, one producer and fixed roots"
        )
    sampler = AdaptiveSampler(adaptive) if adaptive is not None else None
    assignments = [sampler.draw() for _ in range(num_envs)] if sampler else []
    reverse_cells = _reverse_cells(adaptive) if adaptive is not None else []
    uniform_roster: list[Any]
    if deck_source is not None:
        if matchup_curriculum != "fixed":
            raise ValueError("a deck source cannot be combined with a curriculum")
        if match.config.world != WORLD_SETUP:
            raise ValueError("deck-source games play in the world")
        uniform_roster = deck_source.players()
    elif matchup_curriculum == "elemental-uniform":
        uniform_roster = elemental_roster(match)
    else:
        uniform_roster = []
    if uniform_roster and (root is not None or opponent_mode != "self"):
        raise ValueError(
            "per-game roster sampling requires current-self and ordinary roots"
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
    if batch_closure == "exact-quota" and (
        learning_seats == "both"
        or games_per_stream is not None
        or (game_errors is not None and not game_errors.recover_capacity)
    ):
        raise ValueError(
            "exact quota requires one seat, no per-game quota and capacity recovery"
        )
    if game_errors is not None and (record_attempt is None or root is not None):
        raise ValueError("game error recovery needs durable receipts and native games")

    env: Any
    if root is None:
        env = (
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

        if root != "lethal-target-v1":
            raise ValueError("unsupported training root")
        env = TargetPracticeVector(observation_space, match, num_envs, seed)
    try:
        buffers = _allocate_buffers(observation_space, num_envs)
        slots = num_envs * (2 if learning_seats == "both" else 1)
        observation_keys = [
            key for key in observation_space.shapes if key not in ACTION_FIELDS
        ]
        collector = Collector(
            observation_space=observation_space,
            match=match,
            reward=reward,
            num_envs=num_envs,
            opponent_mode=opponent_mode,
            opponent_agent=opponent_agent,
            learning_seats=learning_seats,
            batch_closure=batch_closure,
            games_per_stream=games_per_stream,
            device=torch.device(device),
            precision=precision,
            game_errors=game_errors,
            learner_seat=np.arange(num_envs, dtype=np.int64) % 2,
            stream_matches=[match] * num_envs,
            stats=CollectorStats(),
            game_error_state=GameErrorState(),
            profiler=Profiler(enabled=False),
            _env=env,
            _buffers=buffers,
            _observation_keys=observation_keys,
            _observations=_ObservationRows(buffers, observation_keys, 0),
            _self_rng=torch.Generator(device=torch.device(device)).manual_seed(
                seed + 1
            ),
            _record_attempt=record_attempt,
            _decks_digest=deck_source.digest if deck_source is not None else None,
            _uniform_roster=uniform_roster,
            _win_reward=float(reward.config.win_reward),
            _lose_reward=float(reward.config.lose_reward),
            _streams=[[] for _ in range(slots)],
            _pending=[None] * slots,
            _game_start_rows=[0] * slots,
            _game_started=[time.perf_counter()] * num_envs,
            _game_actions=[[] for _ in range(num_envs)],
            _self_play_decks=[("", "")] * num_envs,
            _sampler=sampler,
            _assignments=assignments,
            _ordinals=[0] * num_envs,
            _reverse_cells=reverse_cells,
        )
        env.set_buffers(
            buffers,
            action_overflow=observation_space.encoder.config.action_overflow,
        )
        if not uniform_roster:
            collector.stream_matches = stream_matches(
                match, num_envs, matchup_curriculum
            )
        if deck_source is not None:
            env.set_training_decks(
                uniform_roster, deck_source.digest, deck_source.sampling
            )
        elif uniform_roster:
            env.set_uniform_roster(uniform_roster)
        if sampler is not None:
            collector.stream_matches = [
                _assigned_match(collector, a) for a in assignments
            ]
            env.reset_each_into_buffers([m.to_rust() for m in collector.stream_matches])
        elif matchup_curriculum in ("fixed", "elemental-uniform"):
            env.reset_all_into_buffers(match.to_rust())
        else:
            env.reset_each_into_buffers([m.to_rust() for m in collector.stream_matches])

        _refresh_roots(collector, np.arange(num_envs))
        if sampler is not None:
            for row in range(num_envs):
                _record_assignment(collector, row)

        if opponent_mode == "random":
            collector._opponent = RandomBatchController(seed=seed + 1)
        elif opponent_mode == "frozen":
            collector._opponent = BatchedSampler(
                opponent_agent, deterministic=False, seed=seed + 1, device=device
            )
        for row in range(num_envs):
            _expose(collector, row, "game_roots")
    except BaseException:
        _close_env(env)
        raise
    return collector


def _close_env(env: Any) -> None:
    if isinstance(env, (ShardedVectorEnv, CapturedVector)):
        env.close()


def close_collector(collector: Collector) -> None:
    """Join owned CPU producers; ordinary single-vector collectors own no pool."""
    _close_env(collector._env)


def _refresh_roots(collector: Collector, rows: np.ndarray) -> None:
    """Read native sampling after resets, before counting the next game's rows.

    Only `rows` hold new games: a stream's setup is fixed until it resets.
    """
    if collector._sampler is not None:
        for row in rows:
            collector.stream_matches[row] = _assigned_match(
                collector, collector._assignments[row]
            )
    if collector._uniform_roster:
        # Actual native setups, not indices: they remain exact for any deck
        # the run submitted, even one outside the supply it began with.
        setups = collector._env.current_player_configs()
        for row in rows:
            collector.stream_matches[row] = Match(pair_match(*setups[row]))
    for row in rows:
        match = collector.stream_matches[row]
        collector._self_play_decks[row] = (
            Deck(match.hero_deck, match.hero_sideboard).identity,
            Deck(match.villain_deck, match.villain_sideboard).identity,
        )


def _expose(collector: Collector, row: int, kind: ExposureKind, count: int = 1) -> None:
    if not collector._uniform_roster and collector._sampler is None:
        return
    a, b = collector._self_play_decks[row]
    seat = (
        "both"
        if collector.learning_seats == "both"
        else str(int(collector.learner_seat[row]))
    )
    key = f"{a}/{b}/learner-{seat}"
    counts = collector.stats.exposure.setdefault(
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


# -----------------------------------------------------------------------------
# Opponent routing
# -----------------------------------------------------------------------------


def _opponent_actions(
    collector: Collector,
    agent: ActorCritic,
    rows: np.ndarray,
    inference: PolicyInference,
) -> np.ndarray:
    if collector.opponent_mode == "self":
        obs = _slice_obs_tensors(collector, rows, inference)
        with torch.inference_mode():
            with forward_precision(collector.precision, collector.device):
                logits, _ = agent.forward(obs)
            logits = logits.float()
            probs = _policy_logs(logits, obs).exp()
            actions = torch.multinomial(
                probs, 1, generator=collector._self_rng
            ).squeeze(-1)
        return actions.cpu().numpy().astype(np.int64)
    return collector._opponent.select(collector._buffers, rows)


def _policy_logs(logits: torch.Tensor, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    try:
        return policy_logs(logits, legal_mask(obs["actions_valid"]))
    except NumericalError as error:
        error.tensors.update(
            {f"observation/{key}": value.detach() for key, value in obs.items()}
        )
        raise


def _slice_obs_tensors(
    collector: Collector, rows: np.ndarray, inference: PolicyInference | None = None
) -> dict[str, torch.Tensor]:
    obs = {}
    for key in collector.observation_space.shapes:
        tensor = torch.from_numpy(collector._buffers[key][rows])
        if key == "action_focus":
            # Training convention (VectorEnv): action_focus as float32.
            tensor = tensor.float()
        obs[key] = tensor
    if inference is not None:
        if inference.device.type != collector.device.type or (
            collector.device.index is not None
            and inference.device.index != collector.device.index
        ):
            raise ValueError("collector and inference policy devices differ")
        return inference.prepare(obs)
    return {key: tensor.to(collector.device) for key, tensor in obs.items()}


def _count_action_types(
    collector: Collector, counter: dict[str, int], rows: np.ndarray, actions: np.ndarray
) -> None:
    chosen = collector._buffers["actions"][rows, actions, :ACTION_TYPE_DIM]
    type_indices = chosen.argmax(axis=1)
    typed = chosen.max(axis=1) > 0
    for idx, has_type in zip(type_indices, typed):
        name = _ACTION_TYPE_NAMES.get(int(idx), "unknown") if has_type else "unknown"
        counter[name] = counter.get(name, 0) + 1


# -----------------------------------------------------------------------------
# Collection
# -----------------------------------------------------------------------------


def collect(
    collector: Collector,
    agent: ActorCritic,
    num_steps: int,
    *,
    deadline_monotonic: float | None = None,
    check: Callable[[], None] | None = None,
) -> RolloutBatch:
    """Collect a batch closed at an exact quota or after complete games.

    Exact quota pauses each stream at num_steps learner rows, and its game
    continues in the next batch. Complete games treat num_steps as a floor on
    the stream's retained decisions, the total across seats when both learn;
    games_per_stream replaces the floor with that many whole games, and no
    sampled action crosses the update boundary.
    """

    if collector.opponent_mode == "learning":
        raise ValueError("use collect_pair for two learning models")
    # No cache or prepared observation survives learning.
    # Both seats in self-play use this same unchanged behavior-policy scope.
    with collector.profiler.track("Collection"), PolicyInference(agent) as inference:
        batch = _collect(
            collector,
            agent,
            num_steps,
            inference,
            deadline_monotonic=deadline_monotonic,
            check=check,
        )
        assert isinstance(batch, RolloutBatch)
        return batch


def collect_pair(
    collector: Collector,
    agent: ActorCritic,
    num_steps: int,
    *,
    deadline_monotonic: float | None = None,
    check: Callable[[], None] | None = None,
) -> tuple[RolloutBatch, RolloutBatch]:
    """Freeze both behavior scopes; return rows owned by each learning model."""
    if collector.opponent_mode != "learning" or collector.opponent_agent is None:
        raise ValueError("collect_pair requires a learning opponent")
    if agent is collector.opponent_agent:
        raise ValueError("use self mode for a shared learning model")
    with (
        collector.profiler.track("Collection"),
        PolicyInference(agent) as inference,
        PolicyInference(collector.opponent_agent) as opponent_inference,
    ):
        batches = _collect(
            collector,
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
    collector: Collector,
    agent: ActorCritic,
    num_steps: int,
    inference: PolicyInference,
    *,
    deadline_monotonic: float | None,
    check: Callable[[], None] | None,
    opponent_inference: PolicyInference | None = None,
) -> RolloutBatch | tuple[RolloutBatch, RolloutBatch]:

    if collector._needs_assignment or collector._results:
        raise ValueError("commit adaptive targets before the next collection")
    start = time.perf_counter()
    env = collector._env
    num_envs = collector.num_envs

    both = collector.learning_seats == "both"
    complete_games = collector.batch_closure == "complete-games"
    if collector.game_error_state.stop_reason:
        raise GameErrorLimit(collector.game_error_state.stop_reason)
    completed = np.zeros(num_envs, dtype=bool)
    games = np.zeros(num_envs, dtype=np.int64)
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    # Update boundaries hold no pending row, so nothing live refers to the
    # previous store; returned batches keep their own reference to it.
    if all(pending is None for pending in collector._pending):
        # The batch holds at least num_steps rows per slot (exact quota) or
        # per stream (complete games); whole-game counts give no floor.
        floor = (
            0
            if collector.games_per_stream is not None
            else num_steps * (num_envs if complete_games else len(collector._streams))
        )
        rows = max(collector._observations.rows, floor)
        collector._observations = _ObservationRows(
            collector._buffers, collector._observation_keys, rows + rows // 4
        )
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
                and not collector.game_errors
                and hasattr(env, "capture_failure")
            ):
                env.capture_failure(error)
            if collector.game_errors:
                seeds = env.current_game_seeds()
                seats = env.current_agent_indices()
                for row, prefix in enumerate(collector._game_actions):
                    if prefix:
                        _settle_attempt(
                            collector,
                            row,
                            seeds[row],
                            seats[row],
                            "deadline" if deadline else "abandoned",
                            error_class=type(error).__name__,
                            error=str(error),
                        )
            raise
        with collector.profiler.track("Stream bookkeeping"):
            acting = np.asarray(env.current_agent_indices(), dtype=np.int64)
            for row in range(num_envs):
                slot = 2 * row + int(acting[row]) if both else row
                if (both or acting[row] == collector.learner_seat[row]) and (
                    collector._pending[slot] is not None
                ):
                    _finalize(collector, slot, reward=0.0, done=False)
        active = (
            ~completed
            if complete_games
            else np.array([len(stream) < num_steps for stream in collector._streams])
        )
        if not active.any():
            break
        collector.stats.idle_stream_steps += num_envs - int(active.sum())
        actions = np.zeros(num_envs, dtype=np.int64)
        learner_rows = np.flatnonzero(active & (acting == collector.learner_seat))
        opp_rows = np.flatnonzero(active & (acting != collector.learner_seat))

        if both and collector.opponent_mode == "self":
            learner_rows = np.flatnonzero(active)
            opp_rows = np.array([], dtype=np.int64)
        if len(learner_rows):
            actions[learner_rows] = _sample_learning_rows(
                collector, agent, learner_rows, acting, inference
            )
            _count_action_types(
                collector,
                collector.stats.learner_action_types,
                learner_rows,
                actions[learner_rows],
            )

        if len(opp_rows):
            with collector.profiler.track("Opponent policy"):
                if collector.opponent_mode == "learning":
                    assert (
                        collector.opponent_agent is not None
                        and opponent_inference is not None
                    )
                    opp_acts = _sample_learning_rows(
                        collector,
                        collector.opponent_agent,
                        opp_rows,
                        acting,
                        opponent_inference,
                    )
                else:
                    opp_acts = _opponent_actions(collector, agent, opp_rows, inference)
            actions[opp_rows] = opp_acts
            if not both:
                collector.stats.opponent_decisions += len(opp_rows)
            _count_action_types(
                collector, collector.stats.opponent_action_types, opp_rows, opp_acts
            )

        for row in np.flatnonzero(active):
            _expose(collector, int(row), "environment_decisions")
        seeds = env.current_game_seeds() if collector.game_errors else []
        with (
            collector.profiler.track("Environment step and encoding"),
            _native_step_split(collector),
        ):
            if collector.game_errors:
                env.step_isolated_into_buffers(
                    actions.tolist(),
                    active.tolist(),
                    collector.game_errors.recover_capacity,
                )
            else:
                env.step_into_buffers(actions.tolist(), active.tolist())
        collector.stats.micro_steps += int(active.sum())

        with collector.profiler.track("Stream bookkeeping"):
            _settle_step(
                collector, actions, active, acting, seeds, games, completed, num_steps
            )

    if collector.game_errors and not complete_games:
        # A paused game's rows so far are learned from by this batch; a later
        # failure of that game must not discard them.
        for slot, stream in enumerate(collector._streams):
            rows = len(stream) - collector._game_start_rows[slot]
            if rows:
                collector.stats.learned_game_rows[slot] = (
                    collector.stats.learned_game_rows.get(slot, 0) + rows
                )
    with collector.profiler.track("Batch assembly"):
        if complete_games:
            if collector.opponent_mode == "learning":
                # Each owner keeps only its own seat in each stream. No row
                # may be trained by a policy that did not generate it.
                first = [
                    2 * row + int(collector.learner_seat[row])
                    for row in range(num_envs)
                ]
                second = [
                    2 * row + 1 - int(collector.learner_seat[row])
                    for row in range(num_envs)
                ]
                batch = (
                    _build_complete_batch(collector, first),
                    _build_complete_batch(collector, second),
                )
            else:
                batch = _build_complete_batch(
                    collector, list(range(len(collector._streams)))
                )
        else:
            batch = _build_batch(collector, num_steps)
    collector.stats.seconds += time.perf_counter() - start
    collector._game_start_rows = [0] * len(collector._streams)
    if collector._sampler is not None:
        assert isinstance(batch, RolloutBatch)
        collector._uncommitted_batch = batch
    return batch


def _record_assignment(collector: Collector, row: int) -> None:
    assignment = collector._assignments[row]
    first, second = collector._self_play_decks[row]
    collector._allocations.append(
        AllocationRow(
            stream=row,
            game=collector._ordinals[row],
            cell=assignment.cell,
            seat=assignment.seat,
            generation=assignment.generation,
            component=assignment.component,
            probability=assignment.probability,
            learner_deck=first if assignment.seat == 0 else second,
            opponent_deck=second if assignment.seat == 0 else first,
        )
    )


def _assigned_match(collector: Collector, assignment: Assignment) -> Match:
    assert collector._sampler is not None
    match = Match(collector._sampler.config.cells[assignment.cell])
    return match if assignment.seat == 0 else match.swapped()


def _role_cell(collector: Collector, row: int, seat: int) -> int:
    assignment = collector._assignments[row]
    return (
        assignment.cell
        if assignment.seat == seat
        else collector._reverse_cells[assignment.cell]
    )


def _reassign(collector: Collector, rows: set[int]) -> None:
    if not rows:
        return
    assert collector._sampler is not None
    chosen = {row: collector._sampler.draw() for row in sorted(rows)}
    collector._env.reassign_roots_into_buffers(
        [
            _assigned_match(collector, chosen[row]).to_rust() if row in chosen else None
            for row in range(collector.num_envs)
        ]
    )
    for row, assignment in chosen.items():
        collector._assignments[row] = assignment
    collector._needs_assignment -= rows
    _refresh_roots(collector, np.array(sorted(rows), dtype=np.int64))
    for row in sorted(rows):
        _record_assignment(collector, row)
        _expose(collector, row, "game_roots")


def commit_adaptive(
    collector: Collector, batch: RolloutBatch
) -> AdaptiveDiagnostics | None:
    """Commit pre-filter target errors only after a successful learner update."""
    if collector._sampler is None:
        return None
    if batch is not collector._uncommitted_batch:
        raise ValueError("adaptive targets must belong to the uncommitted batch")
    if batch.assignments is None or batch.target_residuals is None:
        raise ValueError(
            "adaptive update requires row identities and actual target residuals"
        )
    residuals = np.abs(batch.target_residuals).reshape(-1)
    labels = batch.assignments.reshape(-1, 4)
    if len(residuals) != len(labels) or not np.isfinite(residuals).all():
        raise ValueError("invalid adaptive target residuals")
    chunks = []
    for cell in sorted(set(labels[:, 0].tolist())):
        selected = labels[:, 0] == cell
        chunks.append(
            ErrorChunk(
                int(cell),
                float(residuals[selected].sum()),
                int(selected.sum()),
                frozenset((int(row[2]), int(row[3])) for row in labels[selected]),
            )
        )
    results = list(collector._results)
    collector._sampler.commit(results, chunks)
    collector._results.clear()
    _reassign(collector, set(collector._needs_assignment))
    collector._uncommitted_batch = None
    exposures: list[ExposureRow] = []
    legal_counts = np.diff(batch.action_data.offsets)
    surprise = -batch.logprobs.reshape(-1)
    for cell, seat in sorted({(int(row[0]), int(row[1])) for row in labels}):
        mask = (labels[:, 0] == cell) & (labels[:, 1] == seat)
        exposures.append(
            ExposureRow(
                cell=cell,
                seat=seat,
                transitions=int(mask.sum()),
                residual_total=float(residuals[mask].sum()),
                action_surprise_total=float(surprise[mask].sum()),
                legal_offer_total=int(legal_counts[mask].sum()),
                forced_actions=int((legal_counts[mask] == 1).sum()),
            )
        )
    allocations, collector._allocations = collector._allocations, []
    return {
        "allocations": allocations,
        "completed": [
            dict(
                stream=r.game[0],
                game=r.game[1],
                cell=r.cell,
                seat=r.seat,
                outcome=r.outcome,
            )
            for r in results
        ],
        "errors": [
            dict(
                cell=c.cell,
                total=c.total,
                count=c.count,
                games=[list(game) for game in sorted(c.games)],
            )
            for c in chunks
        ],
        "exposures": exposures,
        "signal": collector._sampler.config.signal,
        "generation": collector._sampler.state.generation,
        "starts": collector._sampler.state.starts,
        "completed_games": len({result.game for result in results}),
        "completed_roles": len(results),
        "matchup_scores": list(collector._sampler.state.matchup_scores),
        "error_scores": list(collector._sampler.state.error_scores),
        "priority_probabilities": list(collector._sampler.state.probabilities),
        "residual_count": len(residuals),
        "residual_total": float(residuals.sum()),
    }


def _settle_step(
    collector: Collector,
    actions: np.ndarray,
    active: np.ndarray,
    acting: np.ndarray,
    seeds: list[int],
    games: np.ndarray,
    completed: np.ndarray,
    num_steps: int,
) -> None:
    """Record what one native step ended; count closure in `games`/`completed`."""
    buffers = collector._buffers
    env = collector._env
    both = collector.learning_seats == "both"
    complete_games = collector.batch_closure == "complete-games"
    done = (buffers["terminated"] > 0) | (buffers["truncated"] > 0)
    infos = env.get_last_info() if collector.game_errors or done.any() else []
    replace = np.zeros(collector.num_envs, dtype=bool)
    failed = np.zeros(collector.num_envs, dtype=bool)
    if collector.game_errors:
        for row in np.flatnonzero(active):
            collector._game_actions[row].append(int(actions[row]))
            outcome: GameOutcome
            if "game_error" in infos[row]:
                outcome = (
                    "resource_limit"
                    if infos[row].get("game_error_kind") == "resource_limit"
                    else "engine_error"
                )
            elif buffers["truncated"][row]:
                outcome = "truncated"
            elif (
                not done[row]
                and len(collector._game_actions[row])
                >= collector.game_errors.max_game_actions
            ):
                outcome = "action_limit"
            else:
                continue
            collector.stats.truncations += int(outcome == "truncated")
            _settle_attempt(
                collector,
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
    for row in np.flatnonzero(done):
        winner = infos[row].get("winner_index")
        winner = int(winner) if winner is not None else None
        if collector._sampler is not None:
            for seat in (0, 1):
                collector._results.append(
                    GameResult(
                        _role_cell(collector, int(row), seat),
                        seat,
                        (int(row), collector._ordinals[row]),
                        0.5 if winner is None else float(winner == seat),
                    )
                )
        collector.stats.games += 1
        collector.stats.games_by_stream[int(row)] = (
            collector.stats.games_by_stream.get(int(row), 0) + 1
        )
        if buffers["truncated"][row]:
            collector.stats.truncations += 1
            _expose(collector, int(row), "truncated_games")
            raise RuntimeError("collector game truncated; no terminal target available")
        if collector.opponent_mode in ("self", "learning"):
            collector.stats.self_play.record(
                collector._self_play_decks[int(row)], winner
            )
        if winner is None:
            reward = 0.0
        elif winner == int(collector.learner_seat[row]):
            collector.stats.learner_wins += 1
            reward = collector._win_reward
        else:
            reward = collector._lose_reward
        _expose(collector, int(row), "terminal_games")
        if both:
            for seat in (0, 1):
                slot = 2 * int(row) + seat
                seat_reward = (
                    0.0
                    if winner is None
                    else collector._win_reward
                    if winner == seat
                    else collector._lose_reward
                )
                if collector._pending[slot] is not None:
                    _finalize(collector, slot, reward=seat_reward, done=True)
        elif collector._pending[row] is not None:
            _finalize(collector, int(row), reward=reward, done=True)
        if collector.game_errors:
            _settle_attempt(
                collector, int(row), seeds[row], int(acting[row]), "completed"
            )
        if complete_games:
            games[row] += 1
            completed[row] = (
                games[row] >= collector.games_per_stream
                if collector.games_per_stream is not None
                else sum(
                    len(collector._streams[slot])
                    for slot in _game_slots(collector, int(row))
                )
                >= num_steps
            )

    if collector.game_error_state.stop_reason:
        raise GameErrorLimit(collector.game_error_state.stop_reason)
    if replace.any():
        env.replace_failed_into_buffers(replace.tolist())
    ended = np.flatnonzero(done | failed)
    if collector._sampler is not None:
        # Adaptive roots: a finished stream waits for its next assignment,
        # drawn now or after the scores of this update commit.
        for row in ended:
            collector._ordinals[row] += 1
            collector._needs_assignment.add(int(row))
        _reassign(
            collector,
            {row for row in collector._needs_assignment if not completed[row]},
        )
    if ended.size:
        _refresh_roots(collector, ended)
        if collector._sampler is None:
            for row in ended:
                _expose(collector, int(row), "game_roots")


@contextmanager
def _native_step_split(collector: Collector) -> Iterator[None]:
    """Split the running step phase by the engine's clock; Python envs have none."""
    if not isinstance(collector._env, (ShardedVectorEnv, CapturedVector)):
        yield
        return
    stepped, encoded = collector._env.buffered_call_timings()
    yield
    stepped_after, encoded_after = collector._env.buffered_call_timings()
    collector.profiler.record("Native stepping", stepped_after - stepped)
    collector.profiler.record("Observation encoding", encoded_after - encoded)


def _game_slots(collector: Collector, row: int) -> range:
    seats = 2 if collector.learning_seats == "both" else 1
    return range(seats * row, seats * (row + 1))


def _settle_attempt(
    collector: Collector,
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
    earlier games stay valid, and so do a paused game's rows that an earlier
    update learned from. The ledger owns allowance across attempts.
    """
    assert collector._record_attempt is not None
    completed = outcome == "completed"
    valid = 0
    discarded = 0
    learned = 0
    for slot in _game_slots(collector, row):
        count = len(collector._streams[slot]) - collector._game_start_rows[slot]
        earlier = collector.stats.learned_game_rows.pop(slot, 0)
        if completed:
            valid += count + earlier
        else:
            learned += earlier
            discarded += count + int(collector._pending[slot] is not None)
            collector.stats.learner_transitions -= count
            _expose(collector, row, "learner_transitions", -count)
            del collector._streams[slot][collector._game_start_rows[slot] :]
            collector._pending[slot] = None
        collector._game_start_rows[slot] = len(collector._streams[slot])
    collector.game_error_state = collector._record_attempt(
        collector.game_error_state,
        GameAttempt(
            stream=row,
            seed=seed,
            seat=seat,
            matchup=collector.stream_matches[row].config,
            decks_digest=collector._decks_digest,
            outcome=outcome,
            actions=[] if completed else list(collector._game_actions[row]),
            error_class=error_class,
            error=error,
            frame_json=frame_json,
            native_trace=native_trace,
            observation=collector.observation_space.encoder.config
            if outcome == "resource_limit"
            else None,
            discarded_samples=discarded,
            learned_samples=learned,
            valid_samples=valid,
            seconds=time.perf_counter() - collector._game_started[row],
        ),
    )
    collector._game_actions[row].clear()
    collector._game_started[row] = time.perf_counter()


def _sample_learning_rows(
    collector: Collector,
    agent: ActorCritic,
    rows: np.ndarray,
    acting: np.ndarray,
    inference: PolicyInference,
) -> np.ndarray:
    with collector.profiler.track("Observation preparation"):
        obs_t = _slice_obs_tensors(collector, rows, inference)
    with collector.profiler.track("Policy inference and action selection"):
        with torch.no_grad():
            outcomes: np.ndarray | None = None
            with forward_precision(collector.precision, collector.device):
                if agent.config.value_kind == "categorical_wdl":
                    logits, value_t = agent.forward_distribution(obs_t)
                else:
                    logits, value_t = agent(obs_t)
            logits, value_t = logits.float(), value_t.float()
            if agent.config.value_kind == "categorical_wdl":
                outcome_t = value_t.softmax(dim=-1)
                value_t = outcome_t[:, 2] - outcome_t[:, 0]
                outcomes = outcome_t.cpu().numpy()
            logs = _policy_logs(logits, obs_t)
            require(
                torch.isfinite(value_t),
                "nonfinite_collection_values",
                values=value_t,
                valid=obs_t["actions_valid"] > 0,
            )
            weights = logs.exp()
            action_t = torch.multinomial(
                weights, 1, generator=collector._self_rng
            ).squeeze(-1)
            logprob_t = logs.gather(-1, action_t[:, None]).squeeze(-1)
            probabilities = weights.cpu().numpy()
            log_probabilities = logs.cpu().numpy()
        acts = action_t.cpu().numpy().astype(np.int64)
        logprobs = logprob_t.cpu().numpy()
        values = value_t.view(-1).cpu().numpy()
    with collector.profiler.track("Rollout recording"):
        needs = obs_t.needs if isinstance(obs_t, AdmittedObservation) else None
        first = collector._observations.append(rows, needs)
        packed = PackedActions.pack(
            {
                key: collector._buffers[key][rows]
                for key in ACTION_FIELDS
                if key in collector._buffers
            },
            probabilities,
            log_probabilities,
        )
        for j, row in enumerate(rows):
            slot = (
                2 * int(row) + int(acting[row])
                if collector.learning_seats == "both"
                else int(row)
            )
            collector._pending[slot] = _Pending(
                assignment=(
                    _role_cell(collector, int(row), int(acting[row])),
                    int(acting[row]),
                    int(row),
                    collector._ordinals[row],
                )
                if collector._sampler is not None
                else None,
                obs_row=first + j,
                action=int(acts[j]),
                logprob=float(logprobs[j]),
                value=float(values[j]),
                action_data=packed.row(j),
                outcome_probabilities=(
                    None if outcomes is None else outcomes[j].copy()
                ),
            )
    return acts


def _finalize(collector: Collector, row: int, *, reward: float, done: bool) -> None:
    pending = collector._pending[row]
    assert pending is not None
    collector._streams[row].append(_Transition(pending, reward, done))
    collector._pending[row] = None
    collector.stats.learner_transitions += 1
    stream = row // 2 if collector.learning_seats == "both" else row
    _expose(collector, stream, "learner_transitions")


def _build_complete_batch(collector: Collector, slots: list[int]) -> RolloutBatch:
    streams = [
        [transition for slot in slots for transition in collector._streams[slot]]
    ]
    if not streams[0]:
        raise RuntimeError("learning model made no decisions in collected games")
    assert streams[0][-1].done
    for slot in slots:
        collector._streams[slot].clear()
        assert collector._pending[slot] is None
    return _build_batch(collector, len(streams[0]), streams=streams)


def _build_batch(
    collector: Collector,
    num_steps: int,
    *,
    streams: list[list[_Transition]] | None = None,
) -> RolloutBatch:
    packed = streams is not None
    streams = collector._streams if streams is None else streams
    num_envs = len(streams)
    obs_rows = np.zeros((num_steps, num_envs), dtype=np.int64)
    actions = np.zeros((num_steps, num_envs), dtype=np.int64)
    logprobs = np.zeros((num_steps, num_envs), dtype=np.float32)
    rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
    dones = np.zeros((num_steps, num_envs), dtype=bool)
    values = np.zeros((num_steps, num_envs), dtype=np.float32)
    next_obs = {
        key: collector._buffers[key][:num_envs].copy()
        for key in collector.observation_space.shapes
    }
    next_done = np.zeros((num_envs,), dtype=bool)
    # Flatten in time-major order, matching scalar credit/selected actions.
    action_data = PackedActions.concatenate(
        [
            streams[env][step].pending.action_data
            for step in range(num_steps)
            for env in range(num_envs)
        ]
    )
    outcome_probabilities = (
        np.zeros((num_steps, num_envs, 3), dtype=np.float32)
        if streams[0][0].pending.outcome_probabilities is not None
        else None
    )

    assignments = (
        np.empty((num_steps, num_envs, 4), dtype=np.int64)
        if collector._sampler is not None
        else None
    )
    for env_index in range(num_envs):
        transitions = streams[env_index][:num_steps]
        pending = [transition.pending for transition in transitions]
        if outcome_probabilities is not None:
            outcome_probabilities[:, env_index] = [
                row.outcome_probabilities for row in pending
            ]
        if assignments is not None:
            assert all(row.assignment is not None for row in pending)
            assignments[:, env_index] = [row.assignment for row in pending]
        obs_rows[:, env_index] = [row.obs_row for row in pending]
        actions[:, env_index] = [row.action for row in pending]
        logprobs[:, env_index] = [row.logprob for row in pending]
        values[:, env_index] = [row.value for row in pending]
        rewards[:, env_index] = [transition.reward for transition in transitions]
        dones[:, env_index] = [transition.done for transition in transitions]
        del streams[env_index][:num_steps]
        assert not streams[env_index]
        if not packed:
            assert collector._pending[env_index] is None
        next_done[env_index] = dones[num_steps - 1, env_index]

    return RolloutBatch(
        recorded=RecordedObservations(
            collector._observations.arrays(),
            obs_rows.reshape(-1),
            collector._observations.needs(),
        ),
        actions=actions,
        logprobs=logprobs,
        rewards=rewards,
        dones=dones,
        values=values,
        next_obs=next_obs,
        next_done=next_done,
        action_data=action_data,
        outcome_probabilities=outcome_probabilities,
        assignments=assignments,
    )
