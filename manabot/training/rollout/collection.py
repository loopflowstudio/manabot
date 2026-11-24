"""Seat-routed on-policy collection for shared RL learning rules.

A ``Collector`` is the state of some native game streams: ``open_collector``
starts them, ``collect`` (or ``collect_pair``) steps them under a frozen
behavior scope and returns a ``RolloutBatch``, and ``close_collector`` releases
them. A collector is changed in place and cannot be saved or restored; a
resumed loop opens fresh streams. Who learns and when a batch closes are separate declarations. One-seat collection
alternates learner seats by stream; default both-seat self-play keeps a separate
trajectory for each seat. Complete-game closure collects at least the requested
decisions per native stream, then finishes its current game. Complete trajectories
are packed in one column. Snapshot continuations may also close at a decision
horizon, with a frozen bootstrap for each contributing seat. GAE stays within
each trajectory. No sampled action survives an update. Batch row counts may
exceed the quota. Exact-quota closure, one seat only, pauses each stream
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

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import time
from typing import Any, Literal, Protocol, TypedDict

import numpy as np

from manabot.env.failures import RecordedVectorEnv
from manabot.env.match import Match, Reward
from manabot.env.memory_store import (
    MEMORY_FIELDS,
    MemoryStore,
    Prefix,
    PrefixObservation,
    select_rows,
)
from manabot.env.observation import (
    ACTION_TYPE_DIM,
    ActionEnum,
    ObservationBuffers,
    ObservationSpace,
    allocate_vector_buffers,
)
from manabot.infra.config import MatchConfig
from manabot.infra.profiler import Profiler
from manabot.policy.buckets import AXES
from manabot.policy.protocol import BackendPolicy
from manabot.sim.rollout import (
    BatchedSampler,
    RandomBatchController,
)
from manabot.training.decks import UniformDecks
from manabot.training.learning import BatchClosure
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
from manabot.training.rollout.numerics import CollectionScope, CollectionSession
from manabot.training.rollout.producers import BufferedStreams, open_native_streams
from manabot.training.rollout.self_play import GameSetting, SelfPlayCounter
from manabot.training.rollout.snapshot_vector import (
    SlotInstall,
    SnapshotBranch,
    SnapshotVector,
    SnapshotVectorFactory,
)
from manabot.training.rollout.snapshots import (
    BatchPlan,
    CaptureDecision,
    CapturePredicate,
    SnapshotBank,
)
from manabot.training.state.game_errors import (
    GameAttempt,
    GameErrorLimit,
    GameErrorPolicy,
    GameErrorState,
    GameOutcome,
    RecordAttempt,
)
from managym import WORLD_SETUP, Deck, GameSetup, PlayerConfig

OPPONENT_MODES = ("random", "frozen", "self", "learning")

# -----------------------------------------------------------------------------
# Seat-routed rollout collection
# -----------------------------------------------------------------------------


class ObservationRows:
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
    learner regroups them instead of measuring the rows again. Opponent memory
    is retained only as immutable prefix references in the collector's shared
    MemoryStore; it never enters these dense per-update arrays.
    """

    def __init__(
        self, buffers: ObservationBuffers, keys: list[str], capacity: int
    ) -> None:
        self.buffers = buffers
        self._keys = [key for key in keys if key not in MEMORY_FIELDS]
        self.memory: list[Prefix] = []
        self.memory_capacities = (
            tuple(buffers[key].shape[1] for key in MEMORY_FIELDS)
            if "decision_records" in buffers
            else None
        )
        self.rows = 0
        self.capacity = max(capacity, 1)
        self._arrays = self._allocate(self.capacity)
        self._needs = np.full((self.capacity, len(AXES)), -1, np.int64)

    def _allocate(self, capacity: int) -> dict[str, np.ndarray]:
        # Untouched np.empty pages are not resident, so spare capacity is cheap.
        return {
            key: np.empty(
                (capacity, *self.buffers[key].shape[1:]),
                dtype=self.buffers[key].dtype,
            )
            for key in self._keys
        }

    def append(
        self,
        rows: np.ndarray,
        needs: np.ndarray | None = None,
        *,
        prefixes: tuple[Prefix, ...] = (),
    ) -> int:
        """Copy buffer `rows` (and their bucket `needs`, when measured) to the end
        of the store; return the first store row."""
        if self.memory_capacities is not None:
            if len(prefixes) != len(rows):
                raise ValueError("missing decision-memory prefix references")
            self.memory.extend(prefixes)
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
            np.take(self.buffers[key], rows, axis=0, out=array[start:stop], mode="clip")
        self.rows = stop
        return start

    def arrays(self) -> dict[str, np.ndarray]:
        """Used rows as views; rows of discarded failed games stay unreferenced."""
        return {key: array[: self.rows] for key, array in self._arrays.items()}

    def needs(self) -> np.ndarray:
        """Used rows' bucket needs [rows, len(AXES)], -1 where unmeasured."""
        return self._needs[: self.rows]


@dataclass
class _EpisodeAssignment:
    """The branch this stream installed, or its ordinary fresh game."""

    branch: SnapshotBranch | None = None
    generation: int | None = None
    ordinal: int = -1
    choices: int = 0
    fresh_ready: bool = False


@dataclass
class _SnapshotRollout:
    """Only the current batch owns its frozen horizon views and start sequence."""

    starts: Iterator[SnapshotBranch | None]
    ready: list[int]
    plan: BatchPlan | None = None
    frozen: ObservationRows | None = None


@dataclass
class _Pending:
    """A decision awaiting its reward at the same seat's next turn or game end."""

    obs_row: int  # Row in this update's ObservationRows.
    action: int
    logprob: float
    value: float
    action_data: PackedActions
    outcome_probabilities: np.ndarray | None = None
    assignment: tuple[int, int, int, int] | None = None
    branch: SnapshotBranch | None = None
    is_root: bool = False
    episode_ordinal: int = -1


@dataclass
class _Transition:
    """Finalized learner row with its collection-time outcome distribution."""

    pending: _Pending
    reward: float
    done: bool
    bootstrap_index: int | None = None


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
    memory: tuple[Prefix, ...] = ()
    memory_capacities: tuple[int, int, int, int] | None = None

    def gather(self, indexes: np.ndarray) -> dict[str, np.ndarray]:
        """Rows for flat time-major rollout `indexes`, as [batch, ...]."""
        recorded = self.index[indexes]
        rows = {key: value[recorded] for key, value in self.rows.items()}
        if self.memory_capacities is not None:
            return PrefixObservation(
                rows, [self.memory[int(i)] for i in recorded], self.memory_capacities
            )
        return rows

    def gather_needs(self, indexes: np.ndarray) -> np.ndarray | None:
        """Bucket needs of rollout `indexes` measured at collection, or None if
        any of those rows has none."""
        if self.needs is None:
            return None
        needs = self.needs[self.index[indexes]]
        return needs if bool((needs >= 0).all()) else None

    def ordered(self, shape: tuple[int, ...]) -> dict[str, np.ndarray]:
        """Ordinary fields in trajectory order; memory requires selected indexes."""
        return {
            key: array[self.index].reshape(*shape, *array.shape[1:])
            for key, array in self.rows.items()
        }


@dataclass(frozen=True)
class TrajectorySpan:
    """One seat's contiguous rows [start, stop) in a packed column.

    An unfinished span indexes its viewer's frozen observation in next_obs.
    """

    start: int
    stop: int
    seat: int
    bootstrap_index: int | None = None


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
    next_obs: ObservationBuffers
    next_done: np.ndarray
    action_data: PackedActions
    # Collection-time loss/draw/win probabilities [T,E,3], absent for scalar.
    outcome_probabilities: np.ndarray | None = None
    assignments: np.ndarray | None = None
    target_residuals: np.ndarray | None = None
    trajectories: tuple[TrajectorySpan, ...] | None = None

    # Optional per-decision provenance. Fresh=0, snapshot=1, absent IDs=-1.
    episode_source: np.ndarray | None = None
    snapshot_id: np.ndarray | None = None
    branch_group_id: np.ndarray | None = None
    is_root_decision: np.ndarray | None = None
    sample_weight: np.ndarray | None = None

    def __post_init__(self) -> None:
        validate_rollout_metadata(self)

    def bootstrap_indexes(self) -> np.ndarray | None:
        """Frozen observation index at each explicit cut, -1 on every other row."""
        if self.trajectories is None:
            return None
        indexes = np.full(self.actions.shape, -1, dtype=np.intp)
        for span in self.trajectories:
            if span.bootstrap_index is not None:
                indexes[span.stop - 1, 0] = span.bootstrap_index
        return indexes

    def segment_ends(self) -> np.ndarray | None:
        """Explicit trace boundaries, or None for ordinary column trajectories."""
        if self.trajectories is None:
            return None
        ends = np.zeros_like(self.dones)
        for span in self.trajectories:
            ends[span.stop - 1, 0] = True
        return ends

    def trajectory_rows(self) -> Iterator[range]:
        """Flat time-major indexes per trajectory, including censored column tails."""
        if self.trajectories is not None:
            for span in self.trajectories:
                yield range(span.start, span.stop)
            return
        steps, columns = self.actions.shape
        for column in range(columns):
            start = 0
            for terminal in np.flatnonzero(self.dones[:, column]):
                stop = int(terminal) + 1
                yield range(start * columns + column, stop * columns + column, columns)
                start = stop
            if start < steps:
                yield range(start * columns + column, steps * columns + column, columns)

    def bootstrapped_tail_fraction(self) -> float:
        """Share of seat trajectories whose outcomes are still unknown."""
        cuts = (
            np.count_nonzero(~self.dones[-1])
            if self.trajectories is None
            else sum(span.bootstrap_index is not None for span in self.trajectories)
        )
        return float(cuts / (cuts + np.count_nonzero(self.dones)))

    def observations(self, indexes: np.ndarray | None = None) -> ActionRows:
        """Materialize only requested flat rollout rows on CPU.

        Explicit indexes return [batch,...]; omitting them restores
        [time,trajectory,...] for bounded inspection. Learners always select a
        minibatch. Object data and compact actions are the only retained owners.
        """
        if indexes is None and self.recorded.memory_capacities is not None:
            raise ValueError(
                "decision memory requires explicit selected rollout indexes"
            )
        trajectory_shape = self.actions.shape if indexes is None else None
        if indexes is None:
            indexes = np.arange(self.actions.size)
        rows = self.action_data.unpack(indexes)
        observed = self.recorded.gather(indexes)
        if isinstance(observed, PrefixObservation):
            observed.update(rows.observation)
            rows.observation = observed
        else:
            rows.observation.update(observed)
        if trajectory_shape is not None:
            rows.observation = {
                key: value.reshape(*trajectory_shape, *value.shape[1:])
                for key, value in rows.observation.items()
            }
            rows.log_probabilities = rows.log_probabilities.reshape(
                *trajectory_shape, -1
            )
        return rows

    def ordered_obs(self) -> ObservationBuffers:
        """Ordinary non-action fields in [time, trajectory, ...] order."""
        return self.recorded.ordered(self.actions.shape)


def validate_rollout_metadata(batch: RolloutBatch) -> None:
    """Verify the packed rows and frozen observations describe the same trajectories."""
    for name, dtype in (
        ("episode_source", np.int8),
        ("snapshot_id", np.int64),
        ("branch_group_id", np.int64),
        ("is_root_decision", np.bool_),
        ("sample_weight", np.float32),
    ):
        value = getattr(batch, name)
        if value is not None and (
            not isinstance(value, np.ndarray)
            or value.shape != batch.actions.shape
            or value.dtype != dtype
        ):
            raise ValueError(
                f"{name} must match actions.shape and have dtype {np.dtype(dtype)}"
            )
    if (
        batch.episode_source is not None
        and not np.isin(batch.episode_source, (0, 1)).all()
    ):
        raise ValueError("episode_source must be fresh (0) or snapshot (1)")
    for ids in (batch.snapshot_id, batch.branch_group_id):
        if ids is not None and (ids < -1).any():
            raise ValueError("absent provenance IDs use -1")
    weights = batch.sample_weight
    if weights is not None and (not np.isfinite(weights).all() or (weights < 0).any()):
        raise ValueError("sample_weight must be finite and nonnegative")
    if batch.trajectories is None:
        return
    if (
        batch.actions.ndim != 2
        or batch.actions.shape[1] != 1
        or not batch.actions.size
        or batch.dones.shape != batch.actions.shape
    ):
        raise ValueError("explicit trajectories require packed [N,1] actions and dones")
    if batch.dones.dtype != np.bool_ or batch.next_done.dtype != np.bool_:
        raise ValueError("packed terminal flags must be boolean")
    if batch.next_done.ndim != 1 or batch.next_done.any():
        raise ValueError("packed bootstrap observations must all be nonterminal")
    count = len(batch.next_done)
    if (count and not batch.next_obs) or any(
        len(field) != count for field in batch.next_obs.values()
    ):
        raise ValueError("packed bootstrap observation count differs from next_done")
    previous = 0
    used: set[int] = set()
    for span in batch.trajectories:
        if span.start != previous or not span.start < span.stop <= len(batch.actions):
            raise ValueError("trajectory spans must partition all packed rows in order")
        if span.seat not in (0, 1) or batch.dones[span.start : span.stop - 1].any():
            raise ValueError(
                "trajectory spans require one seat and cannot cross terminals"
            )
        for values in (batch.episode_source, batch.snapshot_id, batch.branch_group_id):
            if (
                values is not None
                and (values[span.start : span.stop] != values[span.start]).any()
            ):
                raise ValueError(
                    "trajectory provenance must be constant within each span"
                )
        if (
            batch.is_root_decision is not None
            and batch.is_root_decision[span.start + 1 : span.stop].any()
        ):
            raise ValueError("only the first decision of a trajectory can be a root")
        terminal = batch.dones[span.stop - 1, 0]
        index = span.bootstrap_index
        if terminal != (index is None):
            raise ValueError("only nonterminal spans require a bootstrap index")
        if index is not None:
            if not isinstance(index, (int, np.integer)) or not 0 <= index < count:
                raise ValueError("trajectory bootstrap index outside next_obs")
            used.add(index)
        previous = span.stop
    if previous != len(batch.actions) or len(used) != count:
        raise ValueError(
            "trajectory spans must cover every action and bootstrap observation"
        )


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


class Operation(Protocol):
    def progress(self, decisions: int) -> None: ...


class NoOperation:
    def progress(self, decisions: int) -> None:
        pass


class OperationScope(Protocol):
    """A named boundary in the caller's failure journal.

    `per_step` marks boundaries entered once per native step, which only
    detailed diagnostics journal. The yielded operation reports progress.
    """

    def __call__(
        self, name: str, *, per_step: bool = False
    ) -> AbstractContextManager[Operation]: ...


def no_operation(
    name: str, *, per_step: bool = False
) -> AbstractContextManager[Operation]:
    return nullcontext(NoOperation())


@dataclass
class SnapshotStats:
    """Work on continuations. Rows and captures count what survived collection."""

    starts: int = 0
    choices: int = 0
    rows: int = 0
    terminals: int = 0
    horizon_cuts: int = 0
    captures: int = 0
    groups_completed: int = 0
    groups_discarded: int = 0
    failed_branches: int = 0


@dataclass
class CollectorStats:
    """Cumulative accounting, including the opponent fingerprint."""

    micro_steps: int = 0
    # Stream slots left unstepped while another stream finished its batch share.
    idle_stream_steps: int = 0
    learner_transitions: int = 0
    opponent_decisions: int = 0
    games: int = 0
    skipped_decisions: int = 0  # Completed native games, including initial skips.
    priority_taps: int = 0  # Selected taps in spaces that also offer PassPriority.
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
    snapshot: SnapshotStats | None = None

    @property
    def fresh_learner_transitions(self) -> int:
        return self.learner_transitions - (self.snapshot.rows if self.snapshot else 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            **(
                {
                    "snapshot": asdict(self.snapshot),
                    "fresh_learner_transitions": self.fresh_learner_transitions,
                }
                if self.snapshot is not None
                else {}
            ),
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
    # Rows per policy forward during collection; None runs each group whole.
    execution_rows: int | None
    opponent_mode: str
    opponent_agent: BackendPolicy | None
    learning_seats: Literal["one", "both"]
    batch_closure: BatchClosure
    games_per_stream: int | None
    # The backend session owns the device, precision and private sampling RNG.
    numerics: CollectionSession
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
    # ShardedVectorEnv, RecordedVectorEnv or a diagnostic root's Python vector.
    _env: Any
    _buffers: dict[str, np.ndarray]
    _observation_keys: list[str]
    _observations: ObservationRows
    # Shared immutable prefix chunks of the opponent decision log; None when
    # the observation space has no decision-record capacity.
    _memory_store: MemoryStore | None
    _record_attempt: RecordAttempt | None
    # The deck source's identity is fixed; every game attempt records it.
    _decks_digest: str | None
    uniform_roster: list[PlayerConfig]
    _win_reward: float
    _lose_reward: float
    _streams: list[list[_Transition]]
    _pending: list[_Pending | None]
    _game_start_rows: list[int]
    _game_started: list[float]
    # Legal action prefix of each stream's game in progress, for failure receipts.
    _game_actions: list[list[int]]
    _self_play_decks: list[tuple[str, str]]
    # Each stream's game in progress: the conditions it is played under.
    _game_settings: list[tuple[GameSetting, ...]]
    # Scopes a named boundary (collection, learning, inference) in the caller's
    # failure journal; the step loop wraps collection and learning with it.
    operation: OperationScope = no_operation
    # Per-stream policy annotations for the recorded games of the coming step.
    _record_annotations: list[dict[str, object] | None] = field(default_factory=list)
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
    _episodes: list[_EpisodeAssignment] = field(default_factory=list)
    capture_predicate: CapturePredicate | None = None
    snapshot_bank: SnapshotBank | None = None


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
    execution_rows: int | None = None,
    seed: int = 0,
    opponent_mode: str = "random",
    numerics: CollectionSession,
    opponent_agent: BackendPolicy | None = None,
    learning_seats: Literal["one", "both"] | None = None,
    batch_closure: BatchClosure = "complete-games",
    games_per_stream: int | None = None,
    root: str | None = None,
    matchup_curriculum: Curriculum = "fixed",
    deck_source: UniformDecks | None = None,
    workers: int = 1,
    game_errors: GameErrorPolicy | None = None,
    record_attempt: RecordAttempt | None = None,
    adaptive: AdaptiveConfig | None = None,
    snapshot_vector: SnapshotVector | SnapshotVectorFactory | None = None,
    snapshot_bank: SnapshotBank | None = None,
    snapshots: bool = False,
    capture_predicate: CapturePredicate | None = None,
    operation: OperationScope = no_operation,
    games_path: Path | None = None,
    record_fraction: float = 1.0,
) -> Collector:
    """Start `num_envs` native streams at their first roots.

    ``learning_seats="both"`` is the default for ``self`` (one shared model)
    or ``learning`` (two distinct models). ``one`` retains the learner-only
    control, including in self-play. ``batch_closure`` says separately when a
    batch ends: after whole games, or at an exact one-seat quota that
    bootstraps its mid-game tail from the critic. `record_attempt` persists
    each settled game when `game_errors` is set. `numerics` is the backend's
    collection session: it owns the device, precision and the private sampling
    RNG for learner rows and current-self opponents. The caller closes the
    collector with `close_collector`.
    """
    snapshots = snapshots or snapshot_bank is not None or snapshot_vector is not None
    if snapshot_vector is not None and workers != 1:
        raise ValueError("a custom snapshot vector requires workers=1")
    if snapshots and (
        opponent_mode != "self"
        or learning_seats == "one"
        or adaptive is not None
        or root is not None
    ):
        raise ValueError(
            "snapshot collection requires both-seat self-play with ordinary roots and no adaptive sampler"
        )
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
    taps = observation_space.config.mana_floats
    payment = observation_space.config.mana_payment
    match = Match(
        match.config.model_copy(
            update={"mana_floats": (taps, taps), "mana_payment": (payment, payment)}
        )
    )
    sampler = AdaptiveSampler(adaptive) if adaptive is not None else None
    assignments = [sampler.draw() for _ in range(num_envs)] if sampler else []
    reverse_cells = _reverse_cells(adaptive) if adaptive is not None else []
    uniform_roster: list[PlayerConfig]
    if deck_source is not None:
        if matchup_curriculum != "fixed":
            raise ValueError("a deck source cannot be combined with a curriculum")
        if match.config.world != WORLD_SETUP:
            raise ValueError(
                "sampling decks requires a match configured for submitted decks"
            )
        if (
            deck_source.decklists != "mixed"
            and match.config.decklists != deck_source.decklists
        ):
            raise ValueError(
                "the deck source's disclosure decides every game: match has "
                f"decklists={match.config.decklists!r}, deck source has "
                f"decklists={deck_source.decklists!r}"
            )
        uniform_roster = deck_source.players()
    elif matchup_curriculum == "elemental-uniform":
        uniform_roster = elemental_roster(match)
    else:
        uniform_roster = []
    if uniform_roster and (root is not None or opponent_mode != "self"):
        raise ValueError(
            "per-game roster sampling requires current-self and ordinary roots"
        )
    if workers < 1 or workers > num_envs:
        raise ValueError("workers must be at least 1 and cannot exceed streams")
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

    if execution_rows is not None and execution_rows < 1:
        raise ValueError("execution_rows must be positive")
    env: Any
    if snapshot_vector is not None:
        env = (
            snapshot_vector(observation_space, match, num_envs, seed)
            if callable(snapshot_vector)
            and not isinstance(snapshot_vector, SnapshotVector)
            else snapshot_vector
        )
    elif root is None:
        # Native decision capture is opt-in from the observation's capacity.
        memory = observation_space.config
        capture_decisions = memory.max_decision_records > 0
        env = open_native_streams(
            num_envs,
            seed,
            workers,
            auto_reset=not snapshots,
            decision_memory=capture_decisions,
            decision_memory_include_own=memory.decision_memory_include_own,
            games_path=games_path,
            record_fraction=record_fraction,
        )
    else:
        from manabot.env.target_practice import TargetPracticeVector

        if root != "lethal-target-v1":
            raise ValueError("unsupported training root")
        env = TargetPracticeVector(observation_space, match, num_envs, seed)
    try:
        if snapshots:
            if not isinstance(env, SnapshotVector):
                raise ValueError("snapshot collection requires a snapshot vector")
            if not env.skip_trivial:
                raise ValueError(
                    "snapshot self-play requires native trivial-decision skipping"
                )
        buffers = allocate_vector_buffers(observation_space, num_envs)
        slots = num_envs * (2 if learning_seats == "both" else 1)
        observation_keys = [
            key for key in observation_space.shapes if key not in ACTION_FIELDS
        ]
        observations = ObservationRows(buffers, observation_keys, 0)
        collector = Collector(
            observation_space=observation_space,
            match=match,
            reward=reward,
            num_envs=num_envs,
            execution_rows=execution_rows,
            opponent_mode=opponent_mode,
            opponent_agent=opponent_agent,
            learning_seats=learning_seats,
            batch_closure=batch_closure,
            games_per_stream=games_per_stream,
            numerics=numerics,
            game_errors=game_errors,
            learner_seat=np.arange(num_envs, dtype=np.int64) % 2,
            stream_matches=[match] * num_envs,
            stats=CollectorStats(snapshot=SnapshotStats() if snapshots else None),
            snapshot_bank=snapshot_bank,
            game_error_state=GameErrorState(),
            profiler=Profiler(enabled=False),
            _env=env,
            _buffers=buffers,
            _observation_keys=observation_keys,
            _observations=observations,
            _memory_store=(
                MemoryStore(observations.memory_capacities)
                if observations.memory_capacities is not None
                else None
            ),
            _record_attempt=record_attempt,
            _decks_digest=deck_source.digest if deck_source is not None else None,
            uniform_roster=uniform_roster,
            _win_reward=float(reward.config.win_reward),
            _lose_reward=float(reward.config.lose_reward),
            _streams=[[] for _ in range(slots)],
            _pending=[None] * slots,
            _game_start_rows=[0] * slots,
            _game_started=[time.perf_counter()] * num_envs,
            _game_actions=[[] for _ in range(num_envs)],
            _self_play_decks=[("", "")] * num_envs,
            _game_settings=[()] * num_envs,
            operation=operation,
            _record_annotations=[None] * num_envs,
            _sampler=sampler,
            _assignments=assignments,
            _ordinals=[0] * num_envs,
            _reverse_cells=reverse_cells,
            capture_predicate=capture_predicate,
            _episodes=[_EpisodeAssignment(fresh_ready=True) for _ in range(num_envs)]
            if snapshots
            else [],
        )
        env.set_buffers(
            buffers,
            action_overflow=observation_space.config.action_overflow,
            layout=observation_space.layout,
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
            setup = match.to_rust()
            if deck_source is not None:
                setup = GameSetup(setup, decklists=deck_source.decklists)
            env.reset_all_into_buffers(setup)
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
                opponent_agent,
                deterministic=False,
                seed=seed + 1,
                device=numerics.device_name,
                execution_rows=execution_rows,
            )
        if not snapshots:
            for row in range(num_envs):
                _expose(collector, row, "game_roots")
    except BaseException:
        _close_env(env)
        raise
    return collector


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if close is not None:
        close()


def close_collector(collector: Collector) -> None:
    """Release cached snapshot handles before joining the owned vector producers."""
    if collector.snapshot_bank is not None:
        collector.snapshot_bank.close()
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
    if collector.uniform_roster:
        # Actual native setups, not indices: they remain exact for any deck
        # the run submitted, even one outside the supply it began with.
        setups = collector._env.current_player_configs()
        for row in rows:
            collector.stream_matches[row] = Match(
                MatchConfig.from_players(
                    *setups[row],
                    decklists=setups[row].decklists,
                    mana_floats=setups[row].mana_floats,
                    mana_payment=setups[row].mana_payment,
                )
            )
    # The environment that plays the games states whether it skips forced decisions.
    skip_trivial = "true" if collector._env.skip_trivial else "false"
    for row in rows:
        match = collector.stream_matches[row]
        collector._self_play_decks[row] = (
            Deck(match.hero_deck, match.hero_sideboard).identity,
            Deck(match.villain_deck, match.villain_sideboard).identity,
        )
        collector._game_settings[row] = (
            GameSetting(None, "decklists", match.config.decklists),
            GameSetting(None, "skip_trivial", skip_trivial),
        )


def _expose(collector: Collector, row: int, kind: ExposureKind, count: int = 1) -> None:
    if collector._episodes and collector._episodes[row].branch is not None:
        return
    if not collector.uniform_roster and collector._sampler is None:
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
    collector: Collector, rows: np.ndarray, inference: CollectionScope
) -> np.ndarray:
    with collector.operation("opponent_inference", per_step=True):
        if collector.opponent_mode == "self":
            return inference.opponent_actions(
                _retain_obs_rows(collector, rows),
                execution_rows=collector.execution_rows,
            )
        if collector.opponent_mode == "random":
            return collector._opponent.select(collector._buffers, rows)
        return collector._opponent.select(
            _retain_obs_rows(collector, rows), np.arange(len(rows))
        )


def _capture_memory(
    collector: Collector,
    rows: np.ndarray,
    buffers: ObservationBuffers,
    players: Sequence[int | None] | None = None,
) -> tuple[Prefix, ...]:
    if collector._memory_store is None:
        return ()
    with collector.profiler.track("Memory capture"):
        env = collector._env
        return collector._memory_store.capture_live_rows(
            buffers,
            rows,
            env.current_game_seeds(),
            env.current_agent_indices()
            if players is None
            else [p if p is not None else -1 for p in players],
            env.current_game_generations(),
            env.decision_memory_cursors(),
        )


def _retain_obs_rows(collector: Collector, rows: np.ndarray) -> ObservationBuffers:
    """Freeze bootstrap fields and prefix versions without padded memory."""
    with collector.profiler.track("Observation retention"):
        obs = {
            key: collector._buffers[key][rows]
            for key in collector.observation_space.shapes
            if key not in MEMORY_FIELDS
        }
        if collector._memory_store is not None:
            obs = PrefixObservation(
                obs,
                _capture_memory(collector, rows, collector._buffers),
                collector._memory_store.capacities,
            )
        # Training convention (VectorEnv): action_focus as float32.
        obs["action_focus"] = obs["action_focus"].astype(np.float32)
        return obs


def _count_action_types(
    collector: Collector, counter: dict[str, int], rows: np.ndarray, actions: np.ndarray
) -> None:
    chosen = collector._buffers["actions"][rows, actions, :ACTION_TYPE_DIM]
    type_indices = chosen.argmax(axis=1)
    # At priority all additional mana abilities, fixed and choice-output,
    # follow PassPriority. The first-legal policy projection preserves order.
    floats = (chosen[:, ActionEnum.TAP_FOR_COST] > 0) | (
        chosen[:, ActionEnum.PRIORITY_ACTIVATE_ABILITY] > 0
    )
    if floats.any():
        passes = collector._buffers["actions"][
            rows[floats], :, ActionEnum.PRIORITY_PASS_PRIORITY
        ]
        collector.stats.priority_taps += int(
            (passes.any(axis=1) & (actions[floats] > passes.argmax(axis=1))).sum()
        )
    typed = chosen.max(axis=1) > 0
    for idx, has_type in zip(type_indices, typed):
        name = _ACTION_TYPE_NAMES.get(int(idx), "unknown") if has_type else "unknown"
        counter[name] = counter.get(name, 0) + 1


# -----------------------------------------------------------------------------
# Collection
# -----------------------------------------------------------------------------


def collect(
    collector: Collector,
    agent: BackendPolicy,
    num_steps: int,
    *,
    snapshot_starts: Iterable[SnapshotBranch | None] | None = None,
    snapshot_plan: BatchPlan | None = None,
    deadline_monotonic: float | None = None,
    check: Callable[[], None] | None = None,
) -> RolloutBatch:
    """Collect a batch closed at an exact quota or after complete games.

    Exact quota pauses each stream at num_steps learner rows, and its game
    continues in the next batch. Complete games treat num_steps as a floor on
    the stream's retained decisions, the total across seats when both learn;
    games_per_stream replaces the floor with that many whole games, and no
    sampled action crosses the update boundary. For a snapshot vector, consume
    snapshot_starts in stream order at each allocation boundary. None or an
    exhausted sequence starts a fresh game. A bank plan instead schedules starts
    and exposes its scoring receipt at snapshot_plan.receipt on success.
    """

    if collector.opponent_mode == "learning":
        raise ValueError("use collect_pair for two learning models")
    # No cache or prepared observation survives learning.
    # Both seats in self-play use this same unchanged behavior-policy scope.
    with (
        collector.profiler.track("Collection"),
        collector.numerics.scope(agent) as inference,
    ):
        batch = _collect(
            collector,
            num_steps,
            inference,
            snapshot_starts=snapshot_starts,
            snapshot_plan=snapshot_plan,
            deadline_monotonic=deadline_monotonic,
            check=check,
        )
        assert isinstance(batch, RolloutBatch)
        return batch


def collect_pair(
    collector: Collector,
    agent: BackendPolicy,
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
        collector.numerics.scope(agent) as inference,
        collector.numerics.scope(collector.opponent_agent) as opponent_inference,
    ):
        batches = _collect(
            collector,
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
    num_steps: int,
    inference: CollectionScope,
    *,
    deadline_monotonic: float | None,
    check: Callable[[], None] | None,
    snapshot_starts: Iterable[SnapshotBranch | None] | None = None,
    snapshot_plan: BatchPlan | None = None,
    opponent_inference: CollectionScope | None = None,
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
        collector._observations = ObservationRows(
            collector._buffers, collector._observation_keys, rows + rows // 4
        )
    if snapshot_plan is not None:
        if snapshot_starts is not None or not collector._episodes:
            raise ValueError(
                "snapshot plan requires a snapshot vector and owns episode starts"
            )
        if snapshot_plan.closed or snapshot_plan.vector is not None:
            raise ValueError("snapshot plan already collected")
        snapshot_plan.vector = env
    snapshots = None
    if collector._episodes:
        snapshots = _SnapshotRollout(
            iter(snapshot_starts if snapshot_starts is not None else ()),
            list(range(num_envs)),
            plan=snapshot_plan,
        )
    elif snapshot_starts is not None:
        raise ValueError("snapshot starts require a snapshot vector")
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
            if snapshots is not None and snapshots.ready:
                if _start_episodes(collector, snapshots, games, completed, num_steps):
                    continue
            acting = np.asarray(env.current_agent_indices(), dtype=np.int64)
            for row in range(num_envs):
                if completed[row] or acting[row] < 0:
                    continue
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
        collector._record_annotations = [None] * num_envs
        actions = np.zeros(num_envs, dtype=np.int64)
        learner_rows = np.flatnonzero(active & (acting == collector.learner_seat))
        opp_rows = np.flatnonzero(active & (acting != collector.learner_seat))

        if both and collector.opponent_mode == "self":
            learner_rows = np.flatnonzero(active)
            opp_rows = np.array([], dtype=np.int64)
        if len(learner_rows):
            actions[learner_rows] = _sample_learning_rows(
                collector, learner_rows, acting, inference
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
                    assert opponent_inference is not None
                    opp_acts = _sample_learning_rows(
                        collector,
                        opp_rows,
                        acting,
                        opponent_inference,
                        "opponent_inference",
                    )
                else:
                    opp_acts = _opponent_actions(collector, opp_rows, inference)
            actions[opp_rows] = opp_acts
            if not both:
                collector.stats.opponent_decisions += len(opp_rows)
            _count_action_types(
                collector, collector.stats.opponent_action_types, opp_rows, opp_acts
            )

        if snapshot_plan is not None:
            snapshot_plan.capture(
                _capture_decisions(collector, learner_rows, acting),
                collector.capture_predicate,
            )

        for row in np.flatnonzero(active):
            _expose(collector, int(row), "environment_decisions")
        seeds = env.current_game_seeds() if collector.game_errors else []
        annotations = (
            {"annotations": collector._record_annotations}
            if any(collector._record_annotations)
            else {}
        )
        with (
            collector.profiler.track("Environment step and encoding"),
            _native_step_split(collector),
        ):
            if collector.game_errors:
                env.step_isolated_into_buffers(
                    actions.tolist(),
                    active.tolist(),
                    collector.game_errors.recover_capacity,
                    **annotations,
                )
            else:
                env.step_into_buffers(actions.tolist(), active.tolist(), **annotations)
        collector.stats.micro_steps += int(active.sum())

        with collector.profiler.track("Stream bookkeeping"):
            _settle_step(
                collector,
                actions,
                active,
                acting,
                seeds,
                games,
                completed,
                num_steps,
                snapshots,
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
                    collector, list(range(len(collector._streams))), snapshots
                )
        else:
            batch = _build_rectangular_batch(collector, num_steps)
    if snapshot_plan is not None:
        stats, receipt = collector.stats.snapshot, snapshot_plan.receipt
        assert stats is not None and receipt is not None
        stats.captures += len(receipt.captures)
        stats.groups_completed += len(receipt.visited_groups)
        stats.groups_discarded += len(snapshot_plan.discarded)
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
    config = collector._sampler.config.cells[assignment.cell]
    match = Match(
        config.model_copy(
            update={
                "mana_floats": collector.match.config.mana_floats,
                "mana_payment": collector.match.config.mana_payment,
            }
        )
    )
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


def _install_roots(
    collector: Collector, requests: list[SlotInstall], plan: BatchPlan | None
) -> dict[int, int]:
    """Publish new generations only after verifying every requested root binding."""
    if not requests:
        return {}
    vector = collector._env
    before = vector.current_game_generations()
    installed = (
        vector.install_slots(requests)
        if plan is None
        else plan.bank.install(vector, requests)
    )
    generations = vector.current_game_generations()
    choosers = vector.current_agent_indices()
    seeds = vector.current_game_seeds()
    if len(installed) != len(requests):
        raise ValueError("snapshot vector did not return every installed slot")
    for request, result in zip(requests, installed, strict=True):
        slot = request.slot
        if (result.slot, result.root_chooser) != (slot, choosers[slot]) or before[
            slot
        ] == generations[slot]:
            raise ValueError(
                f"snapshot {request.snapshot_id} did not install a new game in slot {slot}"
            )
        recipe = request.recipe
        forced = (
            request.restore_options is not None
            and request.restore_options.forced_first_action is not None
        )
        terminal = bool(collector._buffers["terminated"][slot])
        if terminal != (choosers[slot] == -1) or (terminal and not forced):
            raise ValueError(
                f"snapshot {request.snapshot_id} has invalid installed terminal state"
            )
        if recipe is not None and (
            result.root_digest != recipe.root_digest
            or seeds[slot] != recipe.seed
            or (not forced and choosers[slot] != recipe.root_chooser)
        ):
            raise ValueError(
                f"snapshot {request.snapshot_id} root identity mismatch in slot {slot}"
            )
    return {request.slot: generations[request.slot] for request in requests}


def _capture_decisions(
    collector: Collector, rows: np.ndarray, acting: np.ndarray
) -> Iterator[CaptureDecision]:
    # Native legality can exceed the learned action slots under first-legal.
    native_counts = (
        collector._env.current_action_summaries()
        if collector.observation_space.config.action_overflow == "first-legal"
        else None
    )
    for row in rows:
        slot, seat = int(row), int(acting[row])
        episode = collector._episodes[slot]
        pending = collector._pending[2 * slot + seat]
        assert pending is not None
        yield CaptureDecision(
            slot,
            episode.choices + 1,
            native_counts[slot][0]
            if native_counts is not None
            else len(pending.action_data.slots),
            seat,
            episode.branch,
            episode.ordinal,
            pending.obs_row,
        )


def _discard_group_rows(collector: Collector, groups: set[int]) -> None:
    """Discard closed siblings. A live sibling keeps its failure evidence until it ends."""
    assert collector.stats.snapshot is not None
    for slot, stream in enumerate(collector._streams):
        boundary = collector._game_start_rows[slot]
        kept = [
            t
            for t in stream[:boundary]
            if t.pending.branch is None
            or t.pending.branch.branch_group_id not in groups
        ]
        stream[:boundary] = kept
        collector._game_start_rows[slot] = len(kept)
        discarded = boundary - len(kept)
        collector.stats.learner_transitions -= discarded
        collector.stats.snapshot.rows -= discarded


def _start_episodes(
    collector: Collector,
    snapshots: _SnapshotRollout,
    games: np.ndarray,
    completed: np.ndarray,
    num_steps: int,
) -> bool:
    """Allocate ready slots. Return True if a forced terminal needs another allocation."""
    plan = snapshots.plan
    starts: dict[int, _EpisodeAssignment] = {}
    parked = []
    if plan is None:
        for slot in snapshots.ready:
            starts[slot] = _EpisodeAssignment(next(snapshots.starts, None))
    else:
        if collector.games_per_stream is not None:
            at_floor = (
                plan.completed_episodes
                >= collector.games_per_stream * collector.num_envs
            )
        else:
            rows = sum(map(len, collector._streams)) + sum(
                p is not None for p in collector._pending
            )
            at_floor = rows >= num_steps * collector.num_envs
        for slot in sorted(snapshots.ready):
            completed[slot] = at_floor and not plan.pending
            if completed[slot]:
                parked.append(slot)
            else:
                ordinal = plan.scheduler.starts
                starts[slot] = _EpisodeAssignment(plan.next_episode(), ordinal=ordinal)
    snapshots.ready = parked
    if not starts:
        return False
    installs = []
    for slot, episode in starts.items():
        branch = episode.branch
        if branch is not None:
            installs.append(
                SlotInstall(
                    slot,
                    branch.snapshot_id,
                    branch.recipe,
                    branch.hot_handle,
                    branch.restore_options,
                )
            )
        elif not collector._episodes[slot].fresh_ready:
            installs.append(SlotInstall(slot, -1, None))
    generations = _install_roots(collector, installs, plan)
    assert collector.stats.snapshot is not None
    for slot, episode in starts.items():
        episode.generation = generations.get(slot)
        collector._episodes[slot] = episode
        collector.stats.snapshot.starts += int(episode.branch is not None)
    _refresh_roots(collector, np.asarray(list(starts), dtype=np.intp))
    for slot, episode in starts.items():
        _expose(collector, slot, "game_roots")
        if episode.branch is None or not collector._buffers["terminated"][slot]:
            continue
        # A forced opening committed inside restore, before either seat sampled.
        assert plan is None
        collector.stats.snapshot.terminals += 1
        branch = episode.branch
        if collector.game_errors:
            _settle_attempt(
                collector,
                slot,
                branch.recipe.seed,
                branch.recipe.root_chooser,
                "completed",
            )
        games[slot] += 1
        completed[slot] = _stream_at_floor(collector, slot, games, num_steps)
        if not completed[slot]:
            snapshots.ready.append(slot)
    return plan is None and bool(snapshots.ready)


def _close_horizons(
    collector: Collector, snapshots: _SnapshotRollout, cuts: list[int]
) -> None:
    """Freeze all contributing viewers before releasing any stream at this boundary."""
    env = collector._env
    space = collector.observation_space
    if snapshots.frozen is None:
        scratch = space.allocate(collector.num_envs)
        snapshots.frozen = ObservationRows(scratch, list(scratch), 0)
    frozen = snapshots.frozen
    # Native reads project one viewer per stream. One pass suffices when every
    # cut has a single contributor, even if they acted from different seats.
    viewers = {
        row: [seat for seat in (0, 1) if collector._pending[2 * row + seat] is not None]
        for row in cuts
    }
    revisions: dict[int, int] = {}
    for index in range(max(map(len, viewers.values()))):
        rows = np.array(
            [row for row, seats in viewers.items() if index < len(seats)], dtype=np.intp
        )
        players: list[int | None] = [None] * collector.num_envs
        for row in rows:
            players[row] = viewers[row][index]
        reads = env.write_observations_for_players(
            players,
            frozen.buffers,
            layout=space.layout,
            action_overflow=space.config.action_overflow,
        )
        if len(reads) != len(rows):
            raise ValueError("missing snapshot horizon read")
        for row, read in zip(rows, reads, strict=True):
            episode = collector._episodes[row]
            assert episode.branch is not None
            if (
                read.slot != row
                or read.player != players[row]
                or read.game_id != episode.generation
                or read.seed != episode.branch.recipe.seed
                or read.revision != revisions.setdefault(int(row), read.revision)
            ):
                raise ValueError(
                    f"snapshot horizon read identity mismatch in slot {row}"
                )
        first = frozen.append(
            rows,
            prefixes=_capture_memory(collector, rows, frozen.buffers, players),
        )
        for offset, row in enumerate(rows):
            _finalize(
                collector,
                2 * int(row) + viewers[row][index],
                reward=0.0,
                done=False,
                bootstrap_index=first + offset,
            )
    if collector.game_errors:
        for row in cuts:
            collector._game_actions[row].clear()
            collector._game_started[row] = time.perf_counter()
            for slot in _game_slots(collector, row):
                collector._game_start_rows[slot] = len(collector._streams[slot])


def _settle_step(
    collector: Collector,
    actions: np.ndarray,
    active: np.ndarray,
    acting: np.ndarray,
    seeds: list[int],
    games: np.ndarray,
    completed: np.ndarray,
    num_steps: int,
    snapshots: _SnapshotRollout | None,
) -> None:
    """Record what one native step ended; count closure in `games`/`completed`."""
    buffers = collector._buffers
    env = collector._env
    both = collector.learning_seats == "both"
    complete_games = collector.batch_closure == "complete-games"
    done = active & ((buffers["terminated"] > 0) | (buffers["truncated"] > 0))
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
        fresh = snapshots is None or collector._episodes[row].branch is None
        if fresh:
            collector.stats.skipped_decisions += infos[row].get("skipped_decisions", 0)
            collector.stats.games += 1
            collector.stats.games_by_stream[int(row)] = (
                collector.stats.games_by_stream.get(int(row), 0) + 1
            )
        if buffers["truncated"][row]:
            collector.stats.truncations += 1
            _expose(collector, int(row), "truncated_games")
            raise RuntimeError("collector game truncated; no terminal target available")
        if fresh and collector.opponent_mode in ("self", "learning"):
            collector.stats.self_play.record(
                collector._self_play_decks[int(row)],
                winner,
                collector._game_settings[int(row)],
            )
        if winner is None:
            reward = 0.0
        elif winner == int(collector.learner_seat[row]):
            collector.stats.learner_wins += int(fresh)
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
    cuts: list[int] = []
    if snapshots is not None:
        stats = collector.stats.snapshot
        assert stats is not None
        for row in np.flatnonzero(active):
            episode = collector._episodes[row]
            branch = episode.branch
            if branch is not None:
                stats.choices += 1
                stats.failed_branches += int(failed[row])
                stats.terminals += int(done[row])
            if failed[row]:
                continue
            episode.choices += 1
            if branch and not done[row] and branch.horizon == episode.choices:
                cuts.append(int(row))
    # Deferred slots keep their scheduler allocation until the next start.
    if collector.game_error_state.stop_reason:
        raise GameErrorLimit(collector.game_error_state.stop_reason)
    auto_reset = getattr(env, "auto_reset", True)
    if replace.any() and auto_reset:
        env.replace_failed_into_buffers(replace.tolist())
    closed = done
    if cuts:
        assert snapshots is not None and collector.stats.snapshot is not None
        _close_horizons(collector, snapshots, cuts)
        collector.stats.snapshot.horizon_cuts += len(cuts)
        closed = done.copy()
        closed[cuts] = True
    # The bank closes the whole batch at allocation time so in-flight siblings drain.
    if complete_games and (snapshots is None or snapshots.plan is None):
        for row in np.flatnonzero(closed):
            games[row] += 1
            completed[row] = _stream_at_floor(collector, int(row), games, num_steps)

    ended = np.flatnonzero(closed | failed)
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
    if snapshots is not None:
        for row in ended:
            episode = collector._episodes[row]
            episode.fresh_ready = auto_reset and episode.branch is None
        plan = snapshots.plan
        if plan is None:
            snapshots.ready.extend(int(row) for row in ended if not completed[row])
        else:
            for row in np.flatnonzero(closed):
                plan.succeed(collector._episodes[row].branch)
            for row in np.flatnonzero(failed):
                episode = collector._episodes[row]
                plan.discard_captures(episode.ordinal)
                if episode.branch is not None:
                    plan.fail(episode.branch)
            if plan.discarded and any(
                (b := collector._episodes[row].branch) is not None
                and b.branch_group_id in plan.discarded
                for row in ended
            ):
                _discard_group_rows(collector, plan.discarded)
            snapshots.ready.extend(map(int, ended))
    elif ended.size:
        _refresh_roots(collector, ended)
        if collector._sampler is None:
            for row in ended:
                _expose(collector, int(row), "game_roots")


def _stream_at_floor(
    collector: Collector, row: int, games: np.ndarray, num_steps: int
) -> bool:
    if collector.games_per_stream is not None:
        return bool(games[row] >= collector.games_per_stream)
    return (
        sum(len(collector._streams[slot]) for slot in _game_slots(collector, row))
        >= num_steps
    )


@contextmanager
def _native_step_split(collector: Collector) -> Iterator[None]:
    """Split the running step phase by the engine's clock; Python envs have none."""
    if not isinstance(collector._env, (BufferedStreams, RecordedVectorEnv)):
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
    branch = collector._episodes[row].branch if collector._episodes else None
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
            if branch is not None:
                assert collector.stats.snapshot is not None
                collector.stats.snapshot.rows -= count
            _expose(collector, row, "learner_transitions", -count)
            del collector._streams[slot][collector._game_start_rows[slot] :]
            collector._pending[slot] = None
        collector._game_start_rows[slot] = len(collector._streams[slot])
    actions: list[int | str] = [] if completed else list(collector._game_actions[row])
    inputs = None
    if branch is not None and not completed:
        if not getattr(collector._env, "auto_reset", True):
            inputs = collector._env.game_inputs(row)
            actions = inputs.actions
        else:
            actions = [*branch.recipe.actions, *actions]
    config = collector.observation_space.config
    observation = (
        config
        if outcome == "resource_limit" or config.decision_memory_include_own
        else None
    )
    # A restore may enable/disable capture at its boundary. Keep the reset
    # setting when the receipt's encoder configuration would infer another.
    initial_memory = (
        inputs.decision_memory
        if inputs is not None
        and inputs.decision_memory
        != bool(observation and observation.max_decision_records)
        else None
    )
    collector.game_error_state = collector._record_attempt(
        collector.game_error_state,
        GameAttempt(
            stream=row,
            seed=seed,
            seat=seat,
            matchup=branch.recipe.match
            if branch is not None
            else collector.stream_matches[row].config,
            decks_digest=collector._decks_digest,
            outcome=outcome,
            actions=actions,
            decision_memory=initial_memory,
            interventions=inputs.interventions if inputs is not None else [],
            object_candidate_frames=inputs.object_candidate_frames
            if inputs is not None
            else [],
            error_class=error_class,
            error=error,
            frame_json=frame_json,
            native_trace=native_trace,
            observation=observation,
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
    rows: np.ndarray,
    acting: np.ndarray,
    inference: CollectionScope,
    operation: str = "learner_inference",
) -> np.ndarray:
    with collector.operation(operation, per_step=True):
        observation = _retain_obs_rows(collector, rows)
        sampled = inference.sample(
            observation, collector.profiler, execution_rows=collector.execution_rows
        )
    acts, logprobs, values = sampled.actions, sampled.logprobs, sampled.values
    outcomes, needs = sampled.outcomes, sampled.needs
    with collector.profiler.track("Rollout recording"):
        first = collector._observations.append(
            rows,
            needs,
            prefixes=observation.prefixes
            if isinstance(observation, PrefixObservation)
            else (),
        )
        packed = PackedActions.pack(
            {
                key: collector._buffers[key][rows]
                for key in ACTION_FIELDS
                if key in collector._buffers
            },
            sampled.log_probabilities,
            validator=collector.numerics,
        )
        env = collector._env
        recording = (
            env.recording_mask()
            if hasattr(env, "recording_mask")
            else [False] * len(collector._record_annotations)
        )
        for j, row in enumerate(rows):
            if recording[int(row)]:
                valid = np.asarray(collector._buffers["actions_valid"][row], dtype=bool)
                collector._record_annotations[int(row)] = {
                    "value": float(values[j]),
                    "probabilities": np.exp(
                        sampled.log_probabilities[j][valid]
                    ).tolist(),
                }
            slot = (
                2 * int(row) + int(acting[row])
                if collector.learning_seats == "both"
                else int(row)
            )
            episode = collector._episodes[row] if collector._episodes else None
            branch = episode.branch if episode else None
            collector._pending[slot] = _Pending(
                assignment=(
                    _role_cell(collector, int(row), int(acting[row])),
                    int(acting[row]),
                    int(row),
                    collector._ordinals[row],
                )
                if collector._sampler is not None
                else None,
                branch=branch,
                episode_ordinal=episode.ordinal if episode else -1,
                is_root=episode is not None
                and branch is not None
                and episode.choices == 0,
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


def _finalize(
    collector: Collector,
    row: int,
    *,
    reward: float,
    done: bool,
    bootstrap_index: int | None = None,
) -> None:
    pending = collector._pending[row]
    assert pending is not None
    collector._streams[row].append(_Transition(pending, reward, done, bootstrap_index))
    collector._pending[row] = None
    collector.stats.learner_transitions += 1
    if pending.branch is not None:
        assert collector.stats.snapshot is not None
        collector.stats.snapshot.rows += 1
    stream = row // 2 if collector.learning_seats == "both" else row
    _expose(collector, stream, "learner_transitions")


def _build_rectangular_batch(collector: Collector, steps: int) -> RolloutBatch:
    streams = collector._streams
    assert all(len(stream) == steps for stream in streams)
    assert all(row is None for row in collector._pending)
    rows = [row for timestep in zip(*streams, strict=True) for row in timestep]
    batch = _build_batch(collector, rows, (steps, len(streams)))
    for stream in streams:
        stream.clear()
    return batch


def _build_complete_batch(
    collector: Collector, slots: list[int], snapshots: _SnapshotRollout | None = None
) -> RolloutBatch:
    rows: list[_Transition] = []
    spans: list[TrajectorySpan] = []
    for slot in slots:
        stream = collector._streams[slot]
        start = len(rows)
        rows.extend(stream)
        if snapshots is not None:
            for stop, row in enumerate(stream, start + 1):
                if row.done or row.bootstrap_index is not None:
                    spans.append(
                        TrajectorySpan(start, stop, slot % 2, row.bootstrap_index)
                    )
                    start = stop
            assert start == len(rows)
        assert collector._pending[slot] is None
    if not rows:
        raise RuntimeError("learning model made no decisions in collected games")
    assert snapshots is not None or rows[-1].done
    batch = _build_batch(collector, rows, (len(rows), 1), snapshots, spans)
    for slot in slots:
        collector._streams[slot].clear()
    return batch


def frozen_bootstraps(
    frozen: ObservationRows | None, spans: list[TrajectorySpan]
) -> tuple[ObservationBuffers, tuple[TrajectorySpan, ...], int]:
    """Keep the frozen observations referenced by surviving trajectories."""
    if frozen is None:
        return {}, tuple(spans), 0
    observations = frozen.arrays()
    observations["action_focus"] = observations["action_focus"].astype(
        np.float32, copy=False
    )
    if frozen.memory_capacities is not None:
        observations = PrefixObservation(
            observations, frozen.memory, frozen.memory_capacities
        )
    indexes = sorted(
        {s.bootstrap_index for s in spans if s.bootstrap_index is not None}
    )
    if len(indexes) != frozen.rows:
        # A later sibling failure can discard a group whose cut was already read.
        remap = {old: new for new, old in enumerate(indexes)}
        observations = select_rows(observations, np.asarray(indexes, dtype=np.intp))
        spans = [
            replace(s, bootstrap_index=remap[s.bootstrap_index])
            if s.bootstrap_index is not None
            else s
            for s in spans
        ]
    return observations, tuple(spans), len(indexes)


def _build_batch(
    collector: Collector,
    rows: list[_Transition],
    shape: tuple[int, int],
    snapshots: _SnapshotRollout | None = None,
    spans: list[TrajectorySpan] | None = None,
) -> RolloutBatch:
    """Write every field and capture reference in the same flat time-major order."""
    pending = [row.pending for row in rows]
    dones = np.asarray([row.done for row in rows], bool).reshape(shape)
    trajectories = None
    provenance: dict[str, np.ndarray] = {}
    if snapshots is None:
        next_obs = _retain_obs_rows(collector, np.arange(shape[1]))
        next_done = dones[-1].copy()
    else:
        assert spans is not None
        next_obs, trajectories, count = frozen_bootstraps(snapshots.frozen, spans)
        next_done = np.zeros(count, bool)
        plan = snapshots.plan
        fresh_weight = plan.config.fresh_weight if plan is not None else 1.0
        branches = [p.branch for p in pending]
        provenance = {
            "episode_source": np.array([b is not None for b in branches], np.int8)[
                :, None
            ],
            "snapshot_id": np.array(
                [b.snapshot_id if b else -1 for b in branches], np.int64
            )[:, None],
            "branch_group_id": np.array(
                [b.branch_group_id if b else -1 for b in branches], np.int64
            )[:, None],
            "sample_weight": np.array(
                [b.sample_weight if b else fresh_weight for b in branches], np.float32
            )[:, None],
            "is_root_decision": np.array([p.is_root for p in pending], bool)[:, None],
        }
    recorded = collector._observations
    batch = RolloutBatch(
        recorded=RecordedObservations(
            recorded.arrays(),
            np.asarray([p.obs_row for p in pending], np.int64),
            recorded.needs(),
            tuple(recorded.memory),
            recorded.memory_capacities,
        ),
        actions=np.asarray([p.action for p in pending], np.int64).reshape(shape),
        logprobs=np.asarray([p.logprob for p in pending], np.float32).reshape(shape),
        rewards=np.asarray([row.reward for row in rows], np.float32).reshape(shape),
        dones=dones,
        values=np.asarray([p.value for p in pending], np.float32).reshape(shape),
        next_obs=next_obs,
        next_done=next_done,
        action_data=PackedActions.concatenate([p.action_data for p in pending]),
        outcome_probabilities=np.asarray(
            [p.outcome_probabilities for p in pending], np.float32
        ).reshape(*shape, 3)
        if pending[0].outcome_probabilities is not None
        else None,
        assignments=np.asarray([p.assignment for p in pending], np.int64).reshape(
            *shape, 4
        )
        if collector._sampler is not None
        else None,
        trajectories=trajectories,
        **provenance,
    )
    if snapshots is not None and snapshots.plan is not None:
        snapshots.plan.finish(pending)
    return batch
