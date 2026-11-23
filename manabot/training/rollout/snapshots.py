"""Schedule sibling continuations and score retained roots after a successful update.

A BatchPlan spends replay allowance in actual episode-start order. Its receipt
joins capture-time observation rows to packed learner rows. Only commit publishes
the scheduler and scores. Hot handles belong to the collecting vector and never
enter the cold state.
"""

from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
import logging
import math
import random
from threading import RLock
from types import TracebackType
from typing import Literal, Protocol, Self

import numpy as np
from pydantic import ConfigDict, Field, TypeAdapter

from manabot.infra.identity import canonical_sha256
from manabot.infra.schema import Strict
from manabot.training.rollout.adaptive import rank_weights
from manabot.training.rollout.signals import BatchSignals
from manabot.training.rollout.snapshot_vector import (
    CapturedSnapshot,
    InstalledSlot,
    SlotInstall,
    SnapshotBranch,
    SnapshotRecipe,
    SnapshotVector,
)


def release_handles(vector: SnapshotVector, handles: Sequence[object]) -> None:
    """Cleanup failures must not turn a published generation into a failed commit."""
    if handles:
        try:
            vector.release_handles(handles)
        except Exception:
            logging.getLogger(__name__).exception(
                "Snapshot vector failed to release handles"
            )


class CapturePolicy(Strict):
    model_config = ConfigDict(frozen=True)

    every_n_decisions: int | None = Field(default=None, ge=1)
    probability: float = Field(default=0, ge=0, le=1)
    min_legal_actions: int = Field(default=2, ge=1)
    max_captures_per_batch: int | None = Field(default=None, ge=0)
    fresh_only: bool = True


class SnapshotConfig(Strict):
    model_config = ConfigDict(frozen=True)

    capacity: int = Field(ge=1)
    replay_fraction: float = Field(ge=0, le=1)
    fresh_floor: float = Field(ge=0, le=1)
    group_size: int = Field(default=1, ge=1)
    horizon: int | None = Field(default=None, ge=1)
    score_half_life: float = Field(default=8, gt=0)
    fresh_weight: float = Field(default=1, ge=0)
    snapshot_weight: float = Field(default=1, ge=0)
    signal: Literal[
        "critic_residual",
        "positive_value_loss",
        "wdl_entropy",
        "policy_entropy",
        "action_value_spread",
        "staleness",
    ] = "critic_residual"
    signal_scope: Literal["root", "episode"] = "episode"
    capture: CapturePolicy = Field(default_factory=CapturePolicy)
    seed: int = 0
    max_group_retries: int = Field(default=1, ge=0)


@dataclass(frozen=True)
class CaptureDecision:
    slot: int
    decision: int  # One-based surfaced choice within this episode.
    legal_actions: int
    chooser: int
    branch: SnapshotBranch | None
    episode: int
    obs_row: int


CapturePredicate = Callable[[CaptureDecision], bool]


@dataclass(frozen=True)
class EntrySignals:
    root_signals: BatchSignals
    episode_signals: BatchSignals
    age_since_visit: int


ScorerFn = Callable[[EntrySignals], float]
_RECIPE = TypeAdapter(SnapshotRecipe)


def root_id(recipe: SnapshotRecipe) -> int:
    """Canonical recipe identity in the signed int64 rollout column."""
    return int(canonical_sha256(_RECIPE.dump_python(recipe, mode="json"))[:16], 16) >> 1


class SnapshotEntry(Strict):
    snapshot_id: int
    recipe: SnapshotRecipe
    tags: tuple[str, ...] = ()
    score: float = Field(ge=0)
    score_generation: int = Field(ge=0)
    last_visit_generation: int = Field(ge=0)
    visits: int = Field(default=0, ge=0)

    def priority(
        self, config: SnapshotConfig, generation: int, scorer: ScorerFn | None = None
    ) -> float:
        if config.signal == "staleness" and scorer is None:
            return max(0, generation - self.last_visit_generation)
        return self.score * 2 ** (
            -(generation - self.score_generation) / config.score_half_life
        )


RandomState = tuple[int, tuple[int, ...], float | None]


class SchedulerState(Strict):
    starts: int = Field(default=0, ge=0)
    groups: int = Field(default=0, ge=0)
    # Units of 1 / replay_rate.denominator starts. Integer arithmetic avoids
    # losing a whole group at decimal boundaries or across JSON round trips.
    allowance: int = Field(default=0, ge=0)
    coverage: tuple[int, ...] = ()
    rng: RandomState
    capture_rng: RandomState


class SnapshotBankState(Strict):
    schema_version: Literal[1] = 1
    config_digest: str
    generation: int = Field(default=0, ge=0)
    entries: tuple[SnapshotEntry, ...] = ()
    scheduler: SchedulerState


@dataclass(frozen=True)
class CapturedCandidate:
    snapshot: CapturedSnapshot
    row: int
    episode_rows: tuple[int, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class VisitedGroup:
    group_id: int
    snapshot_id: int
    roots: tuple[int, ...]
    rows: tuple[int, ...]


@dataclass(frozen=True)
class BatchReceipt:
    base_generation: int
    next_scheduler_state: SchedulerState
    visited_groups: tuple[VisitedGroup, ...]
    captures: tuple[CapturedCandidate, ...]
    row_count: int
    plan: "BatchPlan" = field(repr=False, compare=False)


class PackedDecision(Protocol):
    """Fields shared with a retained collector decision during batch assembly."""

    episode_ordinal: int
    obs_row: int
    branch: SnapshotBranch | None
    is_root: bool


class BatchPlan:
    """One speculative allocation and its captured handles, until commit or close."""

    def __init__(self, bank: "SnapshotBank") -> None:
        self.bank = bank
        self.config = bank.config
        state = bank._state
        self.base_generation = state.generation
        self.entries = {e.snapshot_id: e for e in state.entries}
        self.scheduler = state.scheduler.model_copy()
        self.rng, self.capture_rng = random.Random(), random.Random()
        self.rng.setstate(self.scheduler.rng)
        self.capture_rng.setstate(self.scheduler.capture_rng)
        self.rate = min(
            Fraction(str(self.config.replay_fraction)),
            1 - Fraction(str(self.config.fresh_floor)),
        )
        self.coverage = deque(self.scheduler.coverage)
        self.unvisited = deque(
            sorted(
                (k for k, e in self.entries.items() if not e.visits),
                key=self.visit_order,
            )
        )
        self.pending: deque[SnapshotBranch] = deque()
        self.retry_reserve = 0
        self.completed_episodes = 0
        self.successes: dict[int, int] = defaultdict(int)
        self.retry_counts: dict[int, int] = defaultdict(int)
        self.discarded: set[int] = set()
        self.vector: SnapshotVector | None = None
        self.candidates: dict[
            int, list[tuple[int, CapturedSnapshot, tuple[str, ...]]]
        ] = defaultdict(list)
        self.capture_count = 0
        self.receipt: BatchReceipt | None = None
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.receipt = None
            for episode in list(self.candidates):
                self.discard_captures(episode)

    def discard_captures(self, episode: int) -> None:
        handles = [
            s.hot_handle
            for _, s, _ in self.candidates.pop(episode, ())
            if s.hot_handle is not None
        ]
        if handles:
            assert self.vector is not None
            release_handles(self.vector, handles)

    def visit_order(self, key: int) -> tuple[int, int]:
        return self.entries[key].last_visit_generation, key

    def _draw(self) -> SnapshotEntry:
        if self.unvisited:
            key = self.unvisited.popleft()
        elif (self.scheduler.groups + 1) % 5 == 0:
            if not self.coverage:
                self.coverage.extend(sorted(self.entries, key=self.visit_order))
            key = self.coverage.popleft()
        elif self.rng.random() < 0.25:
            key = min(self.entries, key=self.visit_order)
        else:
            keys = sorted(self.entries, key=self.visit_order)
            scores = [
                self.entries[k].priority(
                    self.config, self.base_generation, self.bank.scorer
                )
                for k in keys
            ]
            [key] = self.rng.choices(keys, weights=rank_weights(scores))
        self.entries[key] = self.entries[key].model_copy(
            update={"last_visit_generation": self.base_generation + 1}
        )
        return self.entries[key]

    def can_replay(self, count: int) -> bool:
        return self.scheduler.allowance >= count * (
            self.rate.denominator - self.rate.numerator
        )

    def next_episode(self) -> SnapshotBranch | None:
        if self.closed or self.receipt is not None:
            raise ValueError("snapshot plan is finished")
        k = self.config.group_size
        # With isolated failures, reserve enough starts to retry every group
        # concurrently in flight. This keeps retries funded even after closure.
        if (
            not self.pending
            and self.entries
            and self.can_replay(k + self.retry_reserve)
        ):
            entry = self._draw()
            group = self.scheduler.groups
            self.scheduler.groups += 1
            for index in range(k):
                self.pending.append(
                    SnapshotBranch(
                        entry.snapshot_id,
                        entry.recipe,
                        self.config.horizon,
                        group,
                        index,
                        self.config.snapshot_weight,
                    )
                )
        self.scheduler.starts += 1
        if self.entries:
            self.scheduler.allowance += self.rate.numerator
        if self.pending:
            self.scheduler.allowance -= self.rate.denominator
            return self.pending.popleft()
        return None

    def fail(self, branch: SnapshotBranch) -> None:
        group = branch.branch_group_id
        if group in self.discarded:
            return
        if self.retry_counts[group] < self.config.max_group_retries:
            self.retry_counts[group] += 1
            self.pending.append(branch)
        else:
            self.discarded.add(group)
            self.pending = deque(b for b in self.pending if b.branch_group_id != group)

    def succeed(self, branch: SnapshotBranch | None) -> None:
        if branch is None:
            self.completed_episodes += 1
        elif branch.branch_group_id not in self.discarded:
            self.successes[branch.branch_group_id] += 1
            if self.successes[branch.branch_group_id] == self.config.group_size:
                self.completed_episodes += self.config.group_size

    def capture(
        self, decisions: Iterable[CaptureDecision], predicate: CapturePredicate | None
    ) -> None:
        """Stage requested pre-step roots on their live episodes."""
        selected = {d.slot: d for d in decisions if self.wants_capture(d, predicate)}
        if not selected:
            return
        assert self.vector is not None
        captures = self.vector.capture_slots(list(selected))
        if (
            len(captures) != len(selected)
            or {c.slot for c in captures} != selected.keys()
            or any(
                (c.root_digest, c.root_chooser)
                != (c.recipe.root_digest, c.recipe.root_chooser)
                or c.root_chooser != selected[c.slot].chooser
                for c in captures
            )
        ):
            release_handles(
                self.vector,
                [c.hot_handle for c in captures if c.hot_handle is not None],
            )
            raise ValueError("snapshot captures differ from requested decisions")
        for capture in captures:
            decision = selected[capture.slot]
            tags = ("snapshot",) if decision.branch is not None else ("fresh",)
            self.candidates[decision.episode].append((decision.obs_row, capture, tags))

    def wants_capture(
        self, decision: CaptureDecision, predicate: CapturePredicate | None
    ) -> bool:
        policy = self.config.capture
        if (
            (policy.fresh_only and decision.branch is not None)
            or decision.legal_actions < policy.min_legal_actions
            or (
                policy.max_captures_per_batch is not None
                and self.capture_count >= policy.max_captures_per_batch
            )
            or (predicate is not None and not predicate(decision))
        ):
            return False
        periodic = (
            policy.every_n_decisions is not None
            and decision.decision % policy.every_n_decisions == 0
        )
        if periodic or (
            policy.probability and self.capture_rng.random() < policy.probability
        ):
            self.capture_count += 1
            return True
        return False

    def finish(self, decisions: Iterable[PackedDecision]) -> BatchReceipt:
        """Resolve surviving observation references after both seats are packed."""
        by_episode: dict[int, list[int]] = defaultdict(list)
        by_group: dict[tuple[int, int], list[int]] = defaultdict(list)
        roots: dict[tuple[int, int], list[int]] = defaultdict(list)
        flat: dict[int, int] = {}
        row_count = 0
        for i, decision in enumerate(decisions):
            row_count += 1
            if decision.episode_ordinal in self.candidates:
                by_episode[decision.episode_ordinal].append(i)
                flat[decision.obs_row] = i
            if (branch := decision.branch) is not None:
                group = (branch.branch_group_id, branch.snapshot_id)
                by_group[group].append(i)
                if decision.is_root:
                    roots[group].append(i)
        visits = []
        for (group, snapshot_id), rows in sorted(by_group.items()):
            root_rows = tuple(roots[group, snapshot_id])
            if group in self.discarded or len(root_rows) != self.config.group_size:
                raise ValueError(f"incomplete snapshot group {group}")
            visits.append(VisitedGroup(group, snapshot_id, root_rows, tuple(rows)))
        captures = []
        for episode in sorted(self.candidates):
            if episode not in by_episode:
                self.discard_captures(episode)
                continue
            episode_rows = tuple(by_episode[episode])
            captures.extend(
                CapturedCandidate(s, flat[row], episode_rows, tags)
                for row, s, tags in self.candidates[episode]
            )
        scheduler = self.scheduler.model_copy(
            update=dict(
                coverage=tuple(self.coverage),
                rng=self.rng.getstate(),
                capture_rng=self.capture_rng.getstate(),
            )
        )
        self.receipt = BatchReceipt(
            self.base_generation,
            scheduler,
            tuple(visits),
            tuple(captures),
            row_count,
            self,
        )
        return self.receipt


def signal_slice(signals: BatchSignals, indexes: Sequence[int]) -> BatchSignals:
    rows = np.asarray(indexes, dtype=np.intp)
    result = {}
    for column in fields(signals):
        values = getattr(signals, column.name)
        if values is not None:
            values = values[rows]
            values.setflags(write=False)
        result[column.name] = values
    return BatchSignals(**result)


@dataclass
class _ScoreRows:
    roots: set[int] = field(default_factory=set)
    episodes: list[Sequence[int]] = field(default_factory=list)
    visits: int = 0


def score_entry(
    sample: _ScoreRows,
    signals: BatchSignals,
    config: SnapshotConfig,
    selected: np.ndarray | None,
    scorer: ScorerFn | None,
    age: int,
) -> float:
    if config.signal == "staleness" and scorer is None:
        return float(age)
    roots = sorted(sample.roots)
    # A branch root may itself have triggered capture. Count each packed
    # decision once even when the receipt reaches it through both paths.
    rows = (
        sorted({r for episode in sample.episodes for r in episode})
        if scorer is not None or config.signal_scope == "episode"
        else roots
    )
    if scorer is not None:
        score = scorer(
            EntrySignals(signal_slice(signals, roots), signal_slice(signals, rows), age)
        )
    else:
        assert selected is not None
        score = float(np.mean(selected[rows]))
    if not math.isfinite(score) or score < 0:
        raise ValueError("snapshot score must be finite and nonnegative")
    return score


class SnapshotBank:
    """A bounded set of scored roots, published one complete generation at a time."""

    def __init__(
        self, config: SnapshotConfig, *, scorer: ScorerFn | None = None
    ) -> None:
        self._lock = RLock()
        self.config = config.model_copy(deep=True)
        self.scorer = scorer
        rng = random.Random(config.seed).getstate()
        self._state = SnapshotBankState(
            config_digest=canonical_sha256(config.model_dump(mode="json")),
            scheduler=SchedulerState(rng=rng, capture_rng=rng),
        )
        self._hot: dict[int, tuple[SnapshotVector, object]] = {}

    @property
    def generation(self) -> int:
        return self._state.generation

    def state(self) -> SnapshotBankState:
        return self._state.model_copy(deep=True)

    @classmethod
    def from_state(
        cls,
        config: SnapshotConfig,
        state: SnapshotBankState,
        *,
        scorer: ScorerFn | None = None,
    ) -> Self:
        bank = cls(config, scorer=scorer)
        if bank._state.config_digest != state.config_digest:
            raise ValueError("snapshot bank configuration differs from saved state")
        ids = {e.snapshot_id for e in state.entries}
        if (
            len(ids) != len(state.entries)
            or len(ids) > config.capacity
            or any(root_id(e.recipe) != e.snapshot_id for e in state.entries)
            or not ids.issuperset(state.scheduler.coverage)
        ):
            raise ValueError("invalid snapshot bank entries or coverage")
        bank._state = state.model_copy(deep=True)
        return bank

    def begin_batch(self) -> BatchPlan:
        with self._lock:
            return BatchPlan(self)

    def install(
        self, vector: SnapshotVector, requests: list[SlotInstall]
    ) -> list[InstalledSlot]:
        # A concurrent commit may evict the root between sibling allocations.
        # Borrow its vector-local handle only for the synchronous native call.
        with self._lock:
            installs = []
            for request in requests:
                cached = self._hot.get(request.snapshot_id)
                installs.append(
                    replace(request, hot_handle=cached[1])
                    if cached and cached[0] is vector
                    else request
                )
            return vector.install_slots(installs)

    def close(self) -> None:
        with self._lock:
            for owner, handle in self._hot.values():
                release_handles(owner, [handle])
            self._hot.clear()

    def commit(self, receipt: BatchReceipt, signals: BatchSignals) -> None:
        """Publish scores and the scheduler together, once per successful update."""
        with self._lock:
            self._commit(receipt, signals)

    def _commit(self, receipt: BatchReceipt, signals: BatchSignals) -> None:
        plan = receipt.plan
        if (
            receipt.base_generation != self.generation
            or plan.bank is not self
            or plan.closed
            or plan.receipt is not receipt
        ):
            raise ValueError("stale or already committed snapshot receipt")
        config = self.config
        for column in fields(signals):
            values = getattr(signals, column.name)
            if values is not None and values.shape != (receipt.row_count,):
                raise ValueError("snapshot signals differ from packed row count")
        selected = getattr(signals, config.signal, None)
        if self.scorer is None and config.signal != "staleness" and selected is None:
            raise ValueError(f"snapshot signal {config.signal} unavailable")
        generation = self.generation + 1
        entries = {e.snapshot_id: e for e in self._state.entries}
        scores: dict[int, _ScoreRows] = defaultdict(_ScoreRows)
        latest: dict[int, CapturedCandidate] = {}
        for visit in receipt.visited_groups:
            sample = scores[visit.snapshot_id]
            sample.roots.update(visit.roots)
            sample.episodes.append(visit.rows)
            sample.visits += 1
        for capture in receipt.captures:
            key = root_id(capture.snapshot.recipe)
            if key not in entries:
                entries[key] = SnapshotEntry(
                    snapshot_id=key,
                    recipe=replace(
                        capture.snapshot.recipe,
                        match=capture.snapshot.recipe.match.model_copy(deep=True),
                    ),
                    tags=capture.tags,
                    score=0,
                    score_generation=generation,
                    last_visit_generation=generation,
                )
            elif entries[key].recipe != capture.snapshot.recipe:
                raise ValueError("snapshot identity collision")
            latest[key] = capture
            scores[key].roots.add(capture.row)
            scores[key].episodes.append(capture.episode_rows)
        for key, sample in scores.items():
            entry = entries[key]
            age = max(0, self.generation - entry.last_visit_generation)
            score = score_entry(sample, signals, config, selected, self.scorer, age)
            entries[key] = entry.model_copy(
                update=dict(
                    score=float(score),
                    score_generation=generation,
                    visits=entry.visits + sample.visits,
                    last_visit_generation=generation
                    if sample.visits
                    else entry.last_visit_generation,
                )
            )
        excess = len(entries) - config.capacity
        if excess > 0:
            for key in sorted(
                entries,
                key=lambda k: (
                    entries[k].priority(config, generation, self.scorer),
                    entries[k].last_visit_generation,
                    k,
                ),
            )[:excess]:
                del entries[key]
        scheduler = receipt.next_scheduler_state.model_copy(
            update=dict(
                coverage=tuple(
                    k for k in receipt.next_scheduler_state.coverage if k in entries
                )
            )
        )
        state = SnapshotBankState(
            config_digest=self._state.config_digest,
            generation=generation,
            scheduler=scheduler,
            entries=tuple(entries[k] for k in sorted(entries)),
        )
        hot = {k: h for k, h in self._hot.items() if k in entries}
        transferred = set()
        for key, capture in latest.items():
            handle = capture.snapshot.hot_handle
            if key in entries and handle is not None:
                assert plan.vector is not None
                hot[key] = (plan.vector, handle)
                transferred.add(id(capture.snapshot))
        old_hot, self._state, self._hot = self._hot, state, hot
        for key, (owner, handle) in old_hot.items():
            if hot.get(key) is not old_hot[key]:
                release_handles(owner, [handle])
        if plan.vector is not None:
            release_handles(
                plan.vector,
                [
                    c.snapshot.hot_handle
                    for c in receipt.captures
                    if c.snapshot.hot_handle is not None
                    and id(c.snapshot) not in transferred
                ],
            )
        plan.candidates.clear()
        plan.close()
