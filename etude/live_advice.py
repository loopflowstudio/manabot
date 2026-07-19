"""Bounded live posterior runtime for the selected GAM-7 advice slice."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import functools
import threading
from typing import Any

from manabot.belief.likelihood import FrozenPolicyLikelihood
from manabot.belief.range import BeliefState
from manabot.belief.tracker import BeliefTracker
from manabot.sim.search_runtime import (
    RetainedCheckpointRegistration,
    retained_int7_policy_only_checkpoint,
)
from managym.decision import SemanticTransition

LIVE_ADVICE_TRAINING_SEED = 197
LIVE_ADVICE_EPSILON = 0.01


class LiveBeliefUnavailable(RuntimeError):
    """The selected live posterior is pending, failed, or unsupported."""


@dataclass(frozen=True, slots=True)
class TrackedPosteriorSnapshot:
    revision: int
    posterior: BeliefState
    checkpoint: RetainedCheckpointRegistration
    viewer_history_sha256: str


@functools.cache
def _likelihood_registration() -> tuple[
    FrozenPolicyLikelihood, RetainedCheckpointRegistration
]:
    registration = retained_int7_policy_only_checkpoint(LIVE_ADVICE_TRAINING_SEED)
    likelihood = FrozenPolicyLikelihood(
        registration.checkpoint_path,
        expected_sha256=registration.checkpoint_sha256,
        counterfactual_seed=LIVE_ADVICE_TRAINING_SEED,
    )
    return likelihood, registration


def _initialize_tracker(root: Any) -> tuple[BeliefTracker, TrackedPosteriorSnapshot]:
    likelihood, registration = _likelihood_registration()
    tracker = BeliefTracker.from_engine(
        root,
        viewer=0,
        likelihood=likelihood,
        epsilon=LIVE_ADVICE_EPSILON,
    )
    snapshot = TrackedPosteriorSnapshot(
        revision=tracker.observation.revision,
        posterior=tracker.posterior,
        checkpoint=registration,
        viewer_history_sha256=str(tracker.replay_receipt()["history_digest"]),
    )
    return tracker, snapshot


class LiveBeliefRuntime:
    """Serialize posterior work away from the authoritative match thread."""

    def __init__(self, root: Any) -> None:
        self._lock = threading.Lock()
        self._initial_root = root
        self._executor: ThreadPoolExecutor | None = None
        self._snapshots: dict[int, TrackedPosteriorSnapshot] = {}
        self._error: BaseException | None = None
        self._future: Future[BeliefTracker] | None = None

    def _initialize(self, root: Any) -> BeliefTracker:
        try:
            tracker, snapshot = _initialize_tracker(root)
            with self._lock:
                self._snapshots[snapshot.revision] = snapshot
            return tracker
        except BaseException as error:
            with self._lock:
                self._error = error
            raise

    def observe(
        self,
        after_root: Any,
        *,
        acting: int,
        transition: SemanticTransition,
        likelihood_root: Any,
    ) -> None:
        with self._lock:
            if self._future is None:
                self._error = LiveBeliefUnavailable(
                    "tracked posterior was not active before transition"
                )
                return
        self._submit_update(
            after_root,
            acting=acting,
            transition=transition,
            likelihood_root=likelihood_root,
        )

    def _submit_update(
        self,
        after_root: Any,
        *,
        acting: int,
        transition: SemanticTransition,
        likelihood_root: Any,
    ) -> None:
        def update(previous: Future[BeliefTracker]) -> BeliefTracker:
            try:
                tracker = previous.result()
                tracker.observe(
                    after_root,
                    acting=acting,
                    transition=transition,
                    likelihood_root=likelihood_root,
                )
                snapshot = TrackedPosteriorSnapshot(
                    revision=tracker.observation.revision,
                    posterior=tracker.posterior,
                    checkpoint=retained_int7_policy_only_checkpoint(
                        LIVE_ADVICE_TRAINING_SEED
                    ),
                    viewer_history_sha256=str(
                        tracker.replay_receipt()["history_digest"]
                    ),
                )
                with self._lock:
                    self._snapshots[snapshot.revision] = snapshot
                return tracker
            except BaseException as error:
                with self._lock:
                    self._error = error
                raise

        with self._lock:
            previous = self._future
            executor = self._executor
            assert previous is not None and executor is not None
            self._future = executor.submit(update, previous)

    def _start(self) -> None:
        with self._lock:
            if self._future is not None or self._error is not None:
                return
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="etude-live-belief",
            )
            self._future = self._executor.submit(
                self._initialize, self._initial_root
            )

    @property
    def started(self) -> bool:
        with self._lock:
            return self._future is not None

    @property
    def valid(self) -> bool:
        with self._lock:
            return self._error is None

    def snapshot(self, revision: int) -> TrackedPosteriorSnapshot:
        self._start()
        with self._lock:
            snapshot = self._snapshots.get(revision)
            error = self._error
        if snapshot is not None:
            return snapshot
        if error is not None:
            raise LiveBeliefUnavailable("tracked posterior failed") from error
        raise LiveBeliefUnavailable("tracked posterior is not ready")

    def invalidate(self) -> None:
        with self._lock:
            if self._future is None and self._error is None:
                self._error = LiveBeliefUnavailable(
                    "tracked posterior was not active before transition"
                )

    def close(self) -> None:
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
