"""Authority-private Study forks over canonical replay decisions.

Game resolves a viewer-safe replay address; managym owns the retained rules
state and structured command execution. This module joins those authorities
without exposing a mixed-view replay or mutable engine root to clients.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from pydantic import Field

import managym

from .replay_index import (
    CanonicalReplayV1,
    DecisionNotFoundError,
    RestoredReplayDecision,
    restore_decision,
)


class StudyBranchUnavailableError(ValueError):
    """The canonical row has no retained authority-private rules root."""


class StudyReturnReceipt(RestoredReplayDecision):
    """Exact canonical return bound to its retained rules authority root."""

    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class StudyBranch:
    """One ephemeral exact fork for a single historical viewer."""

    def __init__(
        self,
        env: managym.Env,
        restored: RestoredReplayDecision,
        source: managym.Env,
        source_digest: str,
    ) -> None:
        self._env: managym.Env | None = env
        self._restored: RestoredReplayDecision | None = restored.model_copy(deep=True)
        self._source: managym.Env | None = source
        self._source_digest: str | None = source_digest
        self._offers: Any | None = None

    @property
    def viewer(self) -> int:
        restored = self._require_open()[1]
        return int(restored.viewer)

    @property
    def address(self) -> str:
        return self._require_open()[1].address

    def structured_offers(self) -> dict[str, Any]:
        """Publish only the current viewer's normal structured offer surface."""
        env, restored = self._require_open()
        if env.current_agent_index() != restored.viewer:
            raise StudyBranchUnavailableError(
                "Study branch is waiting on another player's decision."
            )
        offers = env.structured_offers()
        projection = json.loads(offers.projection_json())
        if projection.get("actor") != restored.viewer:
            raise StudyBranchUnavailableError(
                "Structured offer actor differs from the recorded viewer."
            )
        self._offers = offers
        return projection

    def current_observation(self) -> managym.Observation:
        """Return the branch's current viewer-safe authority observation."""
        env, restored = self._require_open()
        return env.observation_for_player(restored.viewer)

    def submit(
        self,
        submission: Mapping[str, Any],
    ) -> tuple[managym.Observation, float, bool, bool, dict[str, Any], int]:
        """Apply one prompt-bound structured command on the branch only."""
        env, restored = self._require_open()
        if env.current_agent_index() != restored.viewer:
            raise StudyBranchUnavailableError(
                "Study branch is waiting on another player's decision."
            )
        if self._offers is None:
            raise StudyBranchUnavailableError(
                "Publish structured offers before submitting a Study command."
            )

        _, reward, terminated, truncated, info, legacy_actions = env.step_structured(
            self._offers,
            json.dumps(dict(submission), sort_keys=True, separators=(",", ":")),
        )
        self._offers = None
        observation = env.observation_for_player(restored.viewer)
        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
            int(legacy_actions),
        )

    def return_to_recorded(self) -> StudyReturnReceipt:
        """Close the branch and return the source-bound canonical decision."""
        _, restored = self._require_open()
        source = self._source
        source_digest = self._source_digest
        if (
            source is None
            or source_digest is None
            or source.state_digest() != source_digest
        ):
            self._close()
            raise StudyBranchUnavailableError("Retained Study root drifted.")
        returned = StudyReturnReceipt(
            **restored.model_dump(mode="python"),
            source_digest=source_digest,
        )
        self._close()
        return returned

    def _close(self) -> None:
        self._offers = None
        self._env = None
        self._restored = None
        self._source = None
        self._source_digest = None

    def _require_open(self) -> tuple[managym.Env, RestoredReplayDecision]:
        if self._env is None or self._restored is None:
            raise StudyBranchUnavailableError("Study branch has returned to replay.")
        return self._env, self._restored


class StudyForkProvider:
    """Resolve authorized replay addresses to immutable retained roots."""

    def __init__(
        self,
        replay: CanonicalReplayV1,
        roots: Mapping[int, managym.Env],
    ) -> None:
        replay = replay.model_copy(deep=True)
        replay_rows = {int(row.ordinal): row for row in replay.decisions}
        retained: dict[int, tuple[managym.Env, str]] = {}
        for ordinal, env in roots.items():
            row = replay_rows.get(int(ordinal))
            if row is None or env.current_agent_index() != row.viewer:
                raise StudyBranchUnavailableError(
                    "Retained Study root differs from its canonical decision."
                )
            retained[int(ordinal)] = (env, env.state_digest())
        self._replay = replay
        self._roots = retained

    def fork(self, raw_address: str, authorized_viewer: int) -> StudyBranch:
        restored = restore_decision(
            self._replay,
            raw_address,
            authorized_viewer=authorized_viewer,
        )
        retained = self._roots.get(int(restored.ordinal))
        if retained is None:
            raise DecisionNotFoundError("decision not found")
        root, source_digest = retained
        if (
            root.current_agent_index() != restored.viewer
            or root.state_digest() != source_digest
        ):
            raise StudyBranchUnavailableError("Retained Study root drifted.")
        return StudyBranch(root.clone_env(), restored, root, source_digest)
