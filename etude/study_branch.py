"""Authority-private Study forks over canonical replay decisions.

Game resolves a viewer-safe replay address; managym owns the retained rules
state and structured command execution. This module joins those authorities
without exposing a mixed-view replay or mutable engine root to clients.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

import managym

from .replay_index import (
    CanonicalReplayV1,
    DecisionNotFoundError,
    RestoredReplayDecision,
    restore_decision,
)


class StudyBranchUnavailableError(ValueError):
    """The canonical row has no retained authority-private rules root."""


STUDY_BRANCH_DRIVER = "full_clone/current_game_v1"
STUDY_COMMAND_PATH = "structured_offers/step_structured_v1"


class StudyExecutionReceipt(BaseModel):
    """The authority-private execution path used by one consumed branch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    driver: Literal["full_clone/current_game_v1"] = STUDY_BRANCH_DRIVER
    command_path: Literal["structured_offers/step_structured_v1"] = STUDY_COMMAND_PATH
    published_offer_sets: NonNegativeInt
    accepted_commands: NonNegativeInt
    rejected_commands: NonNegativeInt
    committed_engine_actions: NonNegativeInt
    fallback_commands: Literal[0] = 0


class StudyReturnReceipt(RestoredReplayDecision):
    """Exact canonical return bound to its retained rules authority root."""

    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution: StudyExecutionReceipt


@dataclass(frozen=True)
class ResolvedStudyAdvisorDecision:
    """Game-owned handoff of one validated replay row and isolated root."""

    restored: RestoredReplayDecision
    root: managym.Env
    source_digest: str


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
        self._published_offer_sets = 0
        self._accepted_commands = 0
        self._rejected_commands = 0
        self._committed_engine_actions = 0

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
        try:
            offers = env.structured_offers()
        except managym.AgentError as exc:
            raise StudyBranchUnavailableError(
                "Study decision has no native structured offer surface."
            ) from exc
        projection = json.loads(offers.projection_json())
        if projection.get("actor") != restored.viewer:
            raise StudyBranchUnavailableError(
                "Structured offer actor differs from the recorded viewer."
            )
        self._offers = offers
        self._published_offer_sets += 1
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
            self._rejected_commands += 1
            raise StudyBranchUnavailableError(
                "Publish structured offers before submitting a Study command."
            )

        offers = self._offers
        self._offers = None
        branch_digest = env.state_digest()
        try:
            submission_json = json.dumps(
                dict(submission), sort_keys=True, separators=(",", ":")
            )
            (
                _,
                reward,
                terminated,
                truncated,
                info,
                engine_actions,
            ) = env.step_structured(offers, submission_json)
        except (managym.AgentError, TypeError, ValueError, OverflowError) as exc:
            self._rejected_commands += 1
            if env.state_digest() != branch_digest:
                self._close()
                raise StudyBranchUnavailableError(
                    "Rejected Study command mutated its branch."
                ) from exc
            raise StudyBranchUnavailableError(f"Study command rejected: {exc}") from exc

        self._accepted_commands += 1
        self._committed_engine_actions += int(engine_actions)
        observation = env.observation_for_player(restored.viewer)
        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
            int(engine_actions),
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
            execution=self._execution_receipt(),
        )
        self._close()
        return returned

    def _execution_receipt(self) -> StudyExecutionReceipt:
        return StudyExecutionReceipt(
            published_offer_sets=self._published_offer_sets,
            accepted_commands=self._accepted_commands,
            rejected_commands=self._rejected_commands,
            committed_engine_actions=self._committed_engine_actions,
        )

    def _close(self) -> None:
        self._offers = None
        self._env = None
        self._restored = None
        self._source = None
        self._source_digest = None
        self._published_offer_sets = 0
        self._accepted_commands = 0
        self._rejected_commands = 0
        self._committed_engine_actions = 0

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
        restored, root, source_digest = self._resolve(raw_address, authorized_viewer)
        return StudyBranch(
            root.clone_env(),
            restored,
            root,
            source_digest,
        )

    def resolve_advisor_decision(
        self,
        raw_address: str,
        authorized_viewer: int,
    ) -> ResolvedStudyAdvisorDecision:
        """Resolve one exact decision for an injected Intelligence consumer.

        The provider validates replay identity, viewer authority, retained-root
        presence, and source immutability before returning an isolated clone.
        Callers never read GameSession root storage or reconstruct replay facts.
        """

        restored, root, source_digest = self._resolve(raw_address, authorized_viewer)
        return ResolvedStudyAdvisorDecision(
            restored=restored.model_copy(deep=True),
            root=root.clone_env(),
            source_digest=source_digest,
        )

    def _resolve(
        self,
        raw_address: str,
        authorized_viewer: int,
    ) -> tuple[RestoredReplayDecision, managym.Env, str]:
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
        return restored, root, source_digest
