"""manabot consumer of managym's shared semantic decision contract.

The adapter drives one agent/search decision through the revision-bound
semantic authority: read the current :class:`~managym.decision.DecisionFrame`,
build an atomic :class:`~managym.decision.Command`, apply it, and return the
fail-closed :class:`~managym.decision.TransitionReceipt` plus the next
composite :class:`~managym.decision.Observation`.

This is the manabot-side vertical slice of ``docs/ARCHITECTURE.md`` step 1 /
Rules R1. Legal-action identity comes from the authoritative action space via
``structured_search_offers``; the contract is revision-bound and fail-closed.
It does not migrate the legacy positional ``Env.step`` path or duplicate
world-space ownership (RUL-8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from managym.decision import (
    Command,
    DecisionFrame,
    SemanticContractError,
    SemanticTransition,
    apply_semantic_command,
)


@dataclass
class SemanticDecisionContract:
    """One revision-bound agent decision over managym's authority."""

    frame: DecisionFrame

    @classmethod
    def from_env(cls, env: Any) -> "SemanticDecisionContract":
        return cls(frame=DecisionFrame.from_json(env.semantic_decision_frame_json()))

    def command(
        self,
        offer_id: int,
        *,
        command_id: str | None = None,
        answers: Sequence[Mapping[str, Any]] = (),
        object_preconditions: Sequence[Mapping[str, Any]] = (),
    ) -> Command:
        # Fail closed locally before the round-trip: the offer must belong to
        # the exact frame this contract is bound to.
        self.frame.offer(offer_id)
        return Command(
            command_id=command_id or f"manabot:{self.frame.revision}:{offer_id}",
            expected_revision=self.frame.revision,
            offer_id=offer_id,
            answers=tuple(answers),
            object_preconditions=tuple(object_preconditions),
        )

    def apply_command(
        self, env: Any, command: Command | Mapping[str, Any]
    ) -> SemanticTransition:
        transition = apply_semantic_command(env, command)
        if transition.receipt.before_revision != self.frame.revision:
            raise SemanticContractError(
                "receipt before_revision does not match the bound frame"
            )
        return transition

    def apply(
        self,
        env: Any,
        offer_id: int,
        *,
        command_id: str | None = None,
        answers: Sequence[Mapping[str, Any]] = (),
        object_preconditions: Sequence[Mapping[str, Any]] = (),
    ) -> SemanticTransition:
        return self.apply_command(
            env,
            self.command(
                offer_id,
                command_id=command_id,
                answers=answers,
                object_preconditions=object_preconditions,
            ),
        )
