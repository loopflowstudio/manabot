"""Closed control contract for one pilot and one watcher on an Etude table.

The match protocol remains authoritative and role-neutral.  These models cover
only participant leases, role capabilities, belief visibility, and the
participant-local Study controls around that match truth.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, TypeAdapter

from .advice_identity import AdviceIdentity
from .experience_protocol import Command, ProtocolModel, UInt64

TESTING_HOUSE_VERSION: Literal["testing-house-v1"] = "testing-house-v1"


class ViewerRole(str, Enum):
    PILOT = "pilot"
    WATCHER = "watcher"


class TableCapability(str, Enum):
    VIEW_TABLE = "view_table"
    AUTHOR_BELIEF = "author_belief"
    SHARE_BELIEF = "share_belief"
    COMPARE_ADVICE = "compare_advice"
    EXPLORE_STUDY = "explore_study"
    SUBMIT_LIVE_COMMAND = "submit_live_command"
    CONFIGURE_MATCH = "configure_match"
    TRANSFER_PILOT = "transfer_pilot"


ROLE_CAPABILITIES: dict[ViewerRole, tuple[TableCapability, ...]] = {
    ViewerRole.PILOT: tuple(TableCapability),
    ViewerRole.WATCHER: (
        TableCapability.VIEW_TABLE,
        TableCapability.AUTHOR_BELIEF,
        TableCapability.SHARE_BELIEF,
        TableCapability.COMPARE_ADVICE,
        TableCapability.EXPLORE_STUDY,
    ),
}


class ViewerIdentity(ProtocolModel):
    viewer_id: str
    table_id: str
    rules_viewer: Literal[0] = 0


class ViewerAccess(ProtocolModel):
    identity: ViewerIdentity
    role: ViewerRole
    capabilities: list[TableCapability]
    grant_revision: UInt64


class ParticipantPresence(ProtocolModel):
    viewer_id: str
    role: ViewerRole
    connected: bool


class BeliefSource(ProtocolModel):
    decision_address: str
    gam6_scenario_id: str
    advice_identity: AdviceIdentity


class PersonalAudience(ProtocolModel):
    kind: Literal["personal"]


class TableAudience(ProtocolModel):
    kind: Literal["table"]
    table_id: str


BeliefAudience: TypeAlias = Annotated[
    PersonalAudience | TableAudience,
    Field(discriminator="kind"),
]


class PlayerAuthoredBeliefProvenance(ProtocolModel):
    kind: Literal["player_authored"] = "player_authored"
    created_at_table_revision: UInt64
    shared_at_table_revision: UInt64 | None = None


class ModelInferredBeliefProvenance(ProtocolModel):
    kind: Literal["model_inferred"] = "model_inferred"
    belief_model_id: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    viewer_history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


BeliefProvenance: TypeAlias = Annotated[
    PlayerAuthoredBeliefProvenance | ModelInferredBeliefProvenance,
    Field(discriminator="kind"),
]


class BeliefScenario(ProtocolModel):
    id: str
    author_viewer_id: str
    source: BeliefSource
    audience: BeliefAudience
    provenance: BeliefProvenance


class TableDecisionSummary(ProtocolModel):
    address: str
    ordinal: UInt64
    revision: UInt64
    prompt_id: UInt64
    offer_id: int | None


class TableSnapshot(ProtocolModel):
    contract: Literal["testing-house-v1"] = TESTING_HOUSE_VERSION
    table_id: str
    table_revision: UInt64
    mode: Literal["live", "study"]
    access: ViewerAccess
    participants: list[ParticipantPresence]
    beliefs: list[BeliefScenario]
    decisions: list[TableDecisionSummary]
    opponent_label: str | None = None
    watcher_invite: str | None = None


class JoinTableMessage(ProtocolModel):
    type: Literal["join_table"]
    table_id: str
    invite_token: str
    presentation_cursor: UInt64 | None = None


class ResumeMessage(ProtocolModel):
    type: Literal["resume"]
    session_id: str
    resume_token: str
    presentation_cursor: UInt64 | None = None


class NewGameMessage(ProtocolModel):
    type: Literal["new_game"]
    grant_revision: UInt64 | None = None
    config: dict[str, object] = Field(default_factory=dict)


class RematchMessage(ProtocolModel):
    type: Literal["rematch"]
    grant_revision: UInt64
    config: dict[str, object] = Field(default_factory=dict)


class CommandMessage(ProtocolModel):
    type: Literal["command"]
    grant_revision: UInt64 | None = None
    command: Command


class ActionMessage(ProtocolModel):
    type: Literal["action"]
    grant_revision: UInt64 | None = None
    index: int


class PassTurnMessage(ProtocolModel):
    type: Literal["pass_turn"]
    grant_revision: UInt64 | None = None


class SetStopsMessage(ProtocolModel):
    type: Literal["set_stops"]
    grant_revision: UInt64 | None = None
    stops: dict[str, list[str]] | None = None
    stop_on_stack: bool | None = None
    auto_pass: bool | None = None


class TransferPilotMessage(ProtocolModel):
    type: Literal["transfer_pilot"]
    grant_revision: UInt64
    target_viewer_id: str


class AuthorBeliefMessage(ProtocolModel):
    type: Literal["author_belief"]
    grant_revision: UInt64
    scenario_id: str


class ShareBeliefMessage(ProtocolModel):
    type: Literal["share_belief"]
    grant_revision: UInt64
    belief_id: str


class RestoreDecisionMessage(ProtocolModel):
    type: Literal["restore_decision"]
    grant_revision: UInt64
    address: str


class RetryDecisionMessage(ProtocolModel):
    type: Literal["retry_decision"]
    grant_revision: UInt64
    address: str
    command: Command


class BranchRevealMessage(ProtocolModel):
    type: Literal["branch_reveal"]
    grant_revision: UInt64
    attempt_id: str


class BranchPreviewMessage(ProtocolModel):
    type: Literal["branch_preview"]
    grant_revision: UInt64
    attempt_id: str
    plan: Literal["played", "policy", "search"]


class ReturnFromBranchMessage(ProtocolModel):
    type: Literal["return_from_branch"]
    grant_revision: UInt64
    attempt_id: str


class ReturnToLiveMessage(ProtocolModel):
    type: Literal["return_to_live"]
    grant_revision: UInt64


TestingHouseRequest: TypeAlias = Annotated[
    JoinTableMessage
    | ResumeMessage
    | NewGameMessage
    | RematchMessage
    | CommandMessage
    | ActionMessage
    | PassTurnMessage
    | SetStopsMessage
    | TransferPilotMessage
    | AuthorBeliefMessage
    | ShareBeliefMessage
    | RestoreDecisionMessage
    | RetryDecisionMessage
    | BranchRevealMessage
    | BranchPreviewMessage
    | ReturnFromBranchMessage
    | ReturnToLiveMessage,
    Field(discriminator="type"),
]

REQUEST_ADAPTER = TypeAdapter(TestingHouseRequest)
REQUEST_TYPES = (
    "join_table",
    "resume",
    "new_game",
    "rematch",
    "command",
    "action",
    "pass_turn",
    "set_stops",
    "transfer_pilot",
    "author_belief",
    "share_belief",
    "restore_decision",
    "retry_decision",
    "branch_reveal",
    "branch_preview",
    "return_from_branch",
    "return_to_live",
)


class TableSnapshotEvent(ProtocolModel):
    type: Literal["table_snapshot"]
    table: TableSnapshot


class BeliefChangedEvent(ProtocolModel):
    type: Literal["belief_changed"]
    table: TableSnapshot
    belief: BeliefScenario


class RoleChangedEvent(ProtocolModel):
    type: Literal["role_changed"]
    table: TableSnapshot


class DecisionRestoredEvent(ProtocolModel):
    type: Literal["decision_restored"]
    address: str
    restored: dict[str, object]
    table: TableSnapshot | None = None


class BranchUpdatedEvent(ProtocolModel):
    type: Literal["branch_updated"]
    attempt_id: str
    phase: Literal["retry", "revealed", "preview"]
    payload: dict[str, object]
    table: TableSnapshot | None = None


class BranchReturnedEvent(ProtocolModel):
    type: Literal["branch_returned"]
    restored: dict[str, object]
    table: TableSnapshot | None = None


class ControlErrorEvent(ProtocolModel):
    type: Literal["control_error"]
    code: Literal[
        "invalid_message",
        "unsupported_message",
        "forbidden",
        "stale_grant",
        "not_found",
        "conflict",
    ]
    message: str
    table: TableSnapshot | None = None


TestingHouseControlEvent: TypeAlias = Annotated[
    TableSnapshotEvent
    | BeliefChangedEvent
    | RoleChangedEvent
    | DecisionRestoredEvent
    | BranchUpdatedEvent
    | BranchReturnedEvent
    | ControlErrorEvent,
    Field(discriminator="type"),
]


class TestingHouseV1ConformanceBundle(ProtocolModel):
    contract: Literal["testing-house-v1"] = TESTING_HOUSE_VERSION
    requests: list[TestingHouseRequest]
    events: list[TestingHouseControlEvent]


def testing_house_schema() -> dict[str, object]:
    """Return the checked Draft 2020-12 control schema."""
    return TestingHouseV1ConformanceBundle.model_json_schema()


def request_types() -> tuple[str, ...]:
    """Closed operation vocabulary used by both validation and dispatch."""
    return REQUEST_TYPES
