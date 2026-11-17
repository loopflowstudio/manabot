"""Read and answer native mana control offers in an opted-in live game."""

from typing import Literal

from managym import Env
from managym.decision import Command, SemanticTransition
from managym.experience import ObjectSubject, ProtocolModel, UInt8, UInt32, UInt64

type Amounts = dict[
    Literal["white", "blue", "black", "red", "green", "colorless"], UInt8
]


class FloatOffer(ProtocolModel):
    verb: Literal["float"]
    id: UInt32
    source: ObjectSubject
    label: str
    output: Amounts


class UndoOffer(ProtocolModel):
    verb: Literal["undo"]
    id: UInt32
    label: str


class ManaActivation(ProtocolModel):
    label: str
    sources: list[ObjectSubject]
    mana: Amounts


class UndoReason(ProtocolModel):
    code: str
    explanation: str


class UndoStatus(ProtocolModel):
    activation: ManaActivation
    offer: UInt32 | None
    reason: UndoReason | None


class ManaControls(ProtocolModel):
    revision: UInt64
    offers: list[FloatOffer | UndoOffer]
    undo: UndoStatus | None


def mana_controls(env: Env, viewer: int) -> ManaControls:
    return ManaControls.model_validate_json(env.mana_controls_json(viewer))


def execute_mana_control(env: Env, command: Command) -> SemanticTransition:
    return SemanticTransition.from_json(
        env.execute_mana_control_json(command.to_json())
    )
