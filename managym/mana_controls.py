"""Typed view of the native mana control offers in an opted-in live game."""

from typing import Literal

from managym import Env
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
    command_id: str
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


def mana_controls(env: Env) -> ManaControls:
    return ManaControls.model_validate_json(env.mana_controls_json())
