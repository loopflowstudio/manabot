"""Requests a browser sends on `/ws/play`.

A game is named by its id. `resume` takes up a game, in memory or from its
stored record; `new_game` and `rematch` start one; every other request acts
on the game this connection last took up or started. `REQUEST_ADAPTER` parses
any of them by its `type`.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias, get_args

from pydantic import Field, TypeAdapter

from managym.experience import Command, ProtocolModel, UInt64


class ResumeMessage(ProtocolModel):
    type: Literal["resume"]
    game: str = Field(min_length=1, max_length=128)
    # Strict: a cursor is an address, never a coerced boolean or string.
    presentation_cursor: UInt64 | None = Field(default=None, strict=True)
    # False leaves a game another connection holds with that connection
    # (`game_open_elsewhere`); otherwise this one takes it over.
    take_over: bool = True


class NewGameMessage(ProtocolModel):
    """Start a game from `config`; `config.launch` names a configured launch,
    which starts one game however often it is sent."""

    type: Literal["new_game"]
    config: dict[str, Any] = Field(default_factory=dict)


class RematchMessage(ProtocolModel):
    """Start a game from the setup `game` was played with, `config` applied
    over it."""

    type: Literal["rematch"]
    game: str = Field(min_length=1, max_length=128)
    intent: Literal["rematch", "swap_sides"] = "rematch"
    config: dict[str, Any] = Field(default_factory=dict)


class CommandMessage(ProtocolModel):
    type: Literal["command"]
    command: Command


class ActionMessage(ProtocolModel):
    type: Literal["action"]
    index: int


class PassTurnMessage(ProtocolModel):
    type: Literal["pass_turn"]


class RetryMirrorMessage(ProtocolModel):
    type: Literal["retry_mirror"]
    match_id: str
    expected_revision: UInt64


class SetStopsMessage(ProtocolModel):
    type: Literal["set_stops"]
    stops: dict[str, list[str]] | None = None
    stop_on_stack: bool | None = None
    auto_pass: bool | None = None


class ShowAnalysisMessage(ProtocolModel):
    """Show or hide a policy's reading of the person's decisions in this game.
    `policy` names a registered policy; omitted, the opponent's own or the
    first that can read the game."""

    type: Literal["show_analysis"]
    shown: bool
    policy: str | None = None


StartMessage: TypeAlias = ResumeMessage | NewGameMessage | RematchMessage
GameMessage: TypeAlias = (
    CommandMessage
    | ActionMessage
    | PassTurnMessage
    | RetryMirrorMessage
    | SetStopsMessage
    | ShowAnalysisMessage
)
PlayRequest: TypeAlias = Annotated[
    StartMessage | GameMessage, Field(discriminator="type")
]

REQUEST_ADAPTER: TypeAdapter[StartMessage | GameMessage] = TypeAdapter(PlayRequest)
REQUEST_TYPES: frozenset[str] = frozenset(
    get_args(model.model_fields["type"].annotation)[0]
    for alias in (StartMessage, GameMessage)
    for model in get_args(alias)
)
