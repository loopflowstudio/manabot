"""The rules engine and native public API.

The engine types are implemented by PyO3 and described in ``__init__.pyi``;
shared decision/protocol adapters live beside this module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import sys
from typing import Literal, ParamSpec, TypeVar, overload
import warnings

from managym.disclosure import DecklistSampling

from . import _managym
from ._managym import *  # noqa: F403 — expose the native API described by the stub.
from ._managym import format

# PyO3 submodules are attributes, not importable modules; registering the
# native ``format`` lets ``import managym.format`` and ``from managym.format
# import Limited`` resolve to the same object as ``managym.format``.
sys.modules[f"{__name__}.format"] = format

# ---------------------------------------------------------------------------
# Deprecated spellings, removed in 0.3.0. Engine names describe what managym
# produces; these older names described how manabot's policy consumes it.
# ---------------------------------------------------------------------------

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _deprecated_alias(old: str, new: Callable[_P, _R]) -> Callable[_P, _R]:
    def alias(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        warnings.warn(
            f"managym.{old} is deprecated and removed in 0.3.0; "
            f"use managym.{new.__name__}",
            DeprecationWarning,
            stacklevel=2,
        )
        return new(*args, **kwargs)

    alias.__name__ = old
    alias.__qualname__ = old
    alias.__doc__ = f"Deprecated alias of :func:`{new.__name__}`, removed in 0.3.0."
    return alias


learning_catalog = _deprecated_alias("learning_catalog", _managym.catalog_tokens)
learning_token_value = _deprecated_alias(
    "learning_token_value", _managym.catalog_token_value
)
learning_record_names = _deprecated_alias(
    "learning_record_names", _managym.record_vocabulary
)


ManaPayment = Literal["auto", "manual"]
ManaFloats = Literal["off", "usable", "all"]


def _seat_modes(
    players: Sequence[_managym.PlayerConfig],
    field: str,
    value: Sequence[str] | None,
    names: tuple[str, ...],
) -> tuple[str, str]:
    """One mode per native seat; omitted values copy a wrapped setup or default
    to the first name."""
    if value is None:
        value = (
            getattr(players, field)
            if isinstance(players, GameSetup)
            else (names[0],) * 2
        )
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or not all(mode in names for mode in value)
    ):
        raise ValueError(f"{field} must contain two seat modes ({', '.join(names)})")
    return (value[0], value[1])


@dataclass(frozen=True, init=False)
class GameSetup(Sequence[_managym.PlayerConfig]):
    """Read-only seats, decklist disclosure and per-seat mana tap offers.

    Native reset resolves mixed 50/50 from the reset seed, independently of
    deck and shuffle draws. Iterating exposes seats, not a complete game setup.
    """

    players: tuple[_managym.PlayerConfig, ...]
    decklists: DecklistSampling
    mana_floats: tuple[ManaFloats, ManaFloats]
    mana_payment: tuple[ManaPayment, ManaPayment]

    def __init__(
        self,
        players: Sequence[_managym.PlayerConfig],
        *,
        decklists: DecklistSampling | None = None,
        mana_floats: tuple[ManaFloats, ManaFloats] | None = None,
        mana_payment: tuple[ManaPayment, ManaPayment] | None = None,
    ) -> None:
        if decklists is None:
            decklists = players.decklists if isinstance(players, GameSetup) else "open"
        if decklists not in ("open", "closed", "mixed"):
            raise ValueError("decklists must be open, closed or mixed")
        object.__setattr__(
            self,
            "mana_payment",
            _seat_modes(players, "mana_payment", mana_payment, ("manual", "auto")),
        )
        object.__setattr__(
            self,
            "mana_floats",
            _seat_modes(players, "mana_floats", mana_floats, ("off", "usable", "all")),
        )
        object.__setattr__(self, "players", tuple(players))
        object.__setattr__(self, "decklists", decklists)

    def __len__(self) -> int:
        return len(self.players)

    @overload
    def __getitem__(self, index: int) -> _managym.PlayerConfig: ...

    @overload
    def __getitem__(self, index: slice) -> "GameSetup": ...

    def __getitem__(self, index: int | slice) -> _managym.PlayerConfig | "GameSetup":
        if isinstance(index, slice):
            payments = self.mana_payment[: len(self.players)][index]
            seats = self.mana_floats[: len(self.players)][index]
            return GameSetup(
                self.players[index],
                decklists=self.decklists,
                mana_payment=(payments + ("manual", "manual"))[:2],
                mana_floats=(seats + ("off", "off"))[:2],
            )
        return self.players[index]


def __getattr__(name: str) -> str:
    # The engine carries no world label; the last one it carried was w17.
    if name == "WORLD_VERSION":
        warnings.warn(
            "managym.WORLD_VERSION is deprecated and removed in 0.4.0; the "
            "engine has no world label, and managym.world_identity() names "
            "its cards",
            DeprecationWarning,
            stacklevel=2,
        )
        return "w17"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
