"""The rules engine and native public API.

The engine types are implemented by PyO3 and described in ``__init__.pyi``;
shared decision/protocol adapters live beside this module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import sys
from typing import ParamSpec, TypeVar, overload
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


@dataclass(frozen=True, init=False)
class GameSetup(Sequence[_managym.PlayerConfig]):
    """Read-only seats, decklist disclosure and per-seat priority mana offers.

    Native reset resolves mixed 50/50 from the reset seed, independently of
    deck and shuffle draws. Iterating exposes seats, not a complete game setup.
    """

    players: tuple[_managym.PlayerConfig, ...]
    decklists: DecklistSampling
    priority_mana_taps: tuple[bool, bool]

    def __init__(
        self,
        players: Sequence[_managym.PlayerConfig],
        *,
        decklists: DecklistSampling | None = None,
        priority_mana_taps: tuple[bool, bool] | None = None,
    ) -> None:
        if decklists is None:
            decklists = players.decklists if isinstance(players, GameSetup) else "open"
        if decklists not in ("open", "closed", "mixed"):
            raise ValueError("decklists must be open, closed or mixed")
        if priority_mana_taps is None:
            priority_mana_taps = (
                players.priority_mana_taps
                if isinstance(players, GameSetup)
                else (False, False)
            )
        if (
            not isinstance(priority_mana_taps, (tuple, list))
            or len(priority_mana_taps) != 2
            or not all(isinstance(value, bool) for value in priority_mana_taps)
        ):
            raise ValueError("priority_mana_taps must contain two booleans")
        object.__setattr__(self, "priority_mana_taps", tuple(priority_mana_taps))
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
            seats = self.priority_mana_taps[: len(self.players)][index]
            return GameSetup(
                self.players[index],
                decklists=self.decklists,
                priority_mana_taps=(seats + (False, False))[:2],
            )
        return self.players[index]
