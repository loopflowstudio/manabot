"""Server-scoped deck pools and named native deck serialization.

Pools let any caller (training loops, scripts, or external trainers) register
collections of ``managym.Deck`` objects that Academy can sample fresh matchups
from without importing any training framework or corpus format.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import random
import re
from typing import Any

import managym
from managym.disclosure import DecklistDisclosure

POOL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
POOL_PREFIX = "pool:"


@dataclass(frozen=True)
class NamedDeck:
    """One native deck paired with a display label for Academy."""

    name: str
    deck: managym.Deck

    def __post_init__(self) -> None:
        cleaned = self.name.strip()
        if not cleaned or len(cleaned) > 120:
            raise ValueError("Deck name must be 1 to 120 non-whitespace characters.")
        if not isinstance(self.deck, managym.Deck):
            raise ValueError("NamedDeck.deck must be a managym.Deck.")


@dataclass(frozen=True)
class DeckPool:
    """Named population of decks Academy can sample fresh matchups from."""

    id: str
    label: str
    decks: tuple[NamedDeck, ...]
    default_decklists: DecklistDisclosure = "closed"

    def __post_init__(self) -> None:
        if not POOL_ID_PATTERN.match(self.id):
            raise ValueError(
                "Deck pool id must contain only ASCII letters, digits, '.', '_' or '-'."
            )
        if not self.label.strip():
            raise ValueError("Deck pool label must be non-empty.")
        if not self.decks:
            raise ValueError(f"Deck pool {self.id!r} must contain at least one deck.")

    @property
    def key(self) -> str:
        return f"{POOL_PREFIX}{self.id}"

    def sample(self, rng: random.Random) -> NamedDeck:
        return self.decks[rng.randrange(len(self.decks))]

    def sample_pair(self, rng: random.Random) -> tuple[NamedDeck, NamedDeck]:
        """Draw two distinct entries when the pool holds at least two decks."""
        if len(self.decks) == 1:
            return self.decks[0], self.decks[0]
        first, second = rng.sample(range(len(self.decks)), 2)
        return self.decks[first], self.decks[second]


def deck_to_source(deck: managym.Deck) -> str:
    """Format a native deck into Arena-style text for native DeckImport admission."""
    lines = [f"{count} {name}" for name, count in sorted(deck.main.items())]
    if deck.sideboard:
        lines.append("Sideboard")
        lines.extend(
            f"{count} {name}" for name, count in sorted(deck.sideboard.items())
        )
    return "\n".join(lines) + "\n"


def deck_file_payload(
    deck: managym.Deck, *, name: str | None = None, filename: str | None = None
) -> dict[str, str]:
    """Build a DeckFileInput dictionary from a native managym.Deck."""
    payload: dict[str, str] = {
        "source": deck_to_source(deck),
        "filename": filename or f"{deck.identity[:10]}.txt",
    }
    if name is not None:
        payload["name"] = name
    return payload


_pools: dict[str, DeckPool] = {}


def set_deck_pools(pools: Sequence[DeckPool] | None) -> None:
    """Replace the server-scoped deck pool registry."""
    _pools.clear()
    if not pools:
        return
    for pool in pools:
        if pool.id in _pools:
            raise ValueError(f"Deck pool {pool.id!r} is already registered.")
        _pools[pool.id] = pool


def deck_pools() -> dict[str, DeckPool]:
    return dict(_pools)


def is_pool_key(value: object) -> bool:
    return isinstance(value, str) and value.startswith(POOL_PREFIX)


def resolve_deck_pool(key: str) -> DeckPool | None:
    pool_id = key.removeprefix(POOL_PREFIX) if key.startswith(POOL_PREFIX) else key
    return _pools.get(pool_id)


def deck_pool_catalog() -> list[dict[str, Any]]:
    return [
        {
            "id": pool.id,
            "key": pool.key,
            "label": pool.label,
            "count": len(pool.decks),
            "default_decklists": pool.default_decklists,
        }
        for pool in _pools.values()
    ]
