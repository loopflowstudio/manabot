"""Format construction rules (`managym/src/format/`), separate from the cards a
build draws from. A native submodule: reach it as ``managym.format``.

Array forms read and return NumPy arrays over the card axis: index ``i`` is
``card_def_id`` ``i``, with ``CARD_DEFINITION_COUNT`` entries."""

from typing import Literal, TypedDict

import numpy as np

from . import CardPool, DeckIssue

class SetPrinting(TypedDict):
    card_def_id: int
    key: str
    definition_scryfall_id: str
    oracle_id: str
    name: str
    set: str
    scryfall_id: str
    rarity: Literal["common", "uncommon", "rare", "mythic"]

def set_printings(set: str) -> list[SetPrinting]:
    """Complete checked nonbasic printing inventory. Uncurated sets raise ValueError."""

class RotisserieObservation(TypedDict):
    episode_id: str
    set: str
    printing_digest: str
    revision: int
    picks_per_seat: int
    chooser: int | None
    initial: list[int]
    remaining: list[int]
    pools: list[list[int]]
    legal: list[int]

class Rotisserie:
    """One public shared-supply snake draft; count rows use the full catalog axis."""
    def __init__(
        self, set: str, episode_id: str, seats: int = 8, picks_per_seat: int = 45
    ) -> None: ...
    def observe(self) -> RotisserieObservation:
        """Owned snapshot of every pool, supply and legal definition; chooser None ends drafting.

        The round is revision // len(pools). Editing the snapshot cannot change the draft.
        """
    def pick(self, episode_id: str, revision: int, chooser: int, card_def_id: int) -> None:
        """Commit one copy. Wrong episode/chooser, stale revision or unavailable card rejects atomically."""
    def pool(self, seat: int) -> CardPool:
        """Completed seat pool for Limited construction; rejects during drafting."""

class Limited:
    """Limited rules: at least 40 cards, non-basics within pool counts, basics free."""
    def __init__(self) -> None: ...
    def build(self, pool: CardPool, max_cards: int = 60) -> LimitedBuild: ...
    def issues(self, pool: CardPool, deck: dict[str, int]) -> list[DeckIssue]: ...
    def available(
        self, pools: np.ndarray, decks: np.ndarray, max_cards: int = 60
    ) -> tuple[np.ndarray, np.ndarray]:
        """Copies still addable (int32 ``[n, cards]``) and ``can_stop`` (bool ``[n]``).

        Raises ``ValueError`` naming the row for a deck no build could reach."""
    def legal(self, pools: np.ndarray, decks: np.ndarray) -> np.ndarray:
        """Whether each deck passes the rules, with no cap (bool ``[n]``)."""
    @property
    def min_cards(self) -> int: ...
    @property
    def basics(self) -> np.ndarray:
        """Cards a pool does not limit, bool ``[cards]``."""
    @staticmethod
    def basic_cards() -> list[str]: ...

class LimitedBuild:
    """One deck being built from one pool; each selection commits."""
    def select(self, card: str) -> None: ...
    @property
    def available(self) -> dict[str, int]:
        """Each card that can still be selected, with its remaining copies."""
    @property
    def can_stop(self) -> bool: ...
    @property
    def deck(self) -> dict[str, int]: ...
