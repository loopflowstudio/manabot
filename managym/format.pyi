"""Format construction rules (`managym/src/format/`), separate from the cards a
build draws from. A native submodule: reach it as ``managym.format``.

Array forms read and return NumPy arrays over the card axis: index ``i`` is
``card_def_id`` ``i``, with ``CARD_DEFINITION_COUNT`` entries."""

import numpy as np

from . import CardPool, DeckIssue

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
