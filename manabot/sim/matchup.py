"""Explicit authored deck-role matrices for world-bound experiment evaluation."""

from __future__ import annotations

from pydantic import Field, model_validator

from manabot.infra.config import MatchConfig
from manabot.infra.identity import StrictModel, canonical_sha256
from managym import WORLD_SETUP, WORLD_VERSION, Deck


class AuthoredMatchup(StrictModel):
    """Two native authored roles; labels never substitute for actual deck identity."""

    pack_key: str = Field(min_length=1)
    role_a: str = Field(min_length=1)
    role_b: str = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_roles(self) -> "AuthoredMatchup":
        if self.role_a == self.role_b:
            raise ValueError("a four-cell matrix requires two distinct deck roles")
        return self

    @property
    def content_suite(self) -> str:
        return f"{WORLD_VERSION}-authored-matrix-{canonical_sha256(self.model_dump())}"

    def decks(self, leg: int) -> tuple[str, str]:
        if not 0 <= leg < 8:
            raise ValueError("authored matrix leg must be between 0 and 7")
        a, b = self.role_a, self.role_b
        return ((a, b), (b, a), (a, a), (b, b))[leg // 2]

    def match(self, leg: int = 0) -> MatchConfig:
        setup = MatchConfig.authored(
            self.pack_key, *self.decks(leg), hero="arena-seat-0", villain="arena-seat-1"
        )
        # Research packs can register more than two decks, so the cross pair
        # also needs explicit selection. Native membership checks still apply.
        return setup.model_copy(update={"world": self.pack_key})


class DeckMatchup(StrictModel):
    """An explicit imported pair, with corpus-entry identity independent of content.

    Adjacent legs exchange policies. Legs 2/3 reverse the decks too, so each
    policy sees both decks on play and draw. Equal entries are valid mirrors.
    Native game creation remains the deck admission authority.
    """

    population_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: tuple[str, str]
    lists: tuple[Deck, Deck]

    @property
    def content_suite(self) -> str:
        return f"{WORLD_VERSION}-deck-matrix-{canonical_sha256(self.model_dump(mode='json'))}"

    def _indices(self, leg: int) -> tuple[int, int]:
        if not 0 <= leg < 8:
            raise ValueError("deck matrix leg must be between 0 and 7")
        return ((0, 1), (1, 0), (0, 0), (1, 1))[leg // 2]

    def decks(self, leg: int) -> tuple[str, str]:
        a, b = self._indices(leg)
        return self.entries[a], self.entries[b]

    def match(self, leg: int = 0) -> MatchConfig:
        a, b = self._indices(leg)
        first, second = self.lists[a], self.lists[b]
        return MatchConfig(
            world=WORLD_SETUP,
            hero=self.entries[a],
            villain=self.entries[b],
            hero_deck=first.main,
            villain_deck=second.main,
            hero_sideboard=first.sideboard,
            villain_sideboard=second.sideboard,
        )


SELECTED_SUITE = f"{WORLD_VERSION}-allies-lessons-v1"


def selected_decks(leg: int) -> tuple[str, str]:
    """Authored decks in seat order for a retained arena leg.

    Adjacent legs share the same setup/deal and exchange players. Legs 0–3
    cross the decks; 4–7 are mirrors. Execution and report admission share
    this layout so a leg cannot silently acquire a different deck meaning.
    """
    deck_pairs = (
        ("ur_lessons", "gw_allies"),
        ("gw_allies", "ur_lessons"),
        ("ur_lessons", "ur_lessons"),
        ("gw_allies", "gw_allies"),
    )
    if not 0 <= leg < 8:
        raise ValueError("selected arena leg must be between 0 and 7")
    return deck_pairs[leg // 2]


def selected_match() -> MatchConfig:
    return MatchConfig.authored(
        "ur-lessons-vs-gw-allies",
        "ur_lessons",
        "gw_allies",
        hero="arena-seat-0",
        villain="arena-seat-1",
    )
