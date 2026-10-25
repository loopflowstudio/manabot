"""Explicit authored deck-role matrices for world-bound experiment evaluation."""

from __future__ import annotations

from pydantic import Field, model_validator

from manabot.infra.config import MatchConfig
from manabot.infra.identity import StrictModel, canonical_sha256
from managym import WORLD_VERSION


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
        return setup.model_copy(update={"content_pack": self.pack_key})


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
