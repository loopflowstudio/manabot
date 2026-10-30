"""Deck sources: how a training run gets the decks for each new game.

A deck is its main-deck and sideboard counts (`managym.Deck`). A run has a deck
source, not a deck list or files: the experiment builds the source in Python
however it likes (reading decklists, generating decks) and places it in the
regime. Each new game asks the source for its two decks; game creation admits
every deck natively, so a source is never a permission.

`UniformDecks` is the one source so far (`uniform-distinct-v1`): every new game
draws two different decks uniformly from a small list, in a native draw seeded
by the game's reset seed. Exact resume therefore reproduces the same decks for
the same games, and the run also records each game's actual decks.
Checkpoints never know the source; they bind only the world.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from manabot.infra.config import MatchConfig
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.model.world import admit_world_decks
import managym
from managym import Deck


class UniformDecks(StrictModel):
    """uniform-distinct-v1 over distinct decks, in the experiment's order."""

    decks: tuple[Deck, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def distinct(self) -> UniformDecks:
        if len({deck.identity for deck in self.decks}) != len(self.decks):
            raise ValueError("uniform-distinct-v1 requires distinct decks")
        return self

    @property
    def digest(self) -> str:
        """Run recovery identity of the source; list order selects each draw."""
        return canonical_sha256(self.model_dump(mode="json"))

    def check(self) -> None:
        """Admit every deck in the current world before any root is allocated."""
        admit_world_decks(self.players())

    def players(self) -> list[managym.PlayerConfig]:
        return [deck.player() for deck in self.decks]

    def match_config(self) -> MatchConfig:
        """A representative root; sampled roots come from the native draw."""
        return pair_match(*self.players()[:2])


def pair_match(
    first: managym.PlayerConfig, second: managym.PlayerConfig
) -> MatchConfig:
    """Record an actual native root's decks in seat order."""
    return MatchConfig(
        hero=first.name,
        villain=second.name,
        hero_deck=first.decklist,
        villain_deck=second.decklist,
        hero_sideboard=first.sideboard,
        villain_sideboard=second.sideboard,
        content_pack=first.content_pack,
    )
