"""Deck sources: how a training run gets the decks for each new game.

A deck is its main-deck and sideboard counts (`managym.Deck`). A run has a deck
source, not a deck list or files: the experiment builds the source in Python
however it likes (reading decklists, generating decks) and places it in the
regime. Each new game asks the source for its two decks; game creation admits
every deck natively, so a source is never a permission.

`UniformDecks` samples two population entries in a native draw seeded by the
game's reset seed. `uniform-independent-v1` draws each seat independently,
including mirrors; historical `uniform-distinct-v1` excludes mirrors. Exact
resume reproduces the same decks for the same games, and the run also records
each game's actual decks.
Sampling settings belong to the run.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TypedDict

from pydantic import Field, model_validator

from manabot.infra.config import MatchConfig
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.model.world import admit_world_decks
from manabot.model.writer_catalog import WriterCatalog
import managym
from managym import Deck
from managym.disclosure import DecklistSampling


class UniformDecks(StrictModel):
    """Uniform population entries in experiment order, with explicit pair semantics.

    Independent sampling keeps duplicate lists from different source events as
    separate entries. Optional stable entry IDs distinguish those events in game
    receipts; deck content identity remains owned by `Deck`. `decklists` selects
    a fixed disclosure or a balanced, seeded per-game draw ("mixed"). Open is
    omitted from serialized sources so their recovery identities remain unchanged.
    """

    decklists: DecklistSampling = Field(
        default="open", exclude_if=lambda value: value == "open"
    )
    decks: tuple[Deck, ...] = Field(min_length=1)
    sampling: Literal["uniform-distinct-v1", "uniform-independent-v1"] = Field(
        default="uniform-distinct-v1",
        exclude_if=lambda value: value == "uniform-distinct-v1",
    )
    entry_ids: tuple[str, ...] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def valid_population(self) -> UniformDecks:
        if self.sampling == "uniform-distinct-v1" and (
            len(self.decks) < 2
            or len({deck.identity for deck in self.decks}) != len(self.decks)
        ):
            raise ValueError("uniform-distinct-v1 requires distinct decks")
        if self.entry_ids is not None and (
            len(self.entry_ids) != len(self.decks)
            or len(set(self.entry_ids)) != len(self.entry_ids)
            or any(not name.strip() for name in self.entry_ids)
        ):
            raise ValueError(
                "deck population entry IDs must be nonempty, unique and aligned"
            )
        return self

    @property
    def digest(self) -> str:
        """Run recovery identity of the source; list order selects each draw."""
        return canonical_sha256(self.model_dump(mode="json"))

    def check(self) -> None:
        """Admit every deck in the current world before any root is allocated."""
        admit_world_decks(self.players())

    def players(self) -> list[managym.PlayerConfig]:
        players = [deck.player() for deck in self.decks]
        if self.entry_ids is not None:
            for player, name in zip(players, self.entry_ids, strict=True):
                player.name = name
        return players

    def match_config(self) -> MatchConfig:
        """A representative root; sampled roots come from the native draw."""
        players = self.players()
        return MatchConfig.from_players(
            players[0],
            players[1] if len(players) > 1 else players[0],
            decklists="closed" if self.decklists == "closed" else "open",
        )


class CorpusDeck(TypedDict):
    deck: str
    cards: list[str]


def corpus_decks(decks: Iterable[Deck], catalog: WriterCatalog) -> list[CorpusDeck]:
    keys = catalog["definition_keys"]
    names = {key: key for key in keys}
    names.update(
        {
            entry["name"]: keys[index]
            for entry in catalog["naming_catalog"]
            if (index := entry["definition"]) is not None
        }
    )
    return [
        CorpusDeck(
            deck=deck.identity,
            cards=sorted(
                {names[name] for name in deck.main.keys() | deck.sideboard.keys()}
            ),
        )
        for deck in decks
    ]
