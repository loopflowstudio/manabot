"""Decks a training run samples, checked against the run's world before use.

A deck is its main-deck and sideboard counts (`managym.Deck`). Source files
(`managym.Decklist`) are provenance the run retains; they never make a
different kind of deck.
`TrainingDecks` is one run's fixed supply for uniform-distinct-v1 roots: every
list is admitted by the content pool when the run is configured, so bad input
fails before training. Native game creation admits each submitted deck again,
so membership here is a record of what the run supplied, not a permission.
Checkpoints never contain this record; they bind only the world.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import Field, model_validator

from manabot.infra.config import MatchConfig
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.model.world import admit_pool_decks
import managym
from managym import Deck, Decklist


class TrainingDecks(StrictModel):
    """Distinct decks in identity order, with every source record that supplied them."""

    pool: str
    decks: tuple[Deck, ...] = Field(min_length=2)
    sources: tuple[Decklist, ...] = ()

    @model_validator(mode="after")
    def distinct_in_identity_order(self) -> TrainingDecks:
        ids = [deck.identity for deck in self.decks]
        if ids != sorted(set(ids)):
            raise ValueError("uniform-distinct-v1 requires sorted distinct decks")
        return self

    @property
    def digest(self) -> str:
        """Run recovery identity of the supply, including its provenance."""
        return canonical_sha256(self.model_dump(mode="json"))

    @classmethod
    def from_decklists(cls, decklists: Sequence[Decklist], pool: str) -> TrainingDecks:
        """Reject the whole input on any error; source multiplicity never weights draws.

        Raises `managym.DeckImportError` keyed by each failing decklist's position.
        """
        imports = managym.admit_decklists(
            [(str(index), decklist) for index, decklist in enumerate(decklists)], pool
        )
        admitted = {imported.deck.identity: imported.deck for imported in imports}
        if len(admitted) < 2:
            raise ValueError("uniform-distinct-v1 requires at least two distinct decks")
        return cls(
            pool=pool,
            decks=tuple(deck for _, deck in sorted(admitted.items())),
            sources=tuple(
                sorted(
                    (imported.decklist for imported in imports),
                    key=lambda record: (record.filename, record.source),
                )
            ),
        )

    @classmethod
    def from_files(cls, paths: Sequence[str | Path], pool: str) -> TrainingDecks:
        return cls.from_decklists(
            [managym.Decklist.read(Path(path), str(path)) for path in paths], pool
        )

    def check(self) -> None:
        """Admit every deck in the current world before any root is allocated."""
        admit_pool_decks(self.pool, self.players())

    def players(self) -> list[managym.PlayerConfig]:
        return [deck.player(self.pool) for deck in self.decks]

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
