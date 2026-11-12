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
Checkpoints never know the source; they bind only the world.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from manabot.infra.config import MatchConfig
from manabot.infra.identity import StrictModel, canonical_sha256
from manabot.model.world import admit_world_decks
import managym
from managym import Deck


class UniformDecks(StrictModel):
    """Uniform population entries in experiment order, with explicit pair semantics.

    Independent sampling keeps duplicate lists from different sources as
    separate entries. Optional stable entry IDs distinguish them in game
    receipts; deck content identity remains owned by `Deck`. Optional aligned
    `pools` and `builds` name each deck's pool and build (None: the human
    build) for public game results; a deck without a pool has pool None.
    """

    decks: tuple[Deck, ...] = Field(min_length=1)
    sampling: Literal["uniform-distinct-v1", "uniform-independent-v1"] = Field(
        default="uniform-distinct-v1",
        exclude_if=lambda value: value == "uniform-distinct-v1",
    )
    entry_ids: tuple[str, ...] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    pools: tuple[str | None, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    builds: tuple[str | None, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def valid_population(self) -> UniformDecks:
        if (self.pools is None) != (self.builds is None):
            raise ValueError("pools and builds are declared together")
        if self.pools is not None and self.builds is not None:
            if len(self.pools) != len(self.decks) or len(self.builds) != len(
                self.decks
            ):
                raise ValueError("pools and builds must align with decks")
            if any(
                value is not None and not value.strip()
                for value in (*self.pools, *self.builds)
            ):
                raise ValueError("pool and build ids must be nonempty")
            if any(
                pool is None and build is not None
                for pool, build in zip(self.pools, self.builds, strict=True)
            ):
                raise ValueError("a build requires its pool")
            pooled = [pair for pair in self.seats() if pair[0] is not None]
            if len(set(pooled)) != len(pooled):
                raise ValueError("each (pool, build) pair must be unique")
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

    def seats(self) -> list[tuple[str | None, str | None]]:
        """Each deck's (pool, build), in population order."""
        if self.pools is None or self.builds is None:
            return [(None, None)] * len(self.decks)
        return list(zip(self.pools, self.builds, strict=True))

    def _player_names(self) -> list[str | None]:
        """Native player names, which game results join back on; None keeps the deck's."""
        if self.entry_ids is not None:
            return list(self.entry_ids)
        return [
            None if pool is None else pool if build is None else f"{pool}/{build}"
            for pool, build in self.seats()
        ]

    def players(self) -> list[managym.PlayerConfig]:
        players = [deck.player() for deck in self.decks]
        for player, name in zip(players, self._player_names(), strict=True):
            if name is not None:
                player.name = name
        return players

    def seats_by_player(self) -> dict[str, tuple[str, str | None]]:
        """Pooled decks' (pool, build) by native player name."""
        return {
            name: (pool, build)
            for name, (pool, build) in zip(
                self._player_names(), self.seats(), strict=True
            )
            if pool is not None and name is not None
        }

    def match_config(self) -> MatchConfig:
        """A representative root; sampled roots come from the native draw."""
        players = self.players()
        return pair_match(players[0], players[1] if len(players) > 1 else players[0])


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
        world=first.world,
    )
