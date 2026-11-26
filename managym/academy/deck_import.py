"""Validated file payloads for the Academy setup editor and new-game request."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

import managym


class DeckFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(max_length=managym.MAX_DECKLIST_BYTES)
    filename: str = Field(default="deck.txt", max_length=255)
    name: str | None = Field(default=None, max_length=120)

    def decklist(self) -> managym.Decklist:
        return managym.Decklist(self.source, self.filename)

    def display_name(self) -> str:
        if self.name is not None and self.name.strip():
            return self.name.strip()
        return "imported"


class DeckImportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hero: DeckFileInput
    villain: DeckFileInput

    def admit(self) -> tuple[managym.DeckImport, managym.DeckImport]:
        """The `(hero, villain)` imports the installed world admits. Raises
        `managym.DeckImportError` with each failing seat's diagnosed decklist."""
        hero, villain = managym.admit_decklists(
            [("hero", self.hero.decklist()), ("villain", self.villain.decklist())]
        )
        return hero, villain


class DeckImportSides(BaseModel):
    """A request's imported lists before setup fills a side that names a
    bundled deck or a pool: one side or both."""

    model_config = ConfigDict(extra="forbid")
    hero: DeckFileInput | None = None
    villain: DeckFileInput | None = None

    @model_validator(mode="after")
    def names_a_side(self) -> Self:
        if self.hero is None and self.villain is None:
            raise ValueError("deck_import needs a hero or villain list.")
        return self
