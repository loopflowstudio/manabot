"""Validated file payloads for the Academy setup editor and new-game request."""

from pydantic import BaseModel, ConfigDict, Field

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
