"""Validated file payloads for the Academy setup editor and new-game request."""

from pydantic import BaseModel, ConfigDict, Field

import managym


class DeckFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(max_length=managym.MAX_DECKLIST_BYTES)
    filename: str = Field(default="deck.txt", max_length=255)

    def decklist(self) -> managym.Decklist:
        return managym.Decklist(self.source, self.filename)


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
