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
    pack_key: str = Field(default="world", max_length=100)
    hero: DeckFileInput
    villain: DeckFileInput

    def admit(self) -> tuple[managym.DeckImport, managym.DeckImport]:
        """The admitted `(hero, villain)` imports. Raises `managym.DeckImportError`
        with each failing seat's diagnosed decklist."""
        hero, villain = managym.admit_decklists(
            [("hero", self.hero.decklist()), ("villain", self.villain.decklist())],
            self.pack_key,
        )
        return hero, villain
