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
    ruleset: str = Field(default="development-v1", max_length=100)
    hero: DeckFileInput
    villain: DeckFileInput

    def admit(self) -> managym.ImportedMatch:
        """Raises `managym.DeckImportError` with each failing seat's diagnosed decklist."""
        return managym.import_match(
            self.hero.decklist(), self.villain.decklist(), self.pack_key, self.ruleset
        )
