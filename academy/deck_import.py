"""Validated file payloads for the Academy setup editor and new-game request."""

from pydantic import BaseModel, ConfigDict, Field

from manabot.decklist import ImportedMatch, import_match, parse_decklist


class DeckFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(max_length=65_536)
    filename: str = Field(default="deck.txt", max_length=255)


class DeckImportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pack_key: str = Field(default="tla-pool-v1", max_length=100)
    ruleset: str = Field(default="development-v1", max_length=100)
    hero: DeckFileInput
    villain: DeckFileInput

    def admit(self) -> ImportedMatch:
        return import_match(
            parse_decklist(self.hero.source, self.hero.filename),
            parse_decklist(self.villain.source, self.villain.filename),
            self.pack_key,
            self.ruleset,
        )
