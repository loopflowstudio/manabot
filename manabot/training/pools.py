"""Frozen Limited pools and human reference builds, joined by source event.

The corpus owns the split. Loading validates the saved partition and native
construction rules; it never fetches data, repairs lists or resamples events.
"""

from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from manabot import managym
from manabot.infra.identity import SHA256_PATTERN, StrictModel, canonical_sha256
from manabot.training.decks import UniformDecks

Count = Annotated[int, Field(strict=True, gt=0)]
PoolSplit = Literal["train", "held_out"]


def held_out_events(event_ids: Iterable[str], count: int) -> tuple[str, ...]:
    """Reserve evaluation events from the full source, before world admission."""
    return tuple(
        sorted(
            event_ids,
            key=lambda event: (
                sha256(f"mtg-282-held-out-v1:{event}".encode()).hexdigest(),
                event,
            ),
        )[:count]
    )


class PoolExclusion(StrictModel):
    event_id: str = Field(min_length=1)
    unknown_names: tuple[str, ...] = Field(min_length=1)


class PoolProvenance(StrictModel):
    source_corpus_sha256: str = Field(pattern=SHA256_PATTERN)
    source_dump_sha256: str = Field(pattern=SHA256_PATTERN)
    url: str
    attribution: str
    license: str
    source_event_count: Count
    held_out_count: Count
    excluded_events: tuple[PoolExclusion, ...]


class PoolEntry(StrictModel):
    event_id: str = Field(min_length=1)
    entry_id: str = Field(min_length=1)
    pool: dict[str, Count]
    reference: managym.Deck
    split: PoolSplit

    @model_validator(mode="after")
    def legal_reference(self) -> "PoolEntry":
        if self.reference.sideboard:
            raise ValueError(
                "reference is the human main deck; unpicked cards belong to the pool"
            )
        if set(self.pool) & set(managym.LimitedRules.basic_cards()):
            raise ValueError("pool corpus excludes basic lands")
        issues = managym.LimitedRules(self.pool).check(self.reference.main)
        if issues:
            raise ValueError("; ".join(issue.message for issue in issues))
        return self


class PoolCorpus(StrictModel):
    schema_version: Literal[1] = 1
    provenance: PoolProvenance
    world: dict[str, str]
    split_rule: Literal["sha256-mtg-282-held-out-v1"] = "sha256-mtg-282-held-out-v1"
    held_out_event_ids: tuple[str, ...]
    entries: tuple[PoolEntry, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def partition(self) -> "PoolCorpus":
        events = {entry.event_id for entry in self.entries}
        if len(events) != len(self.entries) or len(
            {e.entry_id for e in self.entries}
        ) != len(self.entries):
            raise ValueError(
                "pool corpus requires unique events and population entries"
            )
        held_out = set(self.held_out_event_ids)
        if (
            len(held_out) != len(self.held_out_event_ids)
            or not held_out < events
            or not held_out
        ):
            raise ValueError(
                "held-out partition must be nonempty, unique and leave training events"
            )
        if held_out != {e.event_id for e in self.entries if e.split == "held_out"}:
            raise ValueError("entry splits differ from frozen held-out event IDs")
        excluded = {e.event_id for e in self.provenance.excluded_events}
        if len(excluded) != len(self.provenance.excluded_events) or events & excluded:
            raise ValueError("excluded events must be unique and not admitted")
        source_events = events | excluded
        if len(source_events) != self.provenance.source_event_count:
            raise ValueError("admitted and excluded events must cover the source")
        if self.provenance.held_out_count >= len(source_events):
            raise ValueError("held-out reservation must leave training events")
        reserved = held_out_events(source_events, self.provenance.held_out_count)
        if self.held_out_event_ids != tuple(e for e in reserved if e in events):
            raise ValueError("held-out partition differs from the source hash ranking")
        return self

    @property
    def digest(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))

    def population(self, split: PoolSplit) -> UniformDecks:
        """Human reference population in corpus order, retaining event identities."""
        entries = [entry for entry in self.entries if entry.split == split]
        return UniformDecks(
            decks=tuple(entry.reference for entry in entries),
            entry_ids=tuple(entry.entry_id for entry in entries),
            sampling="uniform-independent-v1",
        )

    @classmethod
    def read(cls, path: Path) -> "PoolCorpus":
        corpus = cls.model_validate_json(path.read_bytes())
        if corpus.world != managym.world_identity():
            raise ValueError(
                "pool corpus world differs from native runtime; derive a new corpus"
            )
        return corpus
