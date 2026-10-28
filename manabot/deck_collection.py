"""Versioned complete decks for reproducible native training roots.

Preparation admits every source through decklist's shared pool contract, then
collapses equal normalized lists without losing their source records. Saved
manifests bind membership, provenance, content and the whole policy vocabulary;
training embeds the manifest rather than retaining a mutable filesystem pointer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from manabot.decklist import (
    AdmittedDeck,
    ContentPool,
    DeckDocument,
    DeckImportError,
    ImportedMatch,
    PoolIdentity,
    admit_document,
    deck_identity,
    parse_decklist,
)
from manabot.infra.identity import StrictModel, canonical_sha256

if TYPE_CHECKING:
    from manabot.infra.config import MatchConfig
    import managym


class CollectionDeck(StrictModel):
    deck_id: str
    main: dict[str, int]
    sideboard: dict[str, int]
    sources: tuple[DeckDocument, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identity_matches(self) -> CollectionDeck:
        if self.deck_id != deck_identity(self.main, self.sideboard):
            raise ValueError("collection deck identity differs from its complete list")
        return self


class DeckCollection(StrictModel):
    schema_version: Literal["deck-collection-v1"] = "deck-collection-v1"
    sampling_policy: Literal["uniform-distinct-v1"] = "uniform-distinct-v1"
    pool: PoolIdentity
    ruleset: Literal["development-v1"] = "development-v1"
    learning_schema_hash: str
    decks: tuple[CollectionDeck, ...] = Field(min_length=2)
    digest: str

    @model_validator(mode="after")
    def membership_matches(self) -> DeckCollection:
        ids = [deck.deck_id for deck in self.decks]
        if ids != sorted(set(ids)):
            raise ValueError("collection requires sorted distinct normalized decks")
        if self.digest != canonical_sha256(
            self.model_dump(mode="json", exclude={"digest"})
        ):
            raise ValueError("deck collection digest mismatch")
        return self

    @classmethod
    def load(cls, path: str | Path) -> DeckCollection:
        collection = cls.model_validate_json(Path(path).read_bytes())
        collection.validate_current()
        return collection

    def validate_current(self) -> None:
        """Re-admit sources and exact content/schema before allocating any roots."""
        rebuilt = prepare_collection(
            [source for deck in self.decks for source in deck.sources],
            self.pool.pack_key,
        )
        if rebuilt != self:
            raise ValueError(
                "deck collection content, source or learning schema mismatch"
            )

    def roster(self) -> list[managym.PlayerConfig]:
        import managym

        return [
            managym.PlayerConfig(
                deck.deck_id, deck.main, deck.sideboard, self.pool.pack_key
            )
            for deck in self.decks
        ]

    def match_config(self, first: int = 0, second: int = 1) -> MatchConfig:
        """Resolve a sampled pair through the same private import record as file play."""
        left, right = self.decks[first], self.decks[second]
        imported = ImportedMatch(
            pool=self.pool,
            ruleset=self.ruleset,
            hero=AdmittedDeck(
                document=left.sources[0], main=left.main, sideboard=left.sideboard
            ),
            villain=AdmittedDeck(
                document=right.sources[0], main=right.main, sideboard=right.sideboard
            ),
        )
        return imported.match_config().model_copy(
            update={"hero": left.deck_id, "villain": right.deck_id}
        )


def prepare_collection(documents: list[DeckDocument], pack_key: str) -> DeckCollection:
    """Reject the whole input on any error; source multiplicity never weights draws."""
    from manabot.semantic.policy_catalog import bind_policy_catalog

    pool = ContentPool.load(pack_key)
    admitted: dict[str, AdmittedDeck] = {}
    sources: dict[str, list[DeckDocument]] = defaultdict(list)
    errors: dict[str, DeckDocument] = {}
    for index, source in enumerate(documents):
        result = admit_document(
            parse_decklist(source.source, source.filename), pool, "development-v1"
        )
        if isinstance(result, DeckDocument):
            errors[str(index)] = result
            continue
        identity = deck_identity(result.main, result.sideboard)
        admitted[identity] = result
        sources[identity].append(result.document)
    if errors:
        raise DeckImportError(errors)
    if len(admitted) < 2:
        raise ValueError("uniform-distinct-v1 requires at least two distinct decks")
    policy = bind_policy_catalog(pack_key)
    decks = tuple(
        CollectionDeck(
            deck_id=identity,
            main=deck.main,
            sideboard=deck.sideboard,
            sources=tuple(
                sorted(
                    sources[identity],
                    key=lambda source: (source.filename, source.source),
                )
            ),
        )
        for identity, deck in sorted(admitted.items())
    )
    payload = {
        "schema_version": "deck-collection-v1",
        "sampling_policy": "uniform-distinct-v1",
        "pool": pool.identity().model_dump(mode="json"),
        "ruleset": "development-v1",
        "learning_schema_hash": policy.schema.schema_hash,
        "decks": [deck.model_dump(mode="json") for deck in decks],
    }
    return DeckCollection.model_validate(
        {**payload, "digest": canonical_sha256(payload)}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    documents = [
        parse_decklist(path.read_bytes().decode("utf-8"), str(path))
        for path in args.files
    ]
    collection = prepare_collection(documents, args.pool)
    # Never overwrite an admitted artifact; callers choose a new version's path.
    with args.out.open("x", encoding="utf-8") as target:
        target.write(collection.model_dump_json(indent=2) + "\n")
    print(f"{len(collection.decks)} distinct decks; {collection.digest}; {args.out}")


if __name__ == "__main__":
    main()
