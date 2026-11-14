"""The frozen TLA pool corpus: 17Lands trophy drafts as plain source data.

A `Pool` is the cards one player drafted (basic lands removed) and the human's
main deck from the draft's seventh win. `PoolCorpus` owns the frozen
train/held-out split over every source draft. This module imports nothing
from the engine or from training: whether a pool's cards are playable belongs
to whoever plays it (`managym.unsupported_cards`), and loading never fetches,
repairs or resamples data.
"""

from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from manabot.infra.identity import SHA256_PATTERN, StrictModel, canonical_sha256

Count = Annotated[int, Field(strict=True, gt=0)]
PoolSplit = Literal["train", "held_out"]
SPLIT_RULE = "mtg-282-held-out-v1"


def held_out_ids(pool_ids: Iterable[str], count: int) -> frozenset[str]:
    """The first `count` drafts by SHA256 of the salted id, ties by id."""
    ranked = sorted(
        pool_ids,
        key=lambda pool: (sha256(f"{SPLIT_RULE}:{pool}".encode()).hexdigest(), pool),
    )
    return frozenset(ranked[:count])


class PoolProvenance(StrictModel):
    source_corpus_sha256: str = Field(pattern=SHA256_PATTERN)
    source_dump_sha256: str = Field(pattern=SHA256_PATTERN)
    url: str
    attribution: str
    license: str
    held_out_count: Count
    # Names treated as basic lands when they were removed from pools.
    basic_lands: tuple[str, ...] = Field(min_length=1)
    # The player's 17Lands build index of each human build, by pool id.
    human_build_indexes: dict[str, Annotated[int, Field(ge=0)]]


class Pool(StrictModel):
    id: str = Field(min_length=1)
    cards: dict[str, Count]
    human_build: dict[str, Count]
    split: PoolSplit


class PoolCorpus(StrictModel):
    schema_version: Literal[2] = 2
    split_rule: Literal["mtg-282-held-out-v1"] = SPLIT_RULE
    provenance: PoolProvenance
    pools: tuple[Pool, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def frozen_split(self) -> "PoolCorpus":
        ids = [pool.id for pool in self.pools]
        if len(set(ids)) != len(ids):
            raise ValueError("pool ids must be unique")
        if set(self.provenance.human_build_indexes) != set(ids):
            raise ValueError("every pool needs exactly one human build index")
        if not 0 < self.provenance.held_out_count < len(ids):
            raise ValueError("held-out count must leave training pools")
        held_out = held_out_ids(ids, self.provenance.held_out_count)
        basics = set(self.provenance.basic_lands)
        for pool in self.pools:
            if pool.split != ("held_out" if pool.id in held_out else "train"):
                raise ValueError(f"pool {pool.id}: split differs from the hash ranking")
            if basics & set(pool.cards):
                raise ValueError(f"pool {pool.id}: pools exclude basic lands")
            extra = sorted(
                name
                for name, count in pool.human_build.items()
                if name not in basics and count > pool.cards.get(name, 0)
            )
            if extra:
                raise ValueError(
                    f"pool {pool.id}: human build exceeds its pool: {', '.join(extra)}"
                )
        return self

    @property
    def digest(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))

    @classmethod
    def read(cls, path: Path) -> "PoolCorpus":
        return cls.model_validate_json(path.read_bytes())
