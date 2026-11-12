"""Derive the TLA pool corpus from retained trophy drafts, never a fresh scrape.

Every source draft becomes a pool, including drafts with cards the installed
engine does not define: playability is a consumer's filter, so defining more
cards never moves a pool between splits. Malformed lists fail derivation.
"""

import argparse
from collections import Counter
from pathlib import Path

from experiments.tla_corpus import Corpus
from manabot import managym
from manabot.infra.files import atomic_json
from manabot.infra.identity import file_sha256
from manabot.pools import Pool, PoolCorpus, PoolProvenance, held_out_ids


def _entries(event_id: str, decklist: managym.Decklist) -> list[managym.DeckEntry]:
    """Parsed entries of unchanged source text; only unknown names may fail."""
    # Reparse: historical admission issues are evidence, not current diagnostics.
    reparsed = managym.Decklist(decklist.source, decklist.filename)
    try:
        managym.admit_decklists([(event_id, reparsed)])
    except managym.DeckImportError as error:
        issues = error.decklists[event_id].issues
        if not issues or any(i.code != "unknown" for i in issues):
            raise
    return reparsed.entries


def derive_pools(source_path: Path, held_out: int = 100) -> PoolCorpus:
    source = Corpus.model_validate_json(source_path.read_bytes())
    if len(source.trophies) != source.target:
        raise ValueError(
            "retained source does not contain its entire selected population"
        )
    ids = [t.event_id for t in source.trophies]
    if len(set(ids)) != len(ids):
        raise ValueError("source draft ids must be unique")
    if not 0 < held_out < len(ids):
        raise ValueError(
            "held-out count must leave nonempty training and evaluation splits"
        )
    evaluation = held_out_ids(ids, held_out)
    basics = managym.format.Limited.basic_cards()
    pools: list[Pool] = []
    for trophy in source.trophies:
        entries = _entries(trophy.event_id, trophy.decklist)
        main: Counter[str] = Counter()
        cards: Counter[str] = Counter()
        for entry in entries:
            cards[entry.name] += entry.count
            if entry.section == "main":
                main[entry.name] += entry.count
        pools.append(
            Pool(
                id=trophy.event_id,
                cards={n: c for n, c in sorted(cards.items()) if n not in basics},
                human_build=dict(sorted(main.items())),
                split="held_out" if trophy.event_id in evaluation else "train",
            )
        )
    return PoolCorpus(
        provenance=PoolProvenance(
            source_corpus_sha256=file_sha256(source_path),
            source_dump_sha256=source.source.sha256,
            url=source.source.url,
            attribution=source.source.attribution,
            license=source.source.license,
            held_out_count=held_out,
            basic_lands=tuple(basics),
            human_build_indexes={
                t.event_id: t.games[-1].build_index for t in source.trophies
            },
        ),
        pools=tuple(pools),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--held-out", type=int, default=100)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(
            f"No corpus written: {args.out} exists; choose a new versioned path"
        )
    try:
        corpus = derive_pools(args.source, args.held_out)
    except managym.DeckImportError as error:
        # Native diagnostics retain the draft, source line and rejection reason.
        raise SystemExit(f"No corpus written: {error}") from None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, corpus.model_dump(mode="json"))
    held = sum(pool.split == "held_out" for pool in corpus.pools)
    print(f"{len(corpus.pools)} pools; {held} held out; sha256={corpus.digest}")


if __name__ == "__main__":
    main()
