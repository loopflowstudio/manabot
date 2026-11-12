"""Derive the Limited corpus from retained trophy events, never a fresh scrape."""

import argparse
from collections import Counter
from pathlib import Path

from experiments.tla_corpus import Corpus
from manabot import managym
from manabot.infra.files import atomic_json
from manabot.infra.identity import file_sha256
from manabot.training.pools import (
    PoolCorpus,
    PoolEntry,
    PoolExclusion,
    PoolProvenance,
    held_out_events,
)


def derive_pools(source_path: Path, held_out: int = 100) -> PoolCorpus:
    source = Corpus.model_validate_json(source_path.read_bytes())
    if len(source.trophies) != source.target:
        raise ValueError(
            "retained source does not contain its entire selected population"
        )
    if not 0 < held_out < source.target:
        raise ValueError(
            "held-out count must leave nonempty training and evaluation partitions"
        )
    event_ids = [t.event_id for t in source.trophies]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("source event IDs must be unique")
    held_out_ids = held_out_events(event_ids, held_out)
    basics = set(managym.LimitedRules.basic_cards())
    entries: list[PoolEntry] = []
    exclusions: list[PoolExclusion] = []
    for trophy in source.trophies:
        # Reparse unchanged text: historical admission issues are evidence, not
        # current eligibility. Only unknown names qualify for this exclusion.
        try:
            [imported] = managym.admit_decklists(
                [
                    (
                        trophy.event_id,
                        managym.Decklist(
                            trophy.decklist.source, trophy.decklist.filename
                        ),
                    )
                ]
            )
        except managym.DeckImportError as error:
            issues = error.decklists[trophy.event_id].issues
            if not issues or any(
                i.code != "unknown" or i.card_name is None for i in issues
            ):
                raise
            exclusions.append(
                PoolExclusion(
                    event_id=trophy.event_id,
                    unknown_names=tuple(
                        sorted({i.card_name for i in issues if i.card_name is not None})
                    ),
                )
            )
            continue
        counts = Counter(imported.deck.main) + Counter(imported.deck.sideboard)
        entries.append(
            PoolEntry(
                event_id=trophy.event_id,
                entry_id=trophy.source_id,
                pool={
                    name: count
                    for name, count in sorted(counts.items())
                    if name not in basics
                },
                reference=managym.Deck(imported.deck.main),
                split="held_out" if trophy.event_id in held_out_ids else "train",
            )
        )
    excluded_ids = {entry.event_id for entry in exclusions}
    return PoolCorpus(
        provenance=PoolProvenance(
            source_corpus_sha256=file_sha256(source_path),
            source_dump_sha256=source.source.sha256,
            url=source.source.url,
            attribution=source.source.attribution,
            license=source.source.license,
            source_event_count=source.target,
            held_out_count=held_out,
            excluded_events=tuple(exclusions),
        ),
        world=managym.world_identity(),
        held_out_event_ids=tuple(e for e in held_out_ids if e not in excluded_ids),
        entries=tuple(entries),
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
        # Native diagnostics retain the event, source line and rejection reason.
        raise SystemExit(f"No corpus written: {error}") from None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, corpus.model_dump(mode="json"))
    print(
        f"{len(corpus.entries)} events; {len(corpus.held_out_event_ids)} held out; "
        f"{len(corpus.provenance.excluded_events)} excluded; sha256={corpus.digest}"
    )


if __name__ == "__main__":
    main()
