"""Derive the Limited corpus from retained trophy events, never a fresh scrape."""

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path

from experiments.tla_corpus import Corpus
from manabot import managym
from manabot.infra.files import atomic_json
from manabot.infra.identity import file_sha256
from manabot.training.pools import PoolCorpus, PoolEntry, PoolProvenance


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
    # Re-admit the original counted text, including formerly unsupported lists.
    # A failure rejects the entire derivation rather than selecting replacements.
    admitted = managym.admit_decklists(
        [
            (t.event_id, managym.Decklist(t.decklist.source, t.decklist.filename))
            for t in source.trophies
        ]
    )
    held_out_ids = tuple(
        sorted(
            (t.event_id for t in source.trophies),
            key=lambda event: (
                sha256(f"mtg-282-held-out-v1:{event}".encode()).hexdigest(),
                event,
            ),
        )[:held_out]
    )
    basics = set(managym.LimitedRules.basic_cards())
    entries: list[PoolEntry] = []
    for trophy, imported in zip(source.trophies, admitted, strict=True):
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
    return PoolCorpus(
        provenance=PoolProvenance(
            source_corpus_sha256=file_sha256(source_path),
            source_dump_sha256=source.source.sha256,
            url=source.source.url,
            attribution=source.source.attribution,
            license=source.source.license,
        ),
        world=managym.world_identity(),
        held_out_event_ids=held_out_ids,
        entries=tuple(entries),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--held-out", type=int, default=100)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("pool corpus is frozen; choose a new output path")
    try:
        corpus = derive_pools(args.source, args.held_out)
    except managym.DeckImportError as error:
        names = sorted(
            {
                issue.card_name
                for decklist in error.decklists.values()
                for issue in decklist.issues
                if issue.card_name is not None
            }
        )
        raise SystemExit(
            f"{len(error.decklists)} selected events fail current-world admission; "
            f"no corpus written. Cards: {', '.join(names)}"
        ) from None
    atomic_json(args.out, corpus.model_dump(mode="json"))
    print(
        f"{len(corpus.entries)} events; {len(corpus.held_out_event_ids)} held out; sha256={corpus.digest}"
    )


if __name__ == "__main__":
    main()
