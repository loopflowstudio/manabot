"""Play u1952 on the first eight trophy lists with a new card in the main deck.

The historical arm uses the committed recheck reference. This arm draws from
the same 17Lands source, including lists excluded by the writing revision.
"""

import argparse
import json
from pathlib import Path
import time

from experiments.tla_corpus import Corpus
from manabot.eval.recheck import ReferencePair, StrengthReference, measure_pairs
from manabot.infra.files import atomic_json
from manabot.infra.identity import file_sha256
from manabot.infra.source import source_commit
from manabot.model.writer_catalog import WriterCatalog
from manabot.sim.matchup import DeckMatchup
import managym

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiments/references/tla-frontier-small-u1952.json"
CORPUS = ROOT / "experiments/data/mtg213/corpus-v1/corpus.json"
WRITER = (
    ROOT
    / "manabot/model/writer_catalogs/0798b2c7c7f54b7329043b13cb1742d74fe21b35373820580b76f979e6b7eb5f.json"
)


def new_card_pairs() -> list[ReferencePair]:
    writer: WriterCatalog = json.loads(WRITER.read_bytes())
    old = set(writer["definition_keys"])
    decks: list[tuple[str, managym.Deck]] = []
    for trophy in Corpus.model_validate_json(CORPUS.read_bytes()).trophies:
        # Reparse the retained text so historical admission issues stay historical.
        decklist = managym.Decklist(trophy.decklist.source, trophy.decklist.filename)
        try:
            (imported,) = managym.admit_decklists([(trophy.source_id, decklist)])
        except managym.DeckImportError:
            continue
        if any(
            e.section == "main" and e.key and e.key not in old
            for e in imported.decklist.entries
        ):
            decks.append((trophy.source_id, imported.deck))
        if len(decks) == 8:
            break
    if len(decks) != 8:
        raise ValueError("fewer than eight eligible new-card trophy decks")
    digest = file_sha256(CORPUS)
    return [
        ReferencePair(
            matchup=DeckMatchup(
                population_sha256=digest, entries=(a, b), lists=(first, second)
            ),
            deal_seed=21591000 + i,
        )
        for i, ((a, first), (b, second)) in enumerate(
            zip(decks[::2], decks[1::2], strict=True)
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference = StrengthReference.model_validate_json(REFERENCE.read_bytes())
    pairs = new_card_pairs()
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(
        args.out / "pairs.json", [pair.model_dump(mode="json") for pair in pairs]
    )
    score = measure_pairs(reference, pairs, out=args.out, deadline=time.time() + 3600)
    measured = dict(
        score=score.model_dump(),
        games=len(pairs) * len(reference.deal_offsets) * 4,
        revision=source_commit(),
    )
    atomic_json(args.out / "measured.json", measured)
    print(json.dumps(measured))


if __name__ == "__main__":
    main()
