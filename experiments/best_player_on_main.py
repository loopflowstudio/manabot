"""Play frontier Small u1952 against the first-64 heuristic on historical or new-card decks."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.tla_corpus import Corpus
from experiments.tla_frontier import PROTOCOL, corpus, deck_matchup
from manabot.eval.evaluation import evaluate_local
from manabot.eval.local import prepare_local_request
from manabot.eval.results import bootstrap_mean_interval
from manabot.infra.artifacts import StoredArtifact
from manabot.infra.files import atomic_json
from manabot.infra.identity import StrictModel, file_sha256
from manabot.infra.source import source_commit
from manabot.model.writer_catalog import WriterCatalog
from manabot.sim.matchup import DeckMatchup
import managym

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = StoredArtifact(
    uri="r2://manabot/jobs/59718fe34945/checkpoints/learner_transitions-82370334.pt",
    sha256="c95c61aa298528e5c2782430d5e3f90929d3e9b7ac8b1fc45f560ed6ab4bbaf8",
    bytes=3945456,
)
CORPUS = ROOT / "experiments/data/mtg213/corpus-v1/corpus.json"
WRITER = (
    ROOT
    / "manabot/model/writer_catalogs/0798b2c7c7f54b7329043b13cb1742d74fe21b35373820580b76f979e6b7eb5f.json"
)


class DeckPair(StrictModel):
    matchup: DeckMatchup
    deal_seed: int


def historical_pairs() -> list[DeckPair]:
    population = corpus()
    return [
        DeckPair(matchup=deck_matchup(population, PROTOCOL, pair), deal_seed=pair.deal)
        for pair in PROTOCOL.pairs
    ]


def new_card_pairs() -> list[DeckPair]:
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
        DeckPair(
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
    parser.add_argument("--arm", choices=("historical", "new-cards"), required=True)
    args = parser.parse_args()
    pairs = historical_pairs() if args.arm == "historical" else new_card_pairs()
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(
        args.out / "pairs.json", [pair.model_dump(mode="json") for pair in pairs]
    )
    deadline = time.time() + 7200
    scores: list[float] = []
    games = 0
    for index, pair in enumerate(pairs):
        request = prepare_local_request(
            CHECKPOINT,
            "heuristic",
            (pair.deal_seed, pair.deal_seed + 16),
            pair.matchup,
            deadline=deadline,
            heuristic_actions=64,
        )
        directory = args.out / f"pair-{index:02d}"
        result = evaluate_local(request, deadline=deadline, out_dir=directory)
        if result.status != "completed" or result.score is None:
            raise ValueError(f"incomplete deck pair at {directory}: {result.error}")
        scores.append(result.score.mean)
        games += len(result.rows)
    # Keep both deals and their crossed legs together when resampling a deck pair.
    indices = np.random.default_rng(21592001).integers(
        len(scores), size=(2000, len(scores))
    )
    score = bootstrap_mean_interval(scores, indices)
    measured = dict(
        score=score.model_dump(),
        games=games,
        revision=source_commit(),
    )
    atomic_json(args.out / "measured.json", measured)
    print(json.dumps(measured))


if __name__ == "__main__":
    main()
