"""Freeze 17Lands trophy events and admit whole lists for the TLA pilot.

The public CC BY 4.0 game dump is the source, not the restricted trophy API.
`prefetch` retains its exact compressed bytes; `build_corpus` selects the first
N events with seven observed wins in source order and retains the seventh-win
main/sideboard through managym's existing counted-list import authority.
No downloading or re-admission occurs in a game/reset/learner hot path.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

from pydantic import Field

from manabot.infra.files import atomic_json
from manabot.infra.identity import StrictModel, canonical_sha256, file_sha256
import managym
from managym import Deck, DeckImportError, Decklist

SOURCE_URL = (
    "https://17lands-public.s3.amazonaws.com/analysis_data/game_data/"
    "game_data_public.TLA.PremierDraft.csv.gz"
)


class Source(StrictModel):
    url: str = SOURCE_URL
    attribution: str = "17Lands (https://www.17lands.com/public_datasets)"
    license: str = "https://creativecommons.org/licenses/by/4.0/"
    retrieved_at: str
    etag: str | None
    last_modified: str | None
    sha256: str
    bytes: int = Field(ge=1)


class GameEvidence(StrictModel):
    """The recorded games establishing an event's seven wins, without player IDs."""

    row: int = Field(ge=1)
    match_number: int = Field(ge=1)
    game_time: str
    build_index: int = Field(ge=0)
    won: bool


class Trophy(StrictModel):
    event_id: str
    draft_time: str
    games: tuple[GameEvidence, ...]
    # This is the exact counted-list translation, with source card names intact.
    # Native import resolves canonical names/keys; it alone decides eligibility.
    decklist: Decklist
    deck: Deck | None

    @property
    def eligible(self) -> bool:
        return self.deck is not None

    @property
    def source_id(self) -> str:
        return f"TLA:PremierDraft:{self.event_id}:{self.games[-1].build_index}"


class Corpus(StrictModel):
    schema_version: Literal[1] = 1
    selection: Literal["first-observed-trophies-v1"] = "first-observed-trophies-v1"
    target: int = Field(default=1000, ge=1)
    source: Source
    world: dict[str, object]
    scanned_rows: int = Field(ge=0)
    exhausted: bool
    trophies: tuple[Trophy, ...]

    @property
    def eligible(self) -> tuple[Trophy, ...]:
        return tuple(trophy for trophy in self.trophies if trophy.eligible)

    @property
    def digest(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


def prefetch(directory: Path) -> Source:
    """Fetch once with bounded I/O; existing bytes must match the saved receipt."""
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "source.json"
    destination = directory / "games.csv.gz"
    if receipt.exists():
        source = Source.model_validate_json(receipt.read_text())
        if file_sha256(destination) != source.sha256:
            raise ValueError("retained 17Lands source differs from its receipt")
        return source
    if destination.exists():
        raise ValueError("source bytes exist without provenance; use a new directory")
    partial = directory / "games.csv.gz.partial"
    with urlopen(SOURCE_URL, timeout=30) as response, partial.open("wb") as output:
        expected = int(response.headers["Content-Length"])
        if expected > 256 * 1024 * 1024:
            raise ValueError("source exceeds the pilot's 256 MiB fetch budget")
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > expected:
                raise ValueError("source exceeded its declared length")
            output.write(chunk)
        if total != expected:
            raise ValueError("incomplete public game dump")
        source = Source(
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
            sha256=file_sha256(partial),
            bytes=total,
        )
    partial.replace(destination)
    atomic_json(receipt, source.model_dump(mode="json"))
    return source


def _decklist(row: dict[str, str], event_id: str) -> Decklist:
    lines: list[str] = []
    for prefix, section in (("deck_", "Main"), ("sideboard_", "Sideboard")):
        lines.append(section)
        for field, value in row.items():
            if field.startswith(prefix):
                count = int(value)
                if count < 0:
                    raise ValueError(f"negative source count: {event_id} {field}")
                if count:
                    lines.append(f"{count} {field.removeprefix(prefix)}")
    return Decklist("\n".join(lines) + "\n", f"{event_id}.txt")


def build_corpus(path: Path, source: Source, target: int = 1000) -> Corpus:
    """Freeze whole-deck admission, including sideboards; never repair a list.

    One event contributes once, even if it used several builds. Identical lists
    from different events remain separate population members. The first target
    trophies are selected before admission, so unsupported cards cannot change
    the source population or cause additional fetching/resampling.
    """
    if target < 1:
        raise ValueError("target must be positive")
    if file_sha256(path) != source.sha256:
        raise ValueError("source digest mismatch")
    events: dict[str, dict[int, GameEvidence]] = {}
    selected: set[str] = set()
    trophies: list[Trophy] = []
    scanned = 0
    exhausted = True
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        required = {
            "expansion",
            "event_type",
            "draft_id",
            "draft_time",
            "game_time",
            "build_index",
            "match_number",
            "game_number",
            "won",
        }
        if (
            not required <= fields
            or not any(f.startswith("deck_") for f in fields)
            or not any(f.startswith("sideboard_") for f in fields)
        ):
            raise ValueError("public game dump is missing event or deck columns")
        for scanned, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"malformed CSV row {scanned}")
            if row["expansion"] != "TLA" or row["event_type"] != "PremierDraft":
                raise ValueError(f"unexpected population in row {scanned}")
            if row["game_number"] != "1" or row["won"] not in ("True", "False"):
                raise ValueError(f"unexpected game result in row {scanned}")
            event_id = row["draft_id"]
            if event_id in selected:
                continue
            game = GameEvidence(
                row=scanned,
                match_number=int(row["match_number"]),
                game_time=row["game_time"],
                build_index=int(row["build_index"]),
                won=row["won"] == "True",
            )
            games = events.setdefault(event_id, {})
            if game.match_number in games:
                raise ValueError(f"duplicate match in source event {event_id}")
            games[game.match_number] = game
            if sum(g.won for g in games.values()) != 7:
                continue
            # The selected build is from the actual seventh win, not a partial
            # or out-of-order event's last encountered row. Missing losses are
            # allowed: seven observed distinct wins already establish a trophy.
            if sum(not g.won for g in games.values()) > 2:
                raise ValueError(f"seven wins after three recorded losses: {event_id}")
            if game.match_number != max(games) or not game.won:
                raise ValueError(f"unordered trophy event {event_id}")
            imported_source = _decklist(row, event_id)
            try:
                (admitted,) = managym.admit_decklists([(event_id, imported_source)])
                decklist, deck = admitted.decklist, admitted.deck
            except DeckImportError as error:
                decklist, deck = error.decklists[event_id], None
            trophies.append(
                Trophy(
                    event_id=event_id,
                    draft_time=row["draft_time"],
                    games=tuple(games[n] for n in sorted(games)),
                    decklist=decklist,
                    deck=deck,
                )
            )
            selected.add(event_id)
            del events[event_id]
            if len(trophies) == target:
                exhausted = False
                break
    return Corpus(
        target=target,
        source=source,
        world=managym.world_identity(),
        scanned_rows=scanned,
        exhausted=exhausted,
        trophies=tuple(trophies),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--target", type=int, default=1000)
    args = parser.parse_args()
    snapshot = args.out / "corpus.json"
    if snapshot.exists():
        raise ValueError("corpus is frozen; choose a new version directory")
    source = prefetch(args.out)
    corpus = build_corpus(args.out / "games.csv.gz", source, args.target)
    atomic_json(snapshot, corpus.model_dump(mode="json"))
    print(
        f"{len(corpus.trophies)} trophies; {len(corpus.eligible)} eligible; sha256={corpus.digest}"
    )
    if not corpus.eligible:
        raise SystemExit("No eligible whole decks; training/evaluation cannot start.")


if __name__ == "__main__":
    main()
