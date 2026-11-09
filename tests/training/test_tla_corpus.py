"""Public trophy selection retains events and rejects whole unsupported lists."""

import csv
import gzip
from pathlib import Path

import pytest

from experiments.tla_corpus import Corpus, Source, build_corpus
from manabot.infra.identity import file_sha256


def _source(tmp_path: Path, rows: list[dict[str, str]]) -> tuple[Path, Source]:
    path = tmp_path / "games.csv.gz"
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path, Source(
        retrieved_at="2026-11-03T00:00:00Z",
        etag=None,
        last_modified=None,
        sha256=file_sha256(path),
        bytes=path.stat().st_size,
    )


def _event(event: str, unsupported: str = "", wins: int = 7) -> list[dict[str, str]]:
    return [
        {
            "expansion": "TLA",
            "event_type": "PremierDraft",
            "draft_id": event,
            "draft_time": "2025-11-18",
            "game_time": f"2025-11-19 00:0{n}:00",
            "build_index": str(int(n > 3)),
            "match_number": str(n),
            "game_number": "1",
            "won": "True",
            "deck_Plains": "20" if n <= 3 else "19",
            "deck_Avatar Enthusiasts": "1" if n <= 3 else "2",
            "sideboard_Island": "1",
            "sideboard_Ally": str(int(unsupported == "sideboard")),
            "deck_Not A Card": str(int(unsupported == "main")),
        }
        for n in range(1, wins + 1)
    ]


def test_trophies_freeze_final_build_and_whole_deck_admission(tmp_path: Path) -> None:
    # Two separate events may have identical decks; neither event disappears.
    rows = _event("a") + _event("b", "sideboard") + _event("c", "main") + _event("d")
    path, source = _source(tmp_path, rows)
    corpus = build_corpus(path, source, target=4)
    assert len(corpus.trophies) == 4
    assert [t.event_id for t in corpus.eligible] == ["a", "d"]
    assert corpus.scanned_rows == 28 and not corpus.exhausted
    first = corpus.eligible[0]
    assert first.source_id == "TLA:PremierDraft:a:1"
    assert first.deck is not None
    assert first.deck.main == {"Plains": 19, "Avatar Enthusiasts": 2}
    assert first.deck.sideboard == {"Island": 1}
    assert sum(game.won for game in first.games) == 7
    assert all(entry.key is not None for entry in first.decklist.entries)
    assert {entry.key for entry in first.decklist.entries} >= {
        "lea.plains",
        "lea.island",
    }
    assert "1 Ally" in corpus.trophies[1].decklist.source
    assert {issue.code for issue in corpus.trophies[1].decklist.issues} == {
        "unsupported"
    }
    assert {issue.code for issue in corpus.trophies[2].decklist.issues} == {"unknown"}
    restored = Corpus.model_validate_json(corpus.model_dump_json())
    assert restored.digest == corpus.digest


def test_target_is_applied_before_admission_without_resampling(tmp_path: Path) -> None:
    path, source = _source(tmp_path, _event("bad", "main") + _event("good"))
    corpus = build_corpus(path, source, target=1)
    assert [t.event_id for t in corpus.trophies] == ["bad"]
    assert corpus.eligible == ()
    assert corpus.scanned_rows == 7


def test_short_source_reports_actual_count_and_incomplete_events(
    tmp_path: Path,
) -> None:
    path, source = _source(tmp_path, _event("a") + _event("partial", wins=6))
    corpus = build_corpus(path, source, target=1000)
    assert len(corpus.trophies) == 1 and corpus.exhausted
    assert corpus.scanned_rows == 13


def test_duplicate_result_cannot_fabricate_a_trophy(tmp_path: Path) -> None:
    rows = _event("a")
    path, source = _source(tmp_path, rows[:3] + [rows[2]] + rows[3:])
    with pytest.raises(ValueError, match="duplicate match"):
        build_corpus(path, source)


def test_source_tampering_fails_before_admission(tmp_path: Path) -> None:
    path, source = _source(tmp_path, _event("a"))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        build_corpus(path, source)


def test_seven_observed_wins_establish_trophy_despite_missing_loss(
    tmp_path: Path,
) -> None:
    rows = _event("missing-loss")
    for row in rows:
        row["match_number"] = str(int(row["match_number"]) + 1)
    path, source = _source(tmp_path, rows)
    corpus = build_corpus(path, source, target=1)
    assert len(corpus.eligible) == 1
    assert [g.match_number for g in corpus.trophies[0].games] == list(range(2, 9))


def test_seven_wins_after_three_losses_is_not_a_trophy(tmp_path: Path) -> None:
    rows = _event("contradictory")
    losses = [dict(rows[0], match_number=str(n), won="False") for n in range(1, 4)]
    for row in rows:
        row["match_number"] = str(int(row["match_number"]) + 3)
    path, source = _source(tmp_path, losses + rows)
    with pytest.raises(ValueError, match="three recorded losses"):
        build_corpus(path, source)
