"""A generated draft's journal reproduces every native choice and admitted deck."""

import json
from pathlib import Path
from typing import TextIO

import pytest

import manabot.drafting as drafting
from manabot.drafting import BuildChoice, DraftChoice, DraftPreparation, prepare_draft
from manabot.infra.identity import file_sha256
import managym
from managym.format import Limited, Rotisserie


def test_prepared_draft_replays_journal_to_all_eight_legal_builds(
    tmp_path: Path,
) -> None:
    out = tmp_path / "draft"
    result = prepare_draft(out, 370)
    restored = DraftPreparation.model_validate_json((out / "receipt.json").read_bytes())
    assert restored.model_dump(mode="json") == result.model_dump(mode="json")
    assert result.status == "software_validation"
    assert result.draft_picks == 360 and result.construction_picks == 320
    assert sum(result.initial) == 664 and sum(result.remaining) == 304
    assert len(result.decks) == len(result.pools) == 8
    assert result.journal_sha256 == file_sha256(out / "choices.jsonl")
    draft = Rotisserie(
        result.set, result.episode_id, result.seats, result.picks_per_seat
    )
    journal = [
        json.loads(line) for line in (out / "choices.jsonl").read_text().splitlines()
    ]
    assert len(journal) == 680
    for row in journal[:360]:
        pick = DraftChoice.model_validate(row)
        assert draft.observe()["chooser"] == pick.chooser
        draft.pick(pick.episode_id, pick.revision, pick.chooser, pick.card_def_id)
    limited = Limited()
    for seat in range(8):
        pool = draft.pool(seat)
        assert pool.cards == result.pools[seat]
        assert sum(pool.cards.values()) == 45
        build = limited.build(pool, max_cards=result.build_cap)
        for revision, row in enumerate(
            journal[360 + seat * 40 : 360 + (seat + 1) * 40]
        ):
            pick = BuildChoice.model_validate(row)
            assert (pick.chooser, pick.revision) == (seat, revision)
            build.select(pick.card)
        assert build.deck == result.decks[seat].main
        assert build.can_stop and sum(build.deck.values()) == 40
        assert limited.issues(pool, build.deck) == []
        assert result.decks[seat].sideboard == {
            name: count - build.deck.get(name, 0)
            for name, count in pool.cards.items()
            if count > build.deck.get(name, 0)
        }
    assert all(not issues for issues in managym.check_decks(list(result.decks)))
    before = (out / "choices.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_draft(out, 371)
    assert (out / "choices.jsonl").read_bytes() == before
    again = prepare_draft(tmp_path / "repeated", 370)
    assert again.journal_sha256 == result.journal_sha256
    assert [deck.identity for deck in again.decks] == [
        deck.identity for deck in result.decks
    ]


def test_interrupted_preparation_keeps_setup_and_intents_without_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = drafting._journal

    def interrupt(stream: TextIO, choice: DraftChoice | BuildChoice) -> None:
        journal(stream, choice)
        if choice.revision == 3:
            raise OSError("controlled disk interruption")

    monkeypatch.setattr(drafting, "_journal", interrupt)
    out = tmp_path / "interrupted"
    with pytest.raises(OSError, match="controlled disk interruption"):
        prepare_draft(out, 370)
    assert (out / "setup.json").exists()
    assert (out / "catalog.json").exists()
    assert (out / "printings.json").exists()
    assert not (out / "receipt.json").exists()
    assert len((out / "choices.jsonl").read_text().splitlines()) == 4
    with pytest.raises(FileExistsError):
        prepare_draft(out, 370)
