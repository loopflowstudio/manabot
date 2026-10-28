"""File imports preserve source while native admission owns playable setups."""

from pathlib import Path

import pytest

from manabot.decklist import (
    ContentPool,
    DeckDocument,
    DeckImportError,
    ImportedMatch,
    admit_document,
    import_files,
    import_match,
    parse_decklist,
)
from manabot.env.match import Match
from manabot.infra.config import SimulationConfig
import managym

PACK = "tla-pool-v1"
HERO = "# Novel development deck\nMain\n10 Forest\n10 Mountain\nSideboard\n1 Island\n"
VILLAIN = "20 Island\n"


def test_decklist_dialect_keeps_original_lines_and_reports_bad_counts() -> None:
    source = "\ufeffDeck\r\n# comment\r\n2 Forest\r\nSideboard:\r\n0 Island\r\n-1 Mountain\r\n2x Forest\r\n99999999999999999999 Island\r\n"
    document = parse_decklist(source, "input.txt")
    assert document.source == source
    assert document.filename == "input.txt"
    assert [(entry.line, entry.name, entry.count) for entry in document.entries] == [
        (3, "Forest", 2)
    ]
    assert [(issue.line, issue.code) for issue in document.issues] == [
        (5, "count"),
        (6, "count"),
        (7, "syntax"),
        (8, "count"),
    ]


def test_decklist_novel_setup_resolves_same_identity_from_files(tmp_path: Path) -> None:
    hero, villain = tmp_path / "hero.txt", tmp_path / "villain.txt"
    hero.write_bytes(HERO.replace("\n", "\r\n").encode())
    villain.write_text(VILLAIN)
    imported = import_files(hero, villain, PACK)
    assert imported.hero.document.source == HERO.replace("\n", "\r\n")
    reformatted = import_match(
        parse_decklist("10 mountain\n5 Forest\n5 forest\nSideboard\n1 Island"),
        parse_decklist(VILLAIN),
        PACK,
    )
    assert imported.setup_id == reformatted.setup_id
    assert imported.hero.main == {"Forest": 10, "Mountain": 10}
    assert imported.hero.sideboard == {"Island": 1}
    assert all(
        entry.card_def_id is not None for entry in imported.hero.document.entries
    )
    assert not any(
        managym.authored_deck_setup("elemental-world-v2", key).decklist
        == imported.hero.main
        for key in managym.authored_deck_keys("elemental-world-v2")
    )
    env = managym.Env(seed=7, skip_trivial=False)
    env.reset(Match(imported.match_config()).to_rust())
    assert env.content_pack_manifest()["content_digest"] == imported.pool.content_digest
    assert (
        env.content_pack_manifest()["compiled_semantics"]
        == imported.pool.compiled_semantics.model_dump()
    )


def test_decklist_reports_unknown_unsupported_and_retains_source() -> None:
    pool = ContentPool.load(PACK)
    unsupported = next(
        row.registry_name
        for row in pool.definitions
        if row.eligibility == "unsupported"
    )
    source = f"10 Forest\n1 Definitely Not A Card\nSideboard\n1 {unsupported}\n"
    with pytest.raises(DeckImportError) as caught:
        import_match(parse_decklist(source), parse_decklist(VILLAIN), PACK)
    document = caught.value.documents["hero"]
    assert document.source == source
    assert {(issue.line, issue.code) for issue in document.issues} >= {
        (2, "unknown"),
        (4, "unsupported"),
    }


def test_decklist_ambiguous_names_fail_without_substitution() -> None:
    pool = ContentPool.load(PACK)
    forest = next(row for row in pool.definitions if row.registry_name == "Forest")
    ambiguous = pool.model_copy(
        update={
            "definitions": (
                *pool.definitions,
                forest.model_copy(update={"registry_name": "FOREST"}),
            )
        }
    )
    result = admit_document(parse_decklist("10 Forest"), ambiguous, "development-v1")
    assert isinstance(result, DeckDocument)
    assert any(issue.code == "ambiguous" and issue.line == 1 for issue in result.issues)
    assert result.source == "10 Forest"


@pytest.mark.parametrize(
    "source", ["Sideboard\n1 Island", "1 Forest\nSideboard\n0 Island", "1001 Forest"]
)
def test_decklist_rejects_construction_errors(source: str) -> None:
    with pytest.raises(DeckImportError):
        import_match(parse_decklist(source), parse_decklist(VILLAIN), PACK)


def test_decklist_rejects_unknown_format_and_retained_source_tampering() -> None:
    with pytest.raises(DeckImportError, match="ruleset"):
        import_match(parse_decklist(HERO), parse_decklist(VILLAIN), PACK, "standard")
    imported = import_match(parse_decklist(HERO), parse_decklist(VILLAIN), PACK)
    retained = imported.model_dump()
    retained["hero"]["document"]["source"] = "10 Island"
    with pytest.raises(ValueError, match="source"):
        ImportedMatch.model_validate(retained).validate_current()
    retained = imported.model_dump()
    retained["pool"]["content_digest"] = "stale"
    with pytest.raises(ValueError, match="identity"):
        ImportedMatch.model_validate(retained).validate_current()


def test_decklist_checkpoint_simulation_is_explicitly_rejected() -> None:
    imported = import_match(parse_decklist(HERO), parse_decklist(VILLAIN), PACK)
    with pytest.raises(ValueError, match="checkpoint setup admission"):
        SimulationConfig(hero="policy.pt", match=imported.match_config())
