"""The planning ledger cannot silently lose or rebind printed TLE clauses."""

import json
from pathlib import Path
import shutil

import pytest

from scripts.check_tle_coverage import bonus_sheet, check_allocation


def test_tle_allocation_covers_pinned_source() -> None:
    cards = check_allocation()
    assert len(cards) == 242
    assert sum(len(card.clauses) for card in cards) == 411


def test_tle_bonus_sheet_is_collector_numbers_one_to_sixty_one() -> None:
    cards = bonus_sheet(check_allocation())
    assert len({card.card_name for card in cards}) == 61
    assert cards[0].card_name == "Brought Back"
    assert cards[-1].card_name == "Valakut, the Molten Pinnacle"


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "owner", "source"])
def test_tle_allocation_rejects_lost_or_rebound_evidence(
    tmp_path: Path, mutation: str
) -> None:
    fixtures = Path(__file__).resolve().parents[2] / "managym/cards/tests/fixtures"
    for filename in ("tle-inventory.jsonl", "tle-allocation.jsonl"):
        shutil.copyfile(fixtures / filename, tmp_path / filename)
    path = tmp_path / "tle-allocation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "omit":
        rows[0]["clauses"].pop()
    elif mutation == "duplicate":
        rows[0]["clauses"][1]["index"] = 0
    elif mutation == "owner":
        rows[0]["definition_file"] = "managym/cards/src/tla.rs"
    else:
        source = tmp_path / "tle-inventory.jsonl"
        source.write_text(source.read_text().replace("airbend another", "airbend each"))
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="TLE"):
        check_allocation(tmp_path)
