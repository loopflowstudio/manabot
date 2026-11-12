"""Audit the pinned TLE allocation and installed catalog, never deck admission.

Run with ``uv run python scripts/check_tle_coverage.py`` after rebuilding the
extension. ``--allocation-only`` checks planning coverage, not card support.
``--bonus-sheet`` restricts the catalog check to the 61 TLE booster printings
(collector numbers 1-61) that TLA play boosters carry. No mode certifies
printed behavior or the 1,000-event corpus outcome.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from manabot import managym

_FIXTURES = Path(__file__).resolve().parents[1] / "managym/cards/tests/fixtures"
_INVENTORY_SHA256 = "334e003532fcc4d3426086da033a02356715faeb7c892279978ba40a394a327d"


class InventoryCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_name: str
    oracle_id: UUID
    first_printing_id: UUID
    first_printing_set: str
    tle_printing_id: UUID
    released_at: str
    clauses: list[str]


class ClauseAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    implementation: str = Field(min_length=1)


class CardAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_name: str
    oracle_id: UUID
    definition_file: str
    clauses: list[ClauseAllocation]


def check_allocation(fixtures: Path = _FIXTURES) -> list[InventoryCard]:
    """Require one ordered allocation for every pinned Oracle paragraph."""
    inventory_bytes = (fixtures / "tle-inventory.jsonl").read_bytes()
    if hashlib.sha256(inventory_bytes).hexdigest() != _INVENTORY_SHA256:
        raise ValueError("TLE source inventory changed; review the source evidence")
    inventory = [
        InventoryCard.model_validate_json(line) for line in inventory_bytes.splitlines()
    ]
    allocations = [
        CardAllocation.model_validate_json(line)
        for line in (fixtures / "tle-allocation.jsonl").read_bytes().splitlines()
    ]
    if (
        len(inventory) != 242
        or len({c.card_name for c in inventory}) != 242
        or len({c.oracle_id for c in inventory}) != 242
        or sum(len(c.clauses) for c in inventory) != 411
        or len(allocations) != len(inventory)
    ):
        raise ValueError("TLE allocation must cover 242 names and 411 paragraphs")
    for source, allocation in zip(inventory, allocations, strict=True):
        if (
            allocation.card_name != source.card_name
            or allocation.oracle_id != source.oracle_id
            or allocation.definition_file
            != f"managym/cards/src/{source.first_printing_set}.rs"
            or [c.index for c in allocation.clauses] != list(range(len(source.clauses)))
        ):
            raise ValueError(
                f"Incomplete or misbound TLE allocation: {source.card_name}"
            )
    return inventory


class BonusSheetCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collector_number: str
    card_name: str
    tle_printing_id: UUID


def bonus_sheet(
    inventory: list[InventoryCard], fixtures: Path = _FIXTURES
) -> list[InventoryCard]:
    """The inventory rows printed on the bonus sheet, in collector order."""
    rows = [
        BonusSheetCard.model_validate_json(line)
        for line in (fixtures / "tle-bonus-sheet.jsonl").read_bytes().splitlines()
    ]
    by_name = {card.card_name: card for card in inventory}
    if [row.collector_number for row in rows] != [str(n) for n in range(1, 62)] or any(
        row.card_name not in by_name
        or by_name[row.card_name].tle_printing_id != row.tle_printing_id
        for row in rows
    ):
        raise ValueError(
            "TLE bonus sheet must be collector numbers 1-61 of the inventory"
        )
    return [by_name[row.card_name] for row in rows]


def catalog_issues(inventory: list[InventoryCard]) -> list[str]:
    """Check actual definitions, first-printing identities and token row presence."""
    pool = managym.content_pool("world")["definitions"]
    by_name = {row["registry_name"]: row for row in pool}
    key_counts = Counter(row["key"] for row in pool)
    catalog = managym.learning_catalog()
    learning_rows = {name: i for i, name in enumerate(catalog["registry_names"])}
    offsets = catalog["definition_offsets"]
    issues: list[str] = []
    for card in inventory:
        row = by_name.get(card.card_name)
        if row is None:
            issues.append(f"{card.card_name}: missing definition")
            continue
        if row["eligibility"] != "eligible":
            issues.append(f"{card.card_name}: not an eligible non-token definition")
        if row["scryfall_id"] != str(card.first_printing_id) or not (
            row["key"] or ""
        ).startswith(f"{card.first_printing_set}."):
            issues.append(f"{card.card_name}: first-printing identity mismatch")
        learning_row = learning_rows.get(card.card_name)
        if learning_row is None or offsets[learning_row] == offsets[learning_row + 1]:
            issues.append(f"{card.card_name}: missing learning tokens")
        if key_counts[row["key"]] != 1:
            issues.append(f"{card.card_name}: duplicate definition key")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-only", action="store_true")
    parser.add_argument("--bonus-sheet", action="store_true")
    args = parser.parse_args()
    inventory = check_allocation()
    checked = bonus_sheet(inventory) if args.bonus_sheet else inventory
    issues = [] if args.allocation_only else catalog_issues(checked)
    print(
        json.dumps(
            {
                "check": "allocation" if args.allocation_only else "catalog_coverage",
                "cards": len(checked),
                "paragraphs": sum(len(c.clauses) for c in checked),
                "issues": issues,
            },
            indent=2,
        )
    )
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
