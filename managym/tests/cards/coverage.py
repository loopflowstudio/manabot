"""Report which defined cards have directed tests.

Usage: uv run python managym/tests/cards/coverage.py [--set tla] [--untested]

Lists every card in the world pool (`managym.content_pool`) by set, with the
tests in `managym/tests/cards/<set>.rs` that name it and how many other Rust
tests mention it. A test names a card when the card's name appears as a string
literal in the test function. This is a reading aid: coverage never decides
whether a card is supported (only its set file does).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import re

from managym import content_pool

MANAGYM = Path(__file__).resolve().parents[2]
TESTS = MANAGYM / "tests"
WORLD_POOL = "world"


@dataclass
class CardCoverage:
    name: str
    set_code: str
    supported: bool
    own_tests: list[str] = field(default_factory=list)
    other_tests: int = 0


def _test_functions(path: Path) -> list[tuple[str, set[str]]]:
    """(test name, string literals) for each `#[test]` function in `path`."""
    tests: list[tuple[str, set[str]]] = []
    for chunk in path.read_text().split("#[test]")[1:]:
        match = re.search(r"\bfn (\w+)", chunk)
        if match is None:
            continue
        body = re.split(r"\n\s*(?:#\[test\]|fn \w+)", chunk[match.end() :], 1)[0]
        tests.append((match.group(1), set(re.findall(r'"([^"]+)"', body))))
    return tests


def coverage() -> list[CardCoverage]:
    cards: dict[str, CardCoverage] = {}
    for row in content_pool(WORLD_POOL)["definitions"]:
        if row["key"] is None:  # a rules fixture, not a printed card
            continue
        set_code = row["key"].split(".")[0]
        if set_code == "token":
            continue
        cards[row["registry_name"]] = CardCoverage(
            name=row["registry_name"],
            set_code=set_code,
            supported=row["eligibility"] == "eligible",
        )
    for path in sorted(TESTS.rglob("*.rs")):
        own_set = path.stem if path.parent.name == "cards" else None
        for test, literals in _test_functions(path):
            for name in literals & cards.keys():
                card = cards[name]
                if card.set_code == own_set:
                    card.own_tests.append(test)
                else:
                    card.other_tests += 1
    return sorted(cards.values(), key=lambda c: (c.set_code, c.name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", dest="set_code", help="only this set code")
    parser.add_argument(
        "--untested", action="store_true", help="only cards without own-set tests"
    )
    args = parser.parse_args()
    rows = [
        card
        for card in coverage()
        if (args.set_code is None or card.set_code == args.set_code)
        and not (args.untested and card.own_tests)
    ]
    current = None
    for card in rows:
        if card.set_code != current:
            current = card.set_code
            print(f"\n## {current}\n")
            print("| Card | Supported | Tests in set file | Other mentions |")
            print("|---|---|---|---|")
        tests = ", ".join(f"`{t}`" for t in sorted(card.own_tests)) or "—"
        supported = "yes" if card.supported else "no"
        print(f"| {card.name} | {supported} | {tests} | {card.other_tests} |")
    tested = sum(1 for card in rows if card.own_tests)
    print(f"\n{tested}/{len(rows)} cards have a test in their set file.")


if __name__ == "__main__":
    main()
