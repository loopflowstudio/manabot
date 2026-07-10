#!/usr/bin/env python3
"""Refresh the Scryfall conformance fixture for all registered real cards.

Usage: uv run scripts/refresh_card_fixture.py

Parses managym/src/cardsets/*.rs for every registered card name, resolves
each real card against Scryfall's Oracle-Cards bulk data (one download, no
per-card API calls — https://scryfall.com/docs/api/bulk-data), and rewrites
managym/tests/fixtures/scryfall_cards.json. Cards registered as tokens
(`is_token: true`) or explicitly marked `// not a real card` are excluded
from the fixture; the cargo conformance test
(managym/tests/conformance_tests.rs) enforces that every other registration
has a fixture entry.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CARDSETS = REPO / "managym" / "src" / "cardsets"
FIXTURE = REPO / "managym" / "tests" / "fixtures" / "scryfall_cards.json"

USER_AGENT = "manabot-research/0.1 (jackstah@gmail.com)"
BULK_INDEX = "https://api.scryfall.com/bulk-data"

# Fields we snapshot per card. They are compared against registrations by the
# cargo conformance test (managym/tests/conformance_tests.rs).
FIELDS = ["name", "mana_cost", "type_line", "power", "toughness", "keywords", "oracle_text", "colors"]


def registered_names() -> list[str]:
    """Every card name registered in the cardsets, minus tokens and cards
    explicitly marked `// not a real card`."""
    names: list[str] = []
    skipped: list[str] = []
    for path in sorted(CARDSETS.glob("*.rs")):
        text = path.read_text()
        # basic_land("Plains", ...)
        names.extend(re.findall(r'basic_land\("([^"]+)"', text))
        # register_creature("Name", ...)
        names.extend(re.findall(r'register_creature\(\s*"([^"]+)"', text))
        # register_card(CardDefinition { name: "Name".to_string(), ... })
        # Split into per-registration chunks so we can see is_token / markers.
        for chunk in re.split(r"self\.register_card\(CardDefinition \{", text)[1:]:
            m = re.search(r'^\s*name: "([^"]+)"', chunk, re.M)
            if not m:
                continue
            name = m.group(1)
            body = chunk.split("\n        });", 1)[0]
            if "is_token: true" in body:
                skipped.append(f"{name} (token)")
                continue
            names.append(name)
        # `// not a real card` marker: the comment goes on the line(s) above
        # the registration; drop any collected name whose registration block
        # carries the marker.
        for marked in re.findall(
            r"// not a real card[^\n]*\n(?:\s*//[^\n]*\n)*?\s*self\.register_card\(CardDefinition \{\s*\n\s*name: \"([^\"]+)\"",
            text,
        ):
            if marked in names:
                names.remove(marked)
                skipped.append(f"{marked} (not a real card)")
    if skipped:
        print(f"skipping (no fixture entry): {', '.join(sorted(skipped))}")
    # de-dup, keep stable order
    seen: set[str] = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def http_get(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            data = gzip.GzipFile(fileobj=io.BytesIO(data)).read()
        return data


def oracle_cards() -> dict[str, dict]:
    """Download the Oracle-Cards bulk file and index by exact card name."""
    index = json.loads(http_get(BULK_INDEX))
    entry = next(item for item in index["data"] if item["type"] == "oracle_cards")
    size_mb = entry.get("size", 0) / 1e6
    print(f"downloading oracle_cards bulk data ({size_mb:.0f} MB, updated {entry.get('updated_at')})...")
    cards = json.loads(http_get(entry["download_uri"]))
    print(f"bulk data: {len(cards)} oracle cards")
    by_name: dict[str, dict] = {}
    for card in cards:
        if card.get("layout") in ("token", "double_faced_token", "emblem", "art_series"):
            continue
        by_name.setdefault(card["name"], card)
    return by_name


def main() -> int:
    names = registered_names()
    print(f"{len(names)} registered real-card candidates")
    by_name = oracle_cards()

    cards: dict[str, dict] = {}
    missing: list[str] = []
    for name in names:
        card = by_name.get(name)
        if card is None:
            missing.append(name)
            print(f"  {name}: NOT FOUND on Scryfall")
            continue
        cards[name] = {field: card.get(field) for field in FIELDS}

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(dict(sorted(cards.items())), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {FIXTURE} ({len(cards)} cards)")
    if missing:
        print("\nNOT on Scryfall (mark `// not a real card` or register as token):")
        for name in missing:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
