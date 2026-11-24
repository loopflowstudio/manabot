#!/usr/bin/env python3
"""Refresh the Scryfall conformance fixture for all registered real cards.

Usage: uv run scripts/refresh_card_fixture.py

Reads every card name from the native world pool (`managym.content_pool`),
resolves each real card against Scryfall's Oracle-Cards bulk data (one
download, no per-card API calls — https://scryfall.com/docs/api/bulk-data), and
rewrites managym/tests/fixtures/scryfall_cards.json. Token definitions are
excluded; the cargo conformance test (managym/tests/conformance_tests.rs)
enforces that every other definition has a fixture entry. Rebuild the
extension after card changes so the pool is current.
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
import re
import sys
import urllib.request

from managym import content_pool

REPO = Path(__file__).resolve().parent.parent
WORLD_POOL = "world"
FIXTURE = REPO / "managym" / "tests" / "fixtures" / "scryfall_cards.json"

USER_AGENT = "manabot-research/0.1 (jackstah@gmail.com)"
BULK_INDEX = "https://api.scryfall.com/bulk-data"

# Fields we snapshot per card. They are compared against registrations by the
# cargo conformance test (managym/tests/conformance_tests.rs).
FIELDS = [
    "name",
    "mana_cost",
    "type_line",
    "power",
    "toughness",
    "keywords",
    "oracle_text",
    "colors",
]


def registered_names(oracle: dict[str, dict]) -> list[str]:
    """Registered printings and their registered back faces, by printed name."""
    rows = content_pool(WORLD_POOL)["definitions"]
    names = {
        row["registry_name"]
        for row in rows
        if row["key"] and not row["key"].startswith("token.")
    }
    registered = {row["registry_name"] for row in rows}
    # Back faces have no key of their own. Join them through the printing
    # so keyless rules fixtures are never mistaken for real cards.
    backs = {
        face["name"]
        for name in names
        if (card := oracle.get(name))
        and card.get("layout") in ("modal_dfc", "transform")
        for face in card["card_faces"][1:]
        if face["name"] in registered
    }
    return sorted(names | backs)


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
    size_mb = entry.get("size", entry.get("compressed_size", 0)) / 1e6
    print(
        f"downloading oracle_cards bulk data ({size_mb:.0f} MB, updated {entry.get('updated_at')})..."
    )
    if "download_uri" in entry:
        cards = json.loads(http_get(entry["download_uri"]))
    else:
        # Scryfall now publishes bulk files as gzipped JSON Lines only.
        data = http_get(entry["jsonl_download_uri"])
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        cards = [json.loads(line) for line in data.splitlines() if line.strip()]
    print(f"bulk data: {len(cards)} oracle cards")
    by_name: dict[str, dict] = {}
    for card in cards:
        if card.get("layout") in (
            "token",
            "double_faced_token",
            "emblem",
            "art_series",
            "front_card",
        ):
            continue
        if card.get("layout") in ("adventure", "prepare"):
            # These layouts register one physical card under its front name.
            # A face's null field (prepare faces print no colors) leaves
            # the card's value in place.
            front = {k: v for k, v in card["card_faces"][0].items() if v is not None}
            by_name.setdefault(front["name"], card | front)
        if card.get("layout") in ("modal_dfc", "transform"):
            for face in card["card_faces"]:
                # Scryfall lists keywords on the whole card. An ability such as
                # Photon's hexproof belongs only to the face that prints it.
                text = face.get("oracle_text", "")
                keywords = [
                    keyword
                    for keyword in card.get("keywords", [])
                    if re.search(
                        rf"(?:^|, ){re.escape(keyword)}(?=\b|$)",
                        text,
                        re.IGNORECASE | re.MULTILINE,
                    )
                ]
                by_name.setdefault(face["name"], card | face | {"keywords": keywords})
        by_name.setdefault(card["name"], card)
    return by_name


def main() -> int:
    by_name = oracle_cards()
    names = registered_names(by_name)
    print(f"{len(names)} registered real-card candidates")

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
    FIXTURE.write_text(
        json.dumps(dict(sorted(cards.items())), indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {FIXTURE} ({len(cards)} cards)")
    if missing:
        print(
            "\nNOT on Scryfall (define tokens under `token.*`; keep invented fixtures out of the pool):"
        )
        for name in missing:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
