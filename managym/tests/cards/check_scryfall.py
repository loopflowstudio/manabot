"""Check every card definition's Scryfall id against Scryfall (network).

Maintenance only: never run in CI or at game time. For each world pool row,
fetch `https://api.scryfall.com/cards/<scryfall_id>` and confirm the printed
name and that the printing's set is the one the definition's key names
(`<set>.<card>`; tokens, keyed `token.*`, must be token printings). Rebuild
the extension after card changes so the pool is current.

Usage: uv run python managym/tests/cards/check_scryfall.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

import managym

USER_AGENT = "manabot-research/0.1"
WORLD_POOL = "tla-pool-v1"


def fetch(scryfall_id: str) -> dict:
    request = urllib.request.Request(
        f"https://api.scryfall.com/cards/{scryfall_id}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def problems(row: dict, card: dict) -> list[str]:
    out = []
    if card["name"] != row["registry_name"]:
        out.append(f"name is {card['name']!r}")
    key_set = row["semantic_key"].split(".", 1)[0]
    if key_set == "token":
        if card.get("layout") not in ("token", "double_faced_token"):
            out.append(f"printing {card['set']} is not a token")
    elif card["set"] != key_set:
        out.append(f"printing is from {card['set']!r}")
    return out


def main() -> int:
    failures = 0
    rows = managym.content_pool(WORLD_POOL)["definitions"]
    for row in sorted(rows, key=lambda row: row["semantic_key"]):
        card = fetch(row["scryfall_id"])
        for problem in problems(row, card):
            failures += 1
            print(f"{row['semantic_key']}: {problem}")
        time.sleep(0.1)  # Scryfall asks for at most ten requests a second.
    print(f"checked {len(rows)} definitions, {failures} problems")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
