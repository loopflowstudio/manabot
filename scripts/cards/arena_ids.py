"""Refresh the offline Arena printing table from the world's Scryfall identities.

Run explicitly after a world update: uv run python scripts/cards/arena_ids.py.
No credentials, uploads or game data are involved. CI reads the checked-in table.
"""

import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import managym


def main() -> None:
    ids = sorted(
        {
            row["scryfall_id"]
            for row in managym.content_pool("world")["definitions"]
            if row["scryfall_id"]
        }
    )
    retained: dict[str, dict[str, object]] = {}
    for offset in range(0, len(ids), 75):
        request = Request(
            "https://api.scryfall.com/cards/collection",
            data=json.dumps(
                {
                    "identifiers": [
                        {"id": identity} for identity in ids[offset : offset + 75]
                    ]
                }
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Managym-Card-Maintenance/1.0",
            },
        )
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        if payload.get("not_found"):
            raise ValueError(
                f"Scryfall could not resolve {len(payload['not_found'])} authored printings"
            )
        for card in payload["data"]:
            retained[card["id"]] = {
                key: card.get(key)
                for key in ("arena_id", "name", "set", "collector_number")
            }
        # A small bounded maintenance batch, respecting the API's request interval.
        if offset + 75 < len(ids):
            time.sleep(0.15)
    path = Path(__file__).resolve().parents[2] / "managym" / "arena_ids.json"
    path.write_text(json.dumps(retained, sort_keys=True, indent=2) + "\n")
    print(
        f"{len(retained)} authored printings; {sum(bool(row['arena_id']) for row in retained.values())} Arena ids"
    )


if __name__ == "__main__":
    main()
