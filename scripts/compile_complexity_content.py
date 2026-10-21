"""Build MTG-122 research content without editing the product's compiled pack.

The research catalog includes every original definition plus a vanilla RW curve
and Lightning Bolt. All six deck roles share this catalog. The separate learning
schema appends subtype IDs; original schema values and program meanings survive.
This command compiles data only: it never constructs a learner or plays games.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from manabot.semantic.compiler import compile_source, load_json, write_or_check

ROOT = Path(__file__).resolve().parents[1] / "content/semantic/v1"
PACK_KEY = "mtg122-complexity-v1"


def research_source() -> dict[str, Any]:
    """Extend validated compiler JSON; untyped values stay at this data boundary."""
    source = deepcopy(load_json(ROOT / "two_deck.source.json"))
    source["pack_key"] = PACK_KEY
    for key, name, mana, subtypes, power, toughness in (
        ("savannah_lions", "Savannah Lions", "W", ["Cat"], 2, 1),
        ("glory_seeker", "Glory Seeker", "1W", ["Human", "Soldier"], 2, 2),
        ("hill_giant", "Hill Giant", "3R", ["Giant"], 3, 3),
    ):
        source["definitions"].append(
            {
                "key": f"research.{key}",
                "registry_name": name,
                "characteristics": {
                    "mana_cost": mana,
                    "types": ["creature"],
                    "subtypes": subtypes,
                    "power": power,
                    "toughness": toughness,
                },
                "programs": [],
            }
        )
    source["definitions"].append(
        {
            "key": "research.lightning_bolt",
            "registry_name": "Lightning Bolt",
            "characteristics": {"mana_cost": "R", "types": ["instant"]},
            "programs": [
                {
                    "key": "spell",
                    "kind": "spell",
                    "cost": {"mana": "R"},
                    "targets": [
                        {
                            "role": "victim",
                            "selector": {"kind": "creature_or_player"},
                            "min": 1,
                            "max": 1,
                        }
                    ],
                    "ops": [{"op": "deal_damage", "amount": 3, "target": "victim"}],
                }
            ],
        }
    )
    for level in (1, 2):
        for role, counts in (("swift", (8, 8, 7)), ("heavy", (4, 8, 11))):
            counts = (
                tuple(n - d for n, d in zip(counts, (2, 3, 3), strict=True))
                if level == 2
                else counts
            )
            cards = [
                {"definition": "basic.mountain", "count": 9},
                {"definition": "basic.plains", "count": 8},
            ]
            cards.extend(
                {"definition": f"research.{key}", "count": count}
                for key, count in zip(
                    ("savannah_lions", "glory_seeker", "hill_giant"),
                    counts,
                    strict=True,
                )
            )
            if level == 2:
                cards.extend(
                    [
                        {"definition": "research.lightning_bolt", "count": 4},
                        {"definition": "tla.yip_yip", "count": 4},
                    ]
                )
            source["decks"].append(
                {
                    "key": f"l{level}_{role}",
                    "card_count": 40,
                    "cards": cards,
                    "sideboard": [],
                }
            )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = research_source()
    ir, coverage = compile_source(source)
    schema = deepcopy(load_json(ROOT / "learning_schema.json"))
    subtypes = schema["enums"]["subtype"]
    subtypes["Giant"] = max(subtypes.values()) + 1
    for path, value in (
        (ROOT / "complexity.source.json", source),
        (ROOT / "complexity.learning_schema.json", schema),
        (ROOT / "generated/complexity.ir.json", ir),
        (ROOT / "generated/complexity.coverage.json", coverage),
    ):
        write_or_check(path, value, check=args.check)


if __name__ == "__main__":
    main()
