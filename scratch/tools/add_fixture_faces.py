"""Add Scryfall card JSON files (one per card, from /cards/named) to the
conformance fixture, one entry per face for transforming double-faced cards.

Usage: python3 -I scratch/tools/add_fixture_faces.py card.json ...
A face's keywords are the card's keywords its own oracle text names
(case-sensitive), since Scryfall lists keywords per card.
"""
import json
import re
import sys

FIXTURE = "managym/tests/fixtures/scryfall_cards.json"
fixture = json.load(open(FIXTURE))
was_sorted = list(fixture) == sorted(fixture)


def entry(face, keywords):
    out = {
        "name": face["name"],
        "mana_cost": face.get("mana_cost") or None,
        "type_line": face["type_line"],
        "power": face.get("power"),
        "toughness": face.get("toughness"),
        "keywords": keywords,
        "oracle_text": face.get("oracle_text"),
        "colors": face.get("colors", []),
    }
    if out["power"] is None:
        out.pop("power")
        out.pop("toughness")
    if out["mana_cost"] is None:
        out.pop("mana_cost")
    return out


for path in sys.argv[1:]:
    card = json.load(open(path))
    faces = card.get("card_faces")
    if faces and card.get("layout") == "transform":
        for face in faces:
            text = face.get("oracle_text", "")
            keywords = [k for k in card["keywords"] if re.search(rf"\b{re.escape(k)}\b", text)]
            fixture[face["name"]] = entry(face, keywords)
    else:
        fixture[card["name"]] = entry(card, card["keywords"])
if was_sorted:
    fixture = dict(sorted(fixture.items()))
open(FIXTURE, "w").write(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
print("sorted" if was_sorted else "appended", len(fixture))
