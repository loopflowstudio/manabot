"""Add pinned scratch/tla-evidence entries to the Scryfall conformance fixture.

Usage: python3 scratch/add_fixture.py "Card Name" ...
"""
import json
import sys

FIXTURE = "managym/tests/fixtures/scryfall_cards.json"
FIELDS = ("name", "mana_cost", "type_line", "power", "toughness", "keywords", "oracle_text", "colors")

evidence = json.load(open("scratch/tla-evidence/cards.json"))
cards = evidence if isinstance(evidence, list) else evidence.get("cards", evidence.get("data"))
by_name = {c["name"]: c for c in cards}
fixture = json.load(open(FIXTURE))
was_sorted = list(fixture) == sorted(fixture)
for name in sys.argv[1:]:
    card = by_name[name]
    fixture[name] = {k: card.get(k) for k in FIELDS if card.get(k) is not None or k in ("power", "toughness")}
    if card.get("power") is None:
        fixture[name].pop("power", None)
        fixture[name].pop("toughness", None)
if was_sorted:
    fixture = dict(sorted(fixture.items()))
open(FIXTURE, "w").write(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
print("sorted" if was_sorted else "appended", len(fixture))
