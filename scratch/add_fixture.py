"""Append Scryfall fixture entries for TLA card names (from tla-evidence/cards.json)."""
import json
import sys

FIXTURE = "managym/tests/fixtures/scryfall_cards.json"
cards = json.load(open("scratch/tla-evidence/cards.json"))
cards = cards if isinstance(cards, list) else list(cards.values())
by_name = {c["name"]: c for c in cards}
fixture = json.load(open(FIXTURE))
for name in sys.argv[1:]:
    c = by_name[name]
    fixture[name] = {
        k: c.get(k)
        for k in ("name", "mana_cost", "type_line", "power", "toughness", "keywords", "oracle_text", "colors")
    }
with open(FIXTURE, "w") as f:
    json.dump(fixture, f, indent=2, ensure_ascii=False)
    f.write("\n")
