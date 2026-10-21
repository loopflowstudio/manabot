"""Research catalog admission preserves product definitions and vanilla decks."""

from copy import deepcopy

from manabot.semantic.compiler import compile_source, load_json
from scripts.compile_complexity_content import ROOT, research_source


def test_research_source_preserves_product_and_full_decks() -> None:
    product = load_json(ROOT / "two_deck.source.json")
    source = research_source()
    assert (
        source["definitions"][: len(product["definitions"])] == product["definitions"]
    )
    assert source["decks"][:2] == product["decks"]
    assert len(source["decks"]) == 6
    definitions = {row["key"]: row for row in source["definitions"]}
    for deck in source["decks"][2:]:
        assert sum(row["count"] for row in deck["cards"]) == 40
        assert not deck["sideboard"]
        for row in deck["cards"]:
            card = definitions[row["definition"]]
            if "creature" in card["characteristics"]["types"]:
                assert not card["programs"]
                assert not card["characteristics"].get("keywords")
    ir, _ = compile_source(source)
    assert ir == load_json(ROOT / "generated/complexity.ir.json")


def test_research_schema_only_appends_giant() -> None:
    original = load_json(ROOT / "learning_schema.json")
    expected = deepcopy(original)
    expected["enums"]["subtype"]["Giant"] = (
        max(original["enums"]["subtype"].values()) + 1
    )
    assert expected == load_json(ROOT / "complexity.learning_schema.json")
