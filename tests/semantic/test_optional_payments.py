"""Optional payments keep both continuations in policy input.

The compiler checks below exercise the retired offline compiler on frozen w11
source; current cards are engine definitions.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from manabot.semantic.compiler import SemanticCompileError, compile_source
from manabot.semantic.learning import TOKEN_KINDS, token_value
from manabot.semantic.policy_catalog import bind_policy_catalog
import managym

SOURCE = Path("content/semantic/w11/elemental_world.source.json")


def _lesson(source: dict[str, Any]) -> dict[str, Any]:
    # JSON is deliberately kept at the compiler boundary in these authoring tests.
    definition = next(
        row
        for row in source["definitions"]
        if row["registry_name"] == "Waterbending Lesson"
    )
    return definition["programs"][0]["ops"][1]


def test_both_payment_branches_reach_policy_input() -> None:
    """The current card's paid and declined continuations are both tokens."""
    pack = bind_policy_catalog("elemental-world-v2")
    pool = managym.content_pool("elemental-world-v2")
    card_def_id = next(
        row["card_def_id"]
        for row in pool["definitions"]
        if row["registry_name"] == "Waterbending Lesson"
    )
    (program,) = pack.program_rows(pack.definition_row(card_def_id))
    start, end = pack.catalog.program_offsets[program : program + 2]
    tokens = list(
        zip(
            pack.catalog.token_kind[start:end].tolist(),
            pack.catalog.token_value[start:end].tolist(),
            strict=True,
        )
    )
    assert (TOKEN_KINDS["variant"], token_value("Effect::OptionalPayment")) in tokens
    for branch in ("if_paid", "if_declined"):
        at = tokens.index((TOKEN_KINDS["field"], token_value(branch)))
        assert tokens[at + 1] == (TOKEN_KINDS["structure"], token_value("["))
    # Declining the waterbend discards; paying it continues with nothing more.
    declined = tokens.index((TOKEN_KINDS["field"], token_value("if_declined")))
    assert tokens[declined + 2] == (
        TOKEN_KINDS["variant"],
        token_value("Effect::Discard"),
    )


@pytest.mark.parametrize("branch", ["if_paid", "if_declined"])
def test_payment_branches_reject_unbound_targets(branch: str) -> None:
    source = json.loads(SOURCE.read_text())
    _lesson(source)[branch] = [{"op": "deal_damage", "target": "unknown", "amount": 1}]
    with pytest.raises(SemanticCompileError, match="unknown target role"):
        compile_source(source)


@pytest.mark.parametrize(
    "cost",
    [
        {},
        {"waterbend": -1},
        {"waterbend": 256},
        {"waterbend": True},
        {"mana": "{2}", "waterbend": 2},
    ],
)
def test_payment_cost_rejects_ambiguous_or_invalid_obligations(
    cost: dict[str, object],
) -> None:
    source = json.loads(SOURCE.read_text())
    _lesson(source)["cost"] = cost
    with pytest.raises(SemanticCompileError):
        compile_source(source)
