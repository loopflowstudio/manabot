"""Optional payments keep both continuations in policy input."""

from manabot.semantic.learning import TOKEN_KINDS, token_value
from manabot.semantic.policy_catalog import bind_policy_catalog
import managym


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
