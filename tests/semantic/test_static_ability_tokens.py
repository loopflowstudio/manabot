"""Static abilities and additional costs reach the policy as learning tokens.

Each card's `statics` and `additional_costs` serialize into its characteristics
span of `managym.learning_catalog`; the native guard in
`managym/tests/cards/observability.rs` covers every variant exhaustively. This
checks the same tokens through the Python boundary, one real card per variant.
"""

from __future__ import annotations

import pytest

from manabot.semantic.learning import TOKEN_KINDS, token_value
import managym

VARIANTS = [
    ("Iroh, Grand Lotus", "StaticAbility::GraveyardSpellsHaveFlashback"),
    ("Great Divide Guide", "StaticAbility::GrantsAnyColorManaAbility"),
    ("June, Bounty Hunter", "StaticAbility::CantBeBlockedIf"),
    ("Giant Cindermaw", "StaticAbility::PlayersCantGainLife"),
    ("Tiger-Dillo", "StaticAbility::CantAttackOrBlockUnlessYouControlAnother"),
    ("Spirit", "BlockRestriction::CantBlock"),
    ("Spirit", "BlockRestriction::CantBeBlockedBy"),
    ("Gran-Gran", "StaticAbility::SpellsCostLess"),
    ("Allies at Last", "StaticAbility::CostsLessPerPermanent"),
    ("Proft's Eidetic Memory", "StaticAbility::NoMaximumHandSize"),
    ("First-Time Flyer", "StaticAbility::PtBuff"),
    ("Suki, Kyoshi Warrior", "StaticAbility::CharacteristicPower"),
    ("Waterfall Aerialist", "StaticAbility::Ward"),
    ("Deadly Precision", "AdditionalCost::Sacrifice"),
    ("Firebending Lesson", "AdditionalCost::Kicker"),
]


def _characteristics(
    catalog: managym.LearningCatalog, name: str
) -> list[tuple[int, int]]:
    row = catalog["registry_names"].index(name)
    start, end = catalog["definition_offsets"][row : row + 2]
    return list(
        zip(
            catalog["token_kind"][start:end],
            catalog["token_value"][start:end],
            strict=True,
        )
    )


@pytest.mark.parametrize("name,variant", VARIANTS)
def test_variant_token_is_in_the_card_characteristics(name: str, variant: str) -> None:
    tokens = _characteristics(managym.learning_catalog(), name)
    assert (TOKEN_KINDS["variant"], token_value(variant)) in tokens


def test_cards_without_statics_carry_no_static_tokens() -> None:
    tokens = _characteristics(managym.learning_catalog(), "Gray Ogre")
    values = {value for kind, value in tokens if kind == TOKEN_KINDS["field"]}
    assert token_value("statics") not in values
    assert token_value("additional_costs") not in values
