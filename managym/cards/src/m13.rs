//! Magic 2013.
use managym_state::{
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics, StaticAbility},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![omniscience()]
}

fn omniscience() -> CardDefinition {
    let rules_text = "You may cast spells from your hand without paying their mana costs.";
    CardDefinition {
        key: "m13.omniscience",
        scryfall_id: "1088f33e-cb5f-4248-ae8e-280c4e41f291",
        printed: PrintedCharacteristics {
            name: "Omniscience".into(),
            mana_cost: Some(ManaCost::parse("{7}{U}{U}{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            statics: vec![StaticAbility::CastFromHandWithoutManaCost],
            text_box: rules_text.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
