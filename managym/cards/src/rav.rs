//! Ravnica cards expressed in the shared proposed-event rules.
use managym_state::{
    card::{CardDefinition, CardType, CardTypes, ReplacementEffect},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![doubling_season()]
}

fn doubling_season() -> CardDefinition {
    let text_box = "If an effect would create one or more tokens under your control, it creates twice that many of those tokens instead.\nIf an effect would put one or more counters on a permanent you control, it puts twice that many of those counters on that permanent instead.";
    CardDefinition {
        key: "rav.doubling_season",
        scryfall_id: "c7e71299-98f6-494e-b187-8d22ce5f50af",
        name: "Doubling Season".into(),
        mana_cost: Some(ManaCost::parse("{4}{G}")),
        types: CardTypes::new([CardType::Enchantment]),
        replacement_effects: vec![
            ReplacementEffect::MultiplyTokens { multiplier: 2 },
            ReplacementEffect::ModifyCounters {
                effects_only: true,
                kind: None,
                predicate: CardPredicate::default(),
                multiplier: 2,
                additional: 0,
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}
