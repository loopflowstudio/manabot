//! Dragons of Tarkir.
use managym_state::{
    ability::{Effect, ObjectSelector, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, StaticAbility},
    mana::{Color, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![rending_volley()]
}

fn rending_volley() -> CardDefinition {
    let text_box = "This spell can't be countered.\nRending Volley deals 4 damage to target white or blue creature.";
    let white_or_blue_creature = TargetSpec::Permanent {
        predicate: CardPredicate {
            card_type: Some(CardType::Creature),
            colors_any: vec![Color::White, Color::Blue],
            ..Default::default()
        },
    };
    CardDefinition {
        key: "dtk.rending_volley",
        scryfall_id: "8234090e-9df1-4915-90ef-8a4bc6212655",
        name: "Rending Volley".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Instant]),
        statics: vec![StaticAbility::CantBeCountered],
        spell_effects: vec![Effect::deal_damage(
            4,
            ObjectSelector::Target(white_or_blue_creature),
        )],
        text_box: text_box.into(),
        ..Default::default()
    }
}
