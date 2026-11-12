//! Dragons of Tarkir.
use managym_state::{
    ability::{Effect, ObjectSelector, Scope, TargetSpec},
    card::{CardDefinition, CardType, CardTypes, StaticAbility},
    mana::{Color, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![clone_legion(), rending_volley()]
}

fn clone_legion() -> CardDefinition {
    let text_box =
        "For each creature target player controls, create a token that's a copy of that creature.";
    CardDefinition {
        key: "dtk.clone_legion",
        scryfall_id: "45c6a201-1d4d-4ae5-a84e-098e094eaf61",
        name: "Clone Legion".into(),
        mana_cost: Some(ManaCost::parse("{7}{U}{U}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::CreateTokenCopy {
            of: ObjectSelector::each(Scope::TargetPlayerControls, CardPredicate::creature()),
            exceptions: vec![],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
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
