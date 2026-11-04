//! Wilds of Eldraine.
use managym_state::{
    ability::Effect,
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![scarecrow_guide()]
}

fn scarecrow_guide() -> CardDefinition {
    let text_box = "Reach\n{1}: Add one mana of any color. Activate only once each turn.";
    CardDefinition {
        key: "woe.scarecrow_guide",
        scryfall_id: "3dc08461-bec6-4a18-b782-da4c92abb789",
        name: "Scarecrow Guide".into(),
        mana_cost: Some(ManaCost::parse("{2}")),
        types: CardTypes::new([CardType::Artifact, CardType::Creature]),
        subtypes: vec!["Scarecrow".into()],
        power: Some(2),
        toughness: Some(1),
        keywords: Keywords::of(&[Keyword::Reach]),
        activated_abilities: vec![ActivatedAbilityDefinition {
            once_each_turn: true,
            mana_cost: ManaCost::parse("1"),
            effects: vec![Effect::any_color_mana()],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
