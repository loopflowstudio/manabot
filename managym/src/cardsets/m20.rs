//! Core Set 2020.
use crate::state::{
    ability::Effect,
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keywords},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![spectral_sailor()]
}

fn spectral_sailor() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nFlying\n{3}{U}: Draw a card.";

    CardDefinition {
        key: "m20.spectral_sailor",
        scryfall_id: "67483891-36d1-46f2-8b4f-b8b7bd54bdcc",
        name: "Spectral Sailor".into(),
        mana_cost: Some(ManaCost::parse("{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Spirit".into(), "Pirate".into()],
        power: Some(1),
        toughness: Some(1),
        keywords: Keywords {
            flying: true,
            flash: true,
            ..Default::default()
        },
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{3}{U}"),
            effects: vec![Effect::DrawCards { count: 1 }],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
