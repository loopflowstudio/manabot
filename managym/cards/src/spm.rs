//! Marvel’s Spider-Man.
use managym_state::{
    ability::Effect,
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, TapCost},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![supportive_parents()]
}

fn supportive_parents() -> CardDefinition {
    let text_box = "Tap two untapped creatures you control: Add one mana of any color.";
    CardDefinition {
        key: "spm.supportive_parents",
        scryfall_id: "d3fe8a5b-4166-46cc-b910-71cd1a19ae1b",
        name: "Supportive Parents".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Citizen".into()],
        power: Some(3),
        toughness: Some(3),
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_cost: Some(TapCost {
                predicate: CardPredicate::creature(),
                count: 2,
            }),
            effects: vec![Effect::any_color_mana()],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}
