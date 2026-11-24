//! Portal Second Age.
use managym_state::{
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![bear_cub()]
}

fn bear_cub() -> CardDefinition {
    CardDefinition {
        key: "p02.bear_cub",
        scryfall_id: "d71abb70-bee5-4823-83dc-db0707023b37",
        printed: PrintedCharacteristics {
            name: "Bear Cub".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bear".into()],
            power: Some(2),
            toughness: Some(2),
            ..Default::default()
        },
        ..Default::default()
    }
}
