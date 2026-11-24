//! Oath of the Gatewatch.
use managym_state::{
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![tajuru_pathwarden()]
}

fn tajuru_pathwarden() -> CardDefinition {
    let text_box = "Vigilance (Attacking doesn't cause this creature to tap.)\nTrample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)";
    CardDefinition {
        key: "ogw.tajuru_pathwarden",
        scryfall_id: "b073e75b-b432-41a5-a71e-d169fecf774f",
        printed: PrintedCharacteristics {
            name: "Tajuru Pathwarden".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Warrior".into(), "Ally".into()],
            power: Some(5),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Vigilance, Keyword::Trample]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
