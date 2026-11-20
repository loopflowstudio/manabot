//! Mirrodin.
use managym_state::{
    ability::Effect,
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
    },
    mana::{Mana, ManaCost},
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![gilded_lotus()]
}

fn gilded_lotus() -> CardDefinition {
    let text_box = "{T}: Add three mana of any one color.";
    CardDefinition {
        key: "mrd.gilded_lotus",
        scryfall_id: "a1d5e4c8-dfd0-45bc-8000-ebfaccfefec3",
        printed: PrintedCharacteristics {
            name: "Gilded Lotus".into(),
            mana_cost: Some(ManaCost::parse("{5}")),
            types: CardTypes::new([CardType::Artifact]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::mana_choice(
                    ["WWW", "UUU", "BBB", "RRR", "GGG"].map(Mana::parse),
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
