//! Mirrodin.
use managym_state::{
    ability::Effect,
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics,
    },
    mana::{Mana, ManaCost},
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![leonin_skyhunter(), gilded_lotus()]
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

fn leonin_skyhunter() -> CardDefinition {
    let text_box = "Flying";
    CardDefinition {
        key: "mrd.leonin_skyhunter",
        scryfall_id: "275a47e1-816c-44f9-bd05-b8b56410436f",
        printed: PrintedCharacteristics {
            name: "Leonin Skyhunter".into(),
            mana_cost: Some(ManaCost::parse("{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Knight".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Flying]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
