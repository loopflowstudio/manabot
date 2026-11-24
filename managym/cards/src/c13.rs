//! Commander 2013.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TargetSpec, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![angel_of_finality()]
}

fn angel_of_finality() -> CardDefinition {
    CardDefinition {
        key: "c13.angel_of_finality",
        scryfall_id: "bd3c34c9-2072-4ebb-93ef-34173015bfb8",
        printed: PrintedCharacteristics {
            name: "Angel of Finality".into(),
            subtypes: vec!["Angel".into()],
            types: CardTypes::new([CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{3}{W}")),
            power: Some(3),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::exile(ObjectSelector::TargetPlayersZone {
                    spec: Box::new(TargetSpec::Player),
                    zone: ZoneType::Graveyard,
                })],
            )],
            text_box: "Flying\nWhen this creature enters, exile target player's graveyard.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
