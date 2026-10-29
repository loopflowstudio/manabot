//! Onslaught.
use crate::cards::authoring::{Card, CardType, Characteristics, PrintedPower};

pub(super) fn cards() -> Vec<Card> {
    vec![glory_seeker()]
}

fn glory_seeker() -> Card {
    let text_box = "";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1W"),
            subtypes: Some(vec!["Human", "Soldier"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "ons.glory_seeker",
        programs: vec![],
        registry_name: "Glory Seeker",
        text_box,
    }
}
