//! Khans of Tarkir.
use crate::cards::authoring::{Card, CardType, Characteristics, Keyword, PrintedPower, Program};

pub(super) fn cards() -> Vec<Card> {
    vec![monastery_swiftspear()]
}

fn monastery_swiftspear() -> Card {
    let text_box = "Haste\nProwess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{R}"),
            subtypes: Some(vec!["Human", "Monk"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Haste]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(2),
            ..Default::default()
        },
        key: "ktk.monastery_swiftspear",
        programs: vec![Program::prowess("trigger_0")],
        registry_name: "Monastery Swiftspear",
        text_box,
    }
}
