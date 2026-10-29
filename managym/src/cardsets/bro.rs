//! The Brothers' War.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Keyword, Op, PrintedPower, Program, ProgramKind,
};

pub(super) fn cards() -> Vec<Card> {
    vec![giant_cindermaw()]
}

fn giant_cindermaw() -> Card {
    let text_box = "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nPlayers can't gain life.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{R}"),
            subtypes: Some(vec!["Dinosaur", "Beast"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Trample]),
            power: Some(PrintedPower::Fixed(4)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "bro.giant_cindermaw",
        programs: vec![Program::new(
            "prevent_life_gain",
            ProgramKind::Static,
            vec![Op::PreventLifeGain],
        )],
        registry_name: "Giant Cindermaw",
        text_box,
    }
}
