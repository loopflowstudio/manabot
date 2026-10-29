//! Core Set 2020.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Cost, Keyword, Op, PrintedPower, Program, ProgramKind,
};

pub(super) fn cards() -> Vec<Card> {
    vec![spectral_sailor()]
}

fn spectral_sailor() -> Card {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nFlying\n{3}{U}: Draw a card.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{U}"),
            subtypes: Some(vec!["Spirit", "Pirate"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying, Keyword::Flash]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "m20.spectral_sailor",
        programs: vec![Program::new(
            "activate_0",
            ProgramKind::Activated,
            vec![Op::DrawCards { count: 1 }],
        )
        .cost(Cost {
            mana: Some("{3}{U}"),
            sacrifice_source: Some(false),
            tap_source: Some(false),
            waterbend: Some(false),
            ..Default::default()
        })],
        registry_name: "Spectral Sailor",
        text_box,
    }
}
