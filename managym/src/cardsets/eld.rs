//! Throne of Eldraine.
use crate::cards::authoring::{Card, CardType, Characteristics, Cost, Op, Program, ProgramKind};

pub(super) fn cards() -> Vec<Card> {
    vec![fabled_passage()]
}

fn fabled_passage() -> Card {
    let text_box = "{T}, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield tapped, then shuffle. Then if you control four or more lands, untap that land.";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec![]),
            types: vec![CardType::Land],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "eld.fabled_passage",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Activated,
            vec![Op::SearchBasicLands {
                bottom_source: false,
                count: 1,
                tapped: true,
                untap_if_lands: Some(4),
            }],
        )
        .cost(Cost {
            mana: Some("0"),
            sacrifice_source: Some(true),
            tap_source: Some(true),
            ..Default::default()
        })],
        registry_name: "Fabled Passage",
        text_box,
    }
}
