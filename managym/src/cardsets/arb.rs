//! Alara Reborn.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Cost, Event, Op, Program, ProgramKind, Subject, Trigger,
};

pub(super) fn cards() -> Vec<Card> {
    vec![prophetic_prism()]
}

fn prophetic_prism() -> Card {
    let text_box = "When this artifact enters, draw a card.\n{1}, {T}: Add one mana of any color.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}"),
            subtypes: Some(vec![]),
            types: vec![CardType::Artifact],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "arb.prophetic_prism",
        programs: vec![
            Program::new(
                "p0",
                ProgramKind::Triggered,
                vec![Op::DrawCards { count: 1 }],
            )
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "p1",
                ProgramKind::Activated,
                vec![Op::choose_mana(&["W", "U", "B", "R", "G"])],
            )
            .cost(Cost {
                mana: Some("1"),
                tap_source: Some(true),
                ..Default::default()
            }),
        ],
        registry_name: "Prophetic Prism",
        text_box,
    }
}
