//! Zendikar.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Event, Op, Predicate, PrintedPower, Program, ProgramKind,
    Subject, Trigger,
};

pub(super) fn cards() -> Vec<Card> {
    vec![lotus_cobra()]
}

fn lotus_cobra() -> Card {
    let text_box = "Landfall — Whenever a land you control enters, add one mana of any color.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1G"),
            subtypes: Some(vec!["Snake"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(1),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "zen.lotus_cobra",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Triggered,
            vec![Op::choose_mana(&["W", "U", "B", "R", "G"])],
        )
        .trigger(Trigger::new(
            Event::EntersBattlefield,
            Subject::AnyYouControl {
                predicate: Predicate::CardType {
                    value: CardType::Land,
                },
            },
        ))],
        registry_name: "Lotus Cobra",
        text_box,
    }
}
