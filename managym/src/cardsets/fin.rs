//! Final Fantasy.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Event, Op, Predicate, PrintedPower, Program, ProgramKind,
    Subject, Trigger,
};

pub(super) fn cards() -> Vec<Card> {
    vec![sazh_s_chocobo()]
}

fn sazh_s_chocobo() -> Card {
    let text_box =
        "Landfall — Whenever a land you control enters, put a +1/+1 counter on this creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("G"),
            subtypes: Some(vec!["Bird"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(0)),
            toughness: Some(1),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "fin.sazh_s_chocobo",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Triggered,
            vec![Op::PutCounters {
                count: 1,
                counter: None,
                target: "source",
            }],
        )
        .trigger(Trigger::new(
            Event::EntersBattlefield,
            Subject::AnyYouControl {
                predicate: Predicate::CardType {
                    value: CardType::Land,
                },
            },
        ))],
        registry_name: "Sazh's Chocobo",
        text_box,
    }
}
