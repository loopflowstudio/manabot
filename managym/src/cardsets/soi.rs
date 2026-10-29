//! Shadows over Innistrad.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Event, Op, Predicate, PrintedPower, Program, ProgramKind,
    Subject, Trigger,
};

pub(super) fn cards() -> Vec<Card> {
    vec![tireless_tracker()]
}

fn tireless_tracker() -> Card {
    let text_box = "Landfall — Whenever a land you control enters, investigate. (Create a Clue token. It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")\nWhenever you sacrifice a Clue, put a +1/+1 counter on this creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2G"),
            subtypes: Some(vec!["Human", "Scout"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(2),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "soi.tireless_tracker",
        programs: vec![
            Program::new(
                "p0",
                ProgramKind::Triggered,
                vec![Op::CreateToken {
                    count: 1,
                    definition_ref: "token.clue",
                    tapped_and_attacking: Some(false),
                }],
            )
            .trigger(Trigger::new(
                Event::EntersBattlefield,
                Subject::AnyYouControl {
                    predicate: Predicate::CardType {
                        value: CardType::Land,
                    },
                },
            )),
            Program::new(
                "p1",
                ProgramKind::Triggered,
                vec![Op::PutCounters {
                    count: 1,
                    target: "source",
                }],
            )
            .trigger(Trigger {
                predicate: Some(Predicate::Subtype { value: "Clue" }),
                controller_only: Some(true),
                ..Trigger::new(Event::Sacrificed, Subject::This)
            }),
        ],
        registry_name: "Tireless Tracker",
        text_box,
    }
}
