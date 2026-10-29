//! The Lost Caverns of Ixalan.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Event, Op, Predicate, PrintedPower, Program, ProgramKind,
    Subject, Trigger,
};

pub(super) fn cards() -> Vec<Card> {
    vec![tireless_provisioner()]
}

fn tireless_provisioner() -> Card {
    let text_box = "Landfall — Whenever a land you control enters, create a Food token or a Treasure token. (Food is an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\" Treasure is an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2G"),
            subtypes: Some(vec!["Elf", "Scout"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(2),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "lci.tireless_provisioner",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Triggered,
            vec![Op::Modal {
                modes: vec![
                    vec![Op::CreateToken {
                        count: 1,
                        definition_ref: "token.food",
                        tapped_and_attacking: Some(false),
                    }],
                    vec![Op::CreateToken {
                        count: 1,
                        definition_ref: "token.treasure",
                        tapped_and_attacking: Some(false),
                    }],
                ],
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
        registry_name: "Tireless Provisioner",
        text_box,
    }
}
