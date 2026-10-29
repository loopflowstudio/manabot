//! Ikoria: Lair of Behemoths.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Cost, Event, Op, Predicate, Program, ProgramKind, Sacrifice,
    Selector, Subject, Target, Trigger,
};

pub(super) fn cards() -> Vec<Card> {
    vec![bastion_of_remembrance(), weaponize_the_monsters()]
}

fn bastion_of_remembrance() -> Card {
    let text_box = "When this enchantment enters, create a 1/1 white Human Soldier creature token.\nWhenever a creature you control dies, each opponent loses 1 life and you gain 1 life.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2B"),
            subtypes: Some(vec![]),
            types: vec![CardType::Enchantment],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "iko.bastion_of_remembrance",
        programs: vec![
            Program::new(
                "p0",
                ProgramKind::Triggered,
                vec![Op::CreateToken {
                    count: 1,
                    definition_ref: "token.human_soldier",
                    tapped_and_attacking: Some(false),
                }],
            )
            .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
            Program::new(
                "p1",
                ProgramKind::Triggered,
                vec![Op::DrainLife { amount: 1 }],
            )
            .trigger(Trigger::new(
                Event::Dies,
                Subject::AnyYouControl {
                    predicate: Predicate::CardType {
                        value: CardType::Creature,
                    },
                },
            )),
        ],
        registry_name: "Bastion of Remembrance",
        text_box,
    }
}

fn weaponize_the_monsters() -> Card {
    let text_box = "{2}, Sacrifice a creature: This enchantment deals 2 damage to any target.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("R"),
            subtypes: Some(vec![]),
            types: vec![CardType::Enchantment],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "iko.weaponize_the_monsters",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Activated,
            vec![Op::DealDamage {
                amount: 2,
                target: "target_0",
            }],
        )
        .cost(Cost {
            mana: Some("2"),
            sacrifice: Some(Sacrifice {
                another: false,
                or_mana: None,
                predicate: Predicate::CardType {
                    value: CardType::Creature,
                },
            }),
            ..Default::default()
        })
        .targets(vec![Target::one("target_0", Selector::CreatureOrPlayer)])
        .inferred()],
        registry_name: "Weaponize the Monsters",
        text_box,
    }
}
