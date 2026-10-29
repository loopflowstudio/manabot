//! War of the Spark.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Event, Op, Predicate, PrintedPower, Program, ProgramKind,
    Selector, Subject, Target, Trigger,
};

pub(super) fn cards() -> Vec<Card> {
    vec![mayhem_devil()]
}

fn mayhem_devil() -> Card {
    let text_box =
        "Whenever a player sacrifices a permanent, this creature deals 1 damage to any target.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1BR"),
            subtypes: Some(vec!["Devil"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(3),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "war.mayhem_devil",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Triggered,
            vec![Op::DealDamage {
                amount: 1,
                target: "target_0",
            }],
        )
        .targets(vec![Target::one("target_0", Selector::CreatureOrPlayer)])
        .trigger(Trigger {
            predicate: Some(Predicate::Any),
            controller_only: Some(false),
            ..Trigger::new(Event::Sacrificed, Subject::This)
        })
        .inferred()],
        registry_name: "Mayhem Devil",
        text_box,
    }
}
