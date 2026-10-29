//! Limited Edition Alpha.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Cost, Op, PrintedPower, Program, ProgramKind, Selector, Target,
};

pub(super) fn cards() -> Vec<Card> {
    vec![hill_giant(), lightning_bolt(), savannah_lions()]
}

fn hill_giant() -> Card {
    let text_box = "";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("3R"),
            subtypes: Some(vec!["Giant"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(3)),
            toughness: Some(3),
            ..Default::default()
        },
        key: "lea.hill_giant",
        programs: vec![],
        registry_name: "Hill Giant",
        text_box,
    }
}

fn lightning_bolt() -> Card {
    let text_box = "Lightning Bolt deals 3 damage to any target.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("R"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "lea.lightning_bolt",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::DealDamage {
                amount: 3,
                target: "victim",
            }],
        )
        .cost(Cost::mana("R"))
        .targets(vec![Target::one("victim", Selector::CreatureOrPlayer)])],
        registry_name: "Lightning Bolt",
        text_box,
    }
}

fn savannah_lions() -> Card {
    let text_box = "";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("W"),
            subtypes: Some(vec!["Cat"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "lea.savannah_lions",
        programs: vec![],
        registry_name: "Savannah Lions",
        text_box,
    }
}
