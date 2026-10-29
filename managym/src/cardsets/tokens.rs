//! Tokens created by cards outside TLA. TLA defines its own fixed tokens.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Color, Cost, Op, PrintedPower, Program, ProgramKind,
};

pub(super) fn cards() -> Vec<Card> {
    vec![human_soldier(), treasure()]
}

fn human_soldier() -> Card {
    let text_box = "";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec!["Human", "Soldier"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            token: Some(true),
            supertypes: Some(vec![]),
            colors: Some(vec![Color::W]),
            ..Default::default()
        },
        key: "token.human_soldier",
        programs: vec![],
        registry_name: "Human Soldier",
        text_box,
    }
}

fn treasure() -> Card {
    let text_box = "{T}, Sacrifice this token: Add one mana of any color.";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec!["Treasure"]),
            types: vec![CardType::Artifact],
            keywords: Some(vec![]),
            token: Some(true),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "token.treasure",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Activated,
            vec![Op::choose_mana(&["W", "U", "B", "R", "G"])],
        )
        .cost(Cost {
            mana: Some("0"),
            sacrifice_source: Some(true),
            tap_source: Some(true),
            ..Default::default()
        })],
        registry_name: "Treasure",
        text_box,
    }
}
