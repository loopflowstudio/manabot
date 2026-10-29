//! Bloomburrow.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Cost, Duration, Op, Program, ProgramKind, Selector, Target,
};

pub(super) fn cards() -> Vec<Card> {
    vec![savor()]
}

fn savor() -> Card {
    let text_box = "Target creature gets -2/-2 until end of turn. Create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{B}"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "blb.savor",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::ModifyPt {
                    duration: Duration::EndOfTurn,
                    power: -2,
                    target: "target_0",
                    toughness: -2,
                    per_creature_type_max: None,
                    per_color_among: None,
                },
                Op::CreateToken {
                    count: 1,
                    definition_ref: "token.food",
                    tapped_and_attacking: Some(false),
                },
            ],
        )
        .cost(Cost::mana("{1}{B}"))
        .targets(vec![Target::one("target_0", Selector::Creature)])
        .inferred()],
        registry_name: "Savor",
        text_box,
    }
}
