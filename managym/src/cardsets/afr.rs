//! Adventures in the Forgotten Realms.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Cost, Op, Predicate, Program, ProgramKind, Sacrifice,
    Selector, Target,
};

pub(super) fn cards() -> Vec<Card> {
    vec![power_word_kill(), deadly_dispute()]
}

fn power_word_kill() -> Card {
    let text_box = "Destroy target non-Angel, non-Demon, non-Devil, non-Dragon creature.";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{B}"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "afr.power_word_kill",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Destroy { target: "target_0" }],
        )
        .cost(Cost::mana("{1}{B}"))
        .targets(vec![Target::one(
            "target_0",
            Selector::Permanent {
                predicate: Predicate::All {
                    predicates: vec![
                        Predicate::NotSubtypes {
                            values: vec!["Angel", "Demon", "Devil", "Dragon"],
                        },
                        Predicate::CardType {
                            value: CardType::Creature,
                        },
                    ],
                },
            },
        )])
        .inferred()],
        registry_name: "Power Word Kill",
        text_box,
    }
}

fn deadly_dispute() -> Card {
    let text_box = "As an additional cost to cast this spell, sacrifice an artifact or creature.\nDraw two cards and create a Treasure token. (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("1B"),
            subtypes: Some(vec![]),
            types: vec![CardType::Instant],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "afr.deadly_dispute",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Spell,
            vec![
                Op::DrawCards { count: 2 },
                Op::CreateToken {
                    count: 1,
                    definition_ref: "token.treasure",
                    tapped_and_attacking: Some(false),
                },
            ],
        )
        .cost(Cost {
            mana: Some("1B"),
            sacrifice: Some(Sacrifice {
                another: false,
                or_mana: None,
                predicate: Predicate::TypesAny {
                    values: vec![CardType::Artifact, CardType::Creature],
                },
            }),
            ..Default::default()
        })],
        registry_name: "Deadly Dispute",
        text_box,
    }
}
