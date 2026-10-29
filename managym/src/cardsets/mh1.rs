//! Modern Horizons.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Cost, Duration, Keyword, Op, Program, ProgramKind, Quantity,
    Selector, Target,
};

pub(super) fn cards() -> Vec<Card> {
    vec![fists_of_flame(), prismatic_vista()]
}

fn fists_of_flame() -> Card {
    let text_box = "Draw a card. Until end of turn, target creature gains trample and gets +1/+0 for each card you've drawn this turn. (A creature with trample can deal excess combat damage to the player or planeswalker it's attacking.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{R}"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "mh1.fists_of_flame",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::DrawCards { count: 1 },
                Op::BuffTargetValue {
                    power: Quantity::CardsDrawnMinus { subtract: 0 },
                    target: "target_0",
                    toughness: 0,
                },
                Op::GrantKeywords {
                    duration: Duration::EndOfTurn,
                    keywords: vec![Keyword::Trample],
                    target: "target_0",
                },
            ],
        )
        .cost(Cost::mana("{1}{R}"))
        .targets(vec![Target::one("target_0", Selector::Creature)])
        .inferred()],
        registry_name: "Fists of Flame",
        text_box,
    }
}

fn prismatic_vista() -> Card {
    let text_box = "{T}, Pay 1 life, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield, then shuffle.";

    Card {
        characteristics: Characteristics {
            subtypes: Some(vec![]),
            types: vec![CardType::Land],
            keywords: Some(vec![]),
            token: Some(false),
            supertypes: Some(vec![]),
            ..Default::default()
        },
        key: "mh1.prismatic_vista",
        programs: vec![Program::new(
            "p0",
            ProgramKind::Activated,
            vec![Op::SearchBasicLands {
                bottom_source: false,
                count: 1,
                tapped: false,
                untap_if_lands: None,
            }],
        )
        .cost(Cost {
            mana: Some("0"),
            sacrifice_source: Some(true),
            tap_source: Some(true),
            life: Some(1),
            ..Default::default()
        })],
        registry_name: "Prismatic Vista",
        text_box,
    }
}
