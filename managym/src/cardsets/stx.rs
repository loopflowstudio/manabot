//! Strixhaven: School of Mages.
use crate::cards::authoring::{
    Card, CardType, Characteristics, Cost, Duration, Event, Keyword, Op, PrintedPower, Program,
    ProgramKind, Selector, Subject, Target, Trigger,
};

pub(super) fn cards() -> Vec<Card> {
    vec![
        enthusiastic_study(),
        eyetwitch(),
        field_trip(),
        gnarled_professor(),
        professor_of_symbology(),
        divide_by_zero(),
        igneous_inspiration(),
        pop_quiz(),
    ]
}

fn enthusiastic_study() -> Card {
    let text_box = "Target creature gets +3/+1 and gains trample until end of turn.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{R}"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "stx.enthusiastic_study",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::ModifyPt {
                    duration: Duration::EndOfTurn,
                    power: 3,
                    target: "target_0",
                    toughness: 1,
                },
                Op::GrantKeywords {
                    duration: Duration::EndOfTurn,
                    keywords: vec![Keyword::Trample],
                    target: "target_0",
                },
                Op::Learn,
            ],
        )
        .cost(Cost::mana("{2}{R}"))
        .targets(vec![Target::one("target_0", Selector::Creature)])
        .inferred()],
        registry_name: "Enthusiastic Study",
        text_box,
    }
}

fn eyetwitch() -> Card {
    let text_box = "Flying\nWhen this creature dies, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{B}"),
            subtypes: Some(vec!["Eye", "Bat"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Flying]),
            power: Some(PrintedPower::Fixed(1)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "stx.eyetwitch",
        programs: vec![
            Program::new("trigger_0", ProgramKind::Triggered, vec![Op::Learn])
                .trigger(Trigger::new(Event::Dies, Subject::This)),
        ],
        registry_name: "Eyetwitch",
        text_box,
    }
}

fn field_trip() -> Card {
    let text_box = "Search your library for a basic Forest card, put that card onto the battlefield tapped, then shuffle.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{G}"),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "stx.field_trip",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::SearchBasicLand {
                    destination: None,
                    subtype: Some("Forest"),
                },
                Op::Learn,
            ],
        )
        .cost(Cost::mana("{2}{G}"))
        .inferred()],
        registry_name: "Field Trip",
        text_box,
    }
}

fn gnarled_professor() -> Card {
    let text_box = "Trample\nWhen this creature enters, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{2}{G}{G}"),
            subtypes: Some(vec!["Treefolk", "Druid"]),
            types: vec![CardType::Creature],
            keywords: Some(vec![Keyword::Trample]),
            power: Some(PrintedPower::Fixed(5)),
            toughness: Some(4),
            ..Default::default()
        },
        key: "stx.gnarled_professor",
        programs: vec![
            Program::new("trigger_0", ProgramKind::Triggered, vec![Op::Learn])
                .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
        ],
        registry_name: "Gnarled Professor",
        text_box,
    }
}

fn professor_of_symbology() -> Card {
    let text_box = "When this creature enters, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("{1}{W}"),
            subtypes: Some(vec!["Kor", "Cleric"]),
            types: vec![CardType::Creature],
            power: Some(PrintedPower::Fixed(2)),
            toughness: Some(1),
            ..Default::default()
        },
        key: "stx.professor_of_symbology",
        programs: vec![
            Program::new("trigger_0", ProgramKind::Triggered, vec![Op::Learn])
                .trigger(Trigger::new(Event::EntersBattlefield, Subject::This)),
        ],
        registry_name: "Professor of Symbology",
        text_box,
    }
}

fn divide_by_zero() -> Card {
    let text_box = "Return target spell or permanent with mana value 1 or greater to its owner's hand.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2U"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "stx.divide_by_zero",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::ReturnToHand { target: "target" }, Op::Learn],
        )
        .cost(Cost::mana("2U"))
        .targets(vec![Target::one(
            "target",
            Selector::SpellOrPermanent { min_mana_value: 1 },
        )])],
        registry_name: "Divide by Zero",
        text_box,
    }
}

fn igneous_inspiration() -> Card {
    let text_box = "Igneous Inspiration deals 3 damage to any target.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2R"),
            types: vec![CardType::Sorcery],
            ..Default::default()
        },
        key: "stx.igneous_inspiration",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![
                Op::DealDamage {
                    amount: 3,
                    target: "victim",
                },
                Op::Learn,
            ],
        )
        .cost(Cost::mana("2R"))
        .targets(vec![Target::one("victim", Selector::CreatureOrPlayer)])],
        registry_name: "Igneous Inspiration",
        text_box,
    }
}

fn pop_quiz() -> Card {
    let text_box = "Draw a card.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    Card {
        characteristics: Characteristics {
            mana_cost: Some("2U"),
            types: vec![CardType::Instant],
            ..Default::default()
        },
        key: "stx.pop_quiz",
        programs: vec![Program::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::DrawCards { count: 1 }, Op::Learn],
        )
        .cost(Cost::mana("2U"))],
        registry_name: "Pop Quiz",
        text_box,
    }
}
