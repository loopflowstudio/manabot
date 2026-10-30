//! Strixhaven: School of Mages.
use crate::state::{
    ability::{Ability, Effect, TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, Keywords},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        divide_by_zero(),
        enthusiastic_study(),
        eyetwitch(),
        field_trip(),
        gnarled_professor(),
        igneous_inspiration(),
        pop_quiz(),
        professor_of_symbology(),
        waterfall_aerialist(),
    ]
}

fn divide_by_zero() -> CardDefinition {
    let text_box = "Return target spell or permanent with mana value 1 or greater to its owner's hand.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.divide_by_zero",
        scryfall_id: "1958d96e-ec44-48ab-80b1-5b01a24ac7b8",
        name: "Divide by Zero".into(),
        mana_cost: Some(ManaCost::parse("{2}{U}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![
            Effect::ReturnToHand {
                target: TargetSpec::SpellOrPermanent { min_mana_value: 1 },
            },
            Effect::Learn,
        ],
        targeting: vec![TargetRequirement {
            spec: TargetSpec::SpellOrPermanent { min_mana_value: 1 },
            min: 1,
            max: 1,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn enthusiastic_study() -> CardDefinition {
    let text_box = "Target creature gets +3/+1 and gains trample until end of turn.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.enthusiastic_study",
        scryfall_id: "543c64ff-2c51-4a63-a940-dc8645717c85",
        name: "Enthusiastic Study".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![
            Effect::BuffTarget {
                power: 3,
                toughness: 1,
                target: TargetSpec::Creature,
            },
            Effect::GrantKeywordsToTarget {
                keywords: Keywords {
                    trample: true,
                    ..Default::default()
                },
                target: TargetSpec::Creature,
            },
            Effect::Learn,
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn eyetwitch() -> CardDefinition {
    let text_box = "Flying\nWhen this creature dies, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.eyetwitch",
        scryfall_id: "1f4d1bb6-cb8f-4d01-9879-0b3a0585cbf4",
        name: "Eyetwitch".into(),
        mana_cost: Some(ManaCost::parse("{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Eye".into(), "Bat".into()],
        power: Some(1),
        toughness: Some(1),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::Learn],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn field_trip() -> CardDefinition {
    let text_box = "Search your library for a basic Forest card, put that card onto the battlefield tapped, then shuffle.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.field_trip",
        scryfall_id: "f235060a-eb49-4a73-bb5f-01228c3c4070",
        name: "Field Trip".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![
            Effect::SearchBasicLand {
                subtype: Some("Forest".into()),
                count: 1,
                tapped: true,
                untap_if_lands: None,
                bottom_source: false,
            },
            Effect::Learn,
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn gnarled_professor() -> CardDefinition {
    let text_box = "Trample\nWhen this creature enters, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.gnarled_professor",
        scryfall_id: "a32338e8-1f6a-49b9-bd93-26578adab6b3",
        name: "Gnarled Professor".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Treefolk".into(), "Druid".into()],
        power: Some(5),
        toughness: Some(4),
        keywords: Keywords {
            trample: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::Learn],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn igneous_inspiration() -> CardDefinition {
    let text_box = "Igneous Inspiration deals 3 damage to any target.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.igneous_inspiration",
        scryfall_id: "5781ad7b-dc1b-4cc1-9e72-6e714b9ba1de",
        name: "Igneous Inspiration".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![
            Effect::DealDamage {
                amount: 3,
                target: TargetSpec::CreatureOrPlayer,
            },
            Effect::Learn,
        ],
        targeting: vec![TargetRequirement {
            spec: TargetSpec::CreatureOrPlayer,
            min: 1,
            max: 1,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn pop_quiz() -> CardDefinition {
    let text_box = "Draw a card.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.pop_quiz",
        scryfall_id: "d16892d8-9d10-45de-ab79-0e645c4b5588",
        name: "Pop Quiz".into(),
        mana_cost: Some(ManaCost::parse("{2}{U}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![Effect::DrawCards { count: 1 }, Effect::Learn],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn professor_of_symbology() -> CardDefinition {
    let text_box = "When this creature enters, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.professor_of_symbology",
        scryfall_id: "f427cf73-9f5e-4ef5-bc4f-29ffbfda9d57",
        name: "Professor of Symbology".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Kor".into(), "Cleric".into()],
        power: Some(2),
        toughness: Some(1),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::Learn],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn waterfall_aerialist() -> CardDefinition {
    let text_box = "Flying\nWard {2} (Whenever this creature becomes the target of a spell or ability an opponent controls, counter it unless that player pays {2}.)";

    CardDefinition {
        key: "stx.waterfall_aerialist",
        scryfall_id: "7bc6966c-18d9-4675-9594-6f0b2d11c405",
        name: "Waterfall Aerialist".into(),
        mana_cost: Some(ManaCost::parse("{3}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Djinn".into(), "Wizard".into()],
        power: Some(3),
        toughness: Some(1),
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        ward: Some(ManaCost::parse("{2}")),
        text_box: text_box.into(),
        ..Default::default()
    }
}
