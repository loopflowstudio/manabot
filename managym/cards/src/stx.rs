//! Strixhaven: School of Mages.
use managym_state::effect::Modification;
use managym_state::{
    ability::{
        Ability, Effect, EffectValue, LibraryEnd, ObjectSelector, PaymentCost, TargetRequirement,
        TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        StaticAbility,
    },
    mana::ManaCost,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        deadly_brew(),
        teach_by_example(),
        campus_guide(),
        burrog_befuddler(),
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
        printed: PrintedCharacteristics {
            name: "Divide by Zero".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::return_to_hand(ObjectSelector::Target(TargetSpec::SpellOrPermanent {
                    min_mana_value: 1,
                })),
                Effect::Learn,
            ],
            targeting: vec![TargetRequirement::one(TargetSpec::SpellOrPermanent {
                min_mana_value: 1,
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn enthusiastic_study() -> CardDefinition {
    let text_box = "Target creature gets +3/+1 and gains trample until end of turn.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.enthusiastic_study",
        scryfall_id: "543c64ff-2c51-4a63-a940-dc8645717c85",
        printed: PrintedCharacteristics {
            name: "Enthusiastic Study".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::pump(3, 1),
                ),
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Trample])),
                ),
                Effect::Learn,
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn eyetwitch() -> CardDefinition {
    let text_box = "Flying\nWhen this creature dies, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.eyetwitch",
        scryfall_id: "1f4d1bb6-cb8f-4d01-9879-0b3a0585cbf4",
        printed: PrintedCharacteristics {
            name: "Eyetwitch".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Eye".into(), "Bat".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Learn],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn field_trip() -> CardDefinition {
    let text_box = "Search your library for a basic Forest card, put that card onto the battlefield tapped, then shuffle.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.field_trip",
        scryfall_id: "f235060a-eb49-4a73-bb5f-01228c3c4070",
        printed: PrintedCharacteristics {
            name: "Field Trip".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::search_library(
                    CardPredicate {
                        subtype: Some("Forest".into()),
                        ..CardPredicate::basic_land()
                    },
                    EffectValue::Fixed(1),
                    vec![Effect::return_to_battlefield(
                        ObjectSelector::Selected,
                        true,
                    )],
                ),
                Effect::Learn,
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gnarled_professor() -> CardDefinition {
    let text_box = "Trample\nWhen this creature enters, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.gnarled_professor",
        scryfall_id: "a32338e8-1f6a-49b9-bd93-26578adab6b3",
        printed: PrintedCharacteristics {
            name: "Gnarled Professor".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Treefolk".into(), "Druid".into()],
            power: Some(5),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Trample]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Learn],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn igneous_inspiration() -> CardDefinition {
    let text_box = "Igneous Inspiration deals 3 damage to any target.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.igneous_inspiration",
        scryfall_id: "5781ad7b-dc1b-4cc1-9e72-6e714b9ba1de",
        printed: PrintedCharacteristics {
            name: "Igneous Inspiration".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::deal_damage(3, ObjectSelector::Target(TargetSpec::AnyTarget)),
                Effect::Learn,
            ],
            targeting: vec![TargetRequirement::one(TargetSpec::AnyTarget)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn pop_quiz() -> CardDefinition {
    let text_box = "Draw a card.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.pop_quiz",
        scryfall_id: "d16892d8-9d10-45de-ab79-0e645c4b5588",
        printed: PrintedCharacteristics {
            name: "Pop Quiz".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                },
                Effect::Learn,
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn professor_of_symbology() -> CardDefinition {
    let text_box = "When this creature enters, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)";

    CardDefinition {
        key: "stx.professor_of_symbology",
        scryfall_id: "f427cf73-9f5e-4ef5-bc4f-29ffbfda9d57",
        printed: PrintedCharacteristics {
            name: "Professor of Symbology".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Kor".into(), "Cleric".into()],
            power: Some(2),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Learn],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn waterfall_aerialist() -> CardDefinition {
    let text_box = "Flying\nWard {2} (Whenever this creature becomes the target of a spell or ability an opponent controls, counter it unless that player pays {2}.)";

    CardDefinition {
        key: "stx.waterfall_aerialist",
        scryfall_id: "7bc6966c-18d9-4675-9594-6f0b2d11c405",
        printed: PrintedCharacteristics {
            name: "Waterfall Aerialist".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Djinn".into(), "Wizard".into()],
            power: Some(3),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying]),
            statics: vec![StaticAbility::Ward(PaymentCost::Mana(ManaCost::parse(
                "{2}",
            )))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn burrog_befuddler() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nWhen this creature enters, target creature an opponent controls gets -1/-0 until end of turn.";
    CardDefinition {
        key: "stx.burrog_befuddler",
        scryfall_id: "6040c573-cd8c-4593-8ade-d9922482035c",
        printed: PrintedCharacteristics {
            name: "Burrog Befuddler".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Frog".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flash]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::CreatureOpponentControls),
                    Modification::pump(-1, 0),
                )],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn campus_guide() -> CardDefinition {
    let text_box = "When this creature enters, you may search your library for a basic land card, reveal it, then shuffle and put that card on top.";
    CardDefinition {
        key: "stx.campus_guide",
        scryfall_id: "87a9a8a2-de81-441b-b501-418311b677f7",
        printed: PrintedCharacteristics {
            name: "Campus Guide".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Golem".into()],
            power: Some(2),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                // "then shuffle and put that card on top": the shuffle
                // precedes the placement, so the revealed card ends on top.
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::SelectCards {
                            from: ObjectSelector::Zone(ZoneType::Library),
                            predicate: CardPredicate::basic_land(),
                            min: 0,
                            max: EffectValue::Fixed(1),
                            effects: vec![
                                Effect::Reveal {
                                    what: ObjectSelector::Selected,
                                },
                                Effect::ShuffleLibrary,
                                Effect::put_in_library(ObjectSelector::Selected, LibraryEnd::Top),
                            ],
                            if_declined: vec![Effect::ShuffleLibrary],
                        }],
                        vec![],
                    ],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn teach_by_example() -> CardDefinition {
    CardDefinition {
        key: "stx.teach_by_example",
        scryfall_id: "aa7fbb9b-50a8-4d18-a667-fe965468ca16",
        printed: PrintedCharacteristics {
            name: "Teach by Example".into(),
            mana_cost: Some(ManaCost::parse("{U/R}{U/R}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::copy_next_spell_this_turn(CardPredicate::instant_or_sorcery())],
            text_box: "({U/R} can be paid with either {U} or {R}.)\nWhen you next cast an instant or sorcery spell this turn, copy that spell. You may choose new targets for the copy.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn deadly_brew() -> CardDefinition {
    let text_box = "Each player sacrifices a creature or planeswalker of their choice. If you sacrificed a permanent this way, you may return another permanent card from your graveyard to your hand.";
    CardDefinition {
        key: "stx.deadly_brew",
        scryfall_id: "87d33e48-90fc-4aac-b09a-68050bc053b5",
        printed: PrintedCharacteristics {
            name: "Deadly Brew".into(),
            mana_cost: Some(ManaCost::parse("{B}{G}")),
            types: CardTypes::new([CardType::Sorcery]),

            spell_effects: vec![
                Effect::EachPlayerSacrifices {
                    predicate: CardPredicate::creature_or_planeswalker(),
                    count: 1,
                },
                Effect::if_you_do(vec![Effect::SelectCards {
                    from: ObjectSelector::ZoneMatching {
                        zone: ZoneType::Graveyard,
                        owner: managym_state::ability::Actor::You,
                        predicate: CardPredicate::permanent(),
                        except: Some(managym_state::ability::Except::Moved),
                    },
                    predicate: CardPredicate::default(),
                    min: 0,
                    max: 1.into(),
                    effects: vec![Effect::return_to_hand(ObjectSelector::Selected)],
                    if_declined: vec![],
                }]),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
