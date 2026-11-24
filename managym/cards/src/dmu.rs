//! Dominaria United.
use managym_state::{
    ability::{
        Ability, Actor, Effect, ObjectSelector, PaymentCost, Scope, TargetSpec, TriggerCondition,
        TriggerSubject,
    },
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        StaticAbility,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        micromancer(),
        pilfer(),
        balmor_battlemage_captain(),
        magnigoth_sentry(),
        resolute_reinforcements(),
        tolarian_terror(),
    ]
}

fn tolarian_terror() -> CardDefinition {
    let text_box = "This spell costs {1} less to cast for each instant and sorcery card in your graveyard.\nWard {2} (Whenever this creature becomes the target of a spell or ability an opponent controls, counter it unless that player pays {2}.)";
    CardDefinition {
        key: "dmu.tolarian_terror",
        scryfall_id: "42f01cba-43d4-46ad-b7a5-d7631b0e1347",
        printed: PrintedCharacteristics {
            name: "Tolarian Terror".into(),
            mana_cost: Some(ManaCost::parse("{6}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Serpent".into()],
            power: Some(5),
            toughness: Some(5),
            statics: vec![
                StaticAbility::CostsLessPerGraveyardCard(CardPredicate::instant_or_sorcery()),
                StaticAbility::Ward(PaymentCost::Mana(ManaCost::parse("{2}"))),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn resolute_reinforcements() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nWhen this creature enters, create a 1/1 white Soldier creature token.";
    CardDefinition {
        key: "dmu.resolute_reinforcements",
        scryfall_id: "3e11ad33-b9d7-43ef-840a-61955683b599",
        printed: PrintedCharacteristics {
            name: "Resolute Reinforcements".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flash]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::create_tokens("token.soldier_1_1_w", 1)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn magnigoth_sentry() -> CardDefinition {
    let text_box = "Reach (This creature can block creatures with flying.)";
    CardDefinition {
        key: "dmu.magnigoth_sentry",
        scryfall_id: "d939d4bc-b7e8-4ee8-b904-68f0bff0fde1",
        printed: PrintedCharacteristics {
            name: "Magnigoth Sentry".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Treefolk".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Reach]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn balmor_battlemage_captain() -> CardDefinition {
    let text_box = "Flying\nWhenever you cast an instant or sorcery spell, creatures you control get +1/+0 and gain trample until end of turn.";
    CardDefinition {
        key: "dmu.balmor_battlemage_captain",
        scryfall_id: "959ba62e-bb3a-49ad-8b1b-e787e413e5d4",
        printed: PrintedCharacteristics {
            name: "Balmor, Battlemage Captain".into(),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Bird".into(), "Wizard".into()],
            types: CardTypes::new([CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{U}{R}")),
            power: Some(1),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(Actor::You, CardPredicate::instant_or_sorcery()),
                intervening_if: None,
                effects: vec![
                    Effect::until_end_of_turn(
                        ObjectSelector::each(Scope::YouControl, CardPredicate::creature()),
                        Modification::pump(1, 0),
                    ),
                    Effect::until_end_of_turn(
                        ObjectSelector::each(Scope::YouControl, CardPredicate::creature()),
                        Modification::AddKeywords(Keywords::of(&[Keyword::Trample])),
                    ),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn pilfer() -> CardDefinition {
    let text_box = "Target opponent reveals their hand. You choose a nonland card from it. That player discards that card.";
    CardDefinition {
        key: "dmu.pilfer",
        scryfall_id: "6d872c10-4126-4130-a74a-1331ed418ca8",
        printed: PrintedCharacteristics {
            spell_effects: vec![Effect::RevealHandDiscardChosen {
                target: TargetSpec::Opponent,
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Land],
                    ..Default::default()
                },
            }],
            name: "Pilfer".into(),
            types: CardTypes::new([CardType::Sorcery]),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn micromancer() -> CardDefinition {
    let text_box = "When this creature enters, you may search your library for an instant or sorcery card with mana value 1, reveal it, put it into your hand, then shuffle.";
    CardDefinition {
        key: "dmu.micromancer",
        scryfall_id: "b21203c8-a935-4ce0-a742-148587e32145",
        printed: PrintedCharacteristics {
            name: "Micromancer".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(3),
            toughness: Some(3),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::search_library(
                            CardPredicate {
                                min_mana_value: Some(1),
                                max_mana_value: Some(1),
                                ..CardPredicate::instant_or_sorcery()
                            },
                            1.into(),
                            vec![Effect::return_to_hand(ObjectSelector::Selected)],
                        )],
                        vec![],
                    ],
                }],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
