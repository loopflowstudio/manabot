//! Core Set 2020.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, PlayerSelector,
        TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject, TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility, StaticScope,
    },
    effect::{EffectDuration, Modification},
    mana::{Color, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        devout_decree(),
        empyrean_eagle(),
        rapacious_dragon(),
        agent_of_treachery(),
        brought_back(),
        spectral_sailor(),
    ]
}

fn spectral_sailor() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nFlying\n{3}{U}: Draw a card.";

    CardDefinition {
        key: "m20.spectral_sailor",
        scryfall_id: "67483891-36d1-46f2-8b4f-b8b7bd54bdcc",
        printed: PrintedCharacteristics {
            name: "Spectral Sailor".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Spirit".into(), "Pirate".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Flash]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{3}{U}"),
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn agent_of_treachery() -> CardDefinition {
    let text_box = "When this creature enters, gain control of target permanent.\nAt the beginning of your end step, if you control three or more permanents you don't own, draw three cards.";
    CardDefinition {
        key: "m20.agent_of_treachery",
        scryfall_id: "cc6686e6-4535-49be-b0b3-e76464656cd2",
        printed: PrintedCharacteristics {
            name: "Agent of Treachery".into(),
            mana_cost: Some(ManaCost::parse("{5}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Rogue".into()],
            power: Some(2),
            toughness: Some(3),
            abilities: vec![
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    effects: vec![Effect::GainControl {
                        player: PlayerSelector::You,
                        what: ObjectSelector::Target(TargetSpec::Permanent {
                            predicate: CardPredicate::default(),
                        }),
                        duration: EffectDuration::Indefinite,
                    }],
                },
                Ability::Triggered {
                    intervening_if: Some(Condition::ControlsUnownedAtLeast { count: 3 }),
                    condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                    effects: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(3),
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn brought_back() -> CardDefinition {
    let text_box = "Choose up to two target permanent cards in your graveyard that were put there from the battlefield this turn. Return them to the battlefield tapped.";
    let spec = TargetSpec::GraveyardCardYouOwn {
        predicate: CardPredicate {
            from_battlefield_this_turn: true,
            ..CardPredicate::permanent()
        },
    };
    CardDefinition {
        key: "m20.brought_back",
        scryfall_id: "aecdc315-95e5-450c-bedc-998eea46cf8c",
        printed: PrintedCharacteristics {
            name: "Brought Back".into(),
            mana_cost: Some(ManaCost::parse("{W}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::return_to_battlefield(
                ObjectSelector::Target(spec.clone()),
                true,
            )],
            targeting: vec![TargetRequirement::up_to(2, spec)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rapacious_dragon() -> CardDefinition {
    let text_box = "Flying\nWhen this creature enters, create two Treasure tokens. (They're artifacts with \"{T}, Sacrifice this token: Add one mana of any color.\")";
    CardDefinition {
        key: "m20.rapacious_dragon",
        scryfall_id: "2c9bf6d8-ebf6-40ff-858a-3483d19bb584",
        printed: PrintedCharacteristics {
            name: "Rapacious Dragon".into(),
            mana_cost: Some(ManaCost::parse("{4}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::create_tokens("token.treasure", 2)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn empyrean_eagle() -> CardDefinition {
    CardDefinition {
        key: "m20.empyrean_eagle",
        scryfall_id: "ac555709-c7cc-4c64-8a6f-8fe2bc149fcd",
        printed: PrintedCharacteristics {
            statics: vec![StaticAbility::continuous(
                StaticScope::OtherYouControl(CardPredicate {
                    flying: true,
                    ..CardPredicate::creature()
                }),
                None,
                vec![Modification::pump(1, 1)],
            )],
            name: "Empyrean Eagle".into(),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bird".into(), "Spirit".into()],
            mana_cost: Some(ManaCost::parse("{1}{W}{U}")),
            power: Some(2),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            text_box: "Flying\nOther creatures you control with flying get +1/+1.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn devout_decree() -> CardDefinition {
    let text_box = "Exile target creature or planeswalker that's black or red. Scry 1. (Look at the top card of your library. You may put that card on the bottom.)";
    CardDefinition {
        key: "m20.devout_decree",
        scryfall_id: "2dcde8fe-d4a4-4c6e-926e-c4a1b45045e4",
        printed: PrintedCharacteristics {
            name: "Devout Decree".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::exile(ObjectSelector::Target(TargetSpec::Permanent {
                    predicate: CardPredicate {
                        colors_any: vec![Color::Black, Color::Red],
                        ..CardPredicate::creature_or_planeswalker()
                    },
                })),
                Effect::Scry { count: 1 },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
