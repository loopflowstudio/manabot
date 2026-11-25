//! Core Set 2020.
use managym_state::ability::{
    Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, Per, PlayerSelector,
    SpellTargetFilter, TargetRelation, TargetRequirement, TargetSpec, TriggerCondition,
    TriggerSubject, TurnStep,
};
use managym_state::card::{
    ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
    ManaAbility, PrintedCharacteristics, StaticAbility, StaticScope,
};
use managym_state::effect::{EffectDuration, Modification};
use managym_state::mana::{Color, Mana, ManaCost};
use managym_state::permanent::CounterKind;
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        drakuseth_maw_of_flames(),
        cryptic_caves(),
        agent_of_treachery(),
        brought_back(),
        spectral_sailor(),
        brineborn_cutthroat(),
        gnarlback_rhino(),
        twinblade_paladin(),
        goblin_smuggler(),
        devout_decree(),
        empyrean_eagle(),
        rapacious_dragon(),
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
                ObjectSelector::Targets(spec.clone()),
                true,
            )],
            targeting: vec![TargetRequirement::up_to(2, spec)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn brineborn_cutthroat() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nWhenever you cast a spell during an opponent's turn, put a +1/+1 counter on this creature.";
    CardDefinition {
        key: "m20.brineborn_cutthroat",
        scryfall_id: "0857765f-afd7-418a-a93b-c0bd1b1f037e",
        printed: PrintedCharacteristics {
            name: "Brineborn Cutthroat".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Merfolk".into(), "Pirate".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flash]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::ActiveIf {
                    active_if: Condition::not(Condition::YourTurn),
                    condition: Box::new(TriggerCondition::casts(
                        Actor::You,
                        CardPredicate::default(),
                    )),
                },
                intervening_if: None,
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gnarlback_rhino() -> CardDefinition {
    let text_box = "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nWhenever you cast a spell that targets this creature, draw a card.";
    CardDefinition {
        key: "m20.gnarlback_rhino",
        scryfall_id: "68a69558-aca0-413d-9762-2fa115b44abd",
        printed: PrintedCharacteristics {
            name: "Gnarlback Rhino".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Rhino".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Trample]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Casts {
                    who: Actor::You,
                    predicate: CardPredicate::default(),
                    nth: None,
                    targets: SpellTargetFilter::TargetsThis,
                },
                intervening_if: None,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn cryptic_caves() -> CardDefinition {
    let text_box = "{T}: Add {C}.\n{1}, {T}, Sacrifice this land: Draw a card. Activate only if you control five or more lands.";
    CardDefinition {
        key: "m20.cryptic_caves",
        scryfall_id: "fde9e9cb-68ab-4856-8ad6-30f66666dd93",
        printed: PrintedCharacteristics {
            name: "Cryptic Caves".into(),
            types: CardTypes::new([CardType::Land]),
            mana_abilities: vec![ManaAbility {
                mana: Mana::parse("C"),
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}"),
                tap_source: true,
                sacrifice_source: true,
                condition: Some(Condition::ControlsAtLeast {
                    count: 5,
                    predicate: CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    },
                    distinct_names: false,
                }),
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

fn goblin_smuggler() -> CardDefinition {
    let text_box = "Haste (This creature can attack and {T} as soon as it comes under your control.)\n{T}: Another target creature with power 2 or less can't be blocked this turn.";
    let target = TargetSpec::Permanent {
        predicate: CardPredicate {
            card_type: Some(CardType::Creature),
            max_power: Some(2),
            ..Default::default()
        },
    };
    CardDefinition {
        key: "m20.goblin_smuggler",
        scryfall_id: "95dc1a65-271c-455a-ae0c-f652444a53ac",
        printed: PrintedCharacteristics {
            name: "Goblin Smuggler".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Rogue".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Haste]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::another_target(target, |target| {
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::Target(target),
                        Modification::CantBeBlocked,
                    )]
                })],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn twinblade_paladin() -> CardDefinition {
    let text_box = "Whenever you gain life, put a +1/+1 counter on this creature.\nAs long as you have 25 or more life, this creature has double strike. (It deals both first-strike and regular combat damage.)";
    CardDefinition {
        key: "m20.twinblade_paladin",
        scryfall_id: "6397d426-00e0-44da-b23c-44ccea65f5aa",
        printed: PrintedCharacteristics {
            name: "Twinblade Paladin".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Knight".into()],
            power: Some(3),
            toughness: Some(3),
            abilities: vec![Ability::when_you_gain_life(Effect::put_counters(
                ObjectSelector::This,
                CounterKind::PlusOnePlusOne,
                1,
            ))],
            statics: vec![StaticAbility::continuous(
                StaticScope::This,
                Some(Condition::LifeAtLeast { amount: 25 }),
                vec![Modification::AddKeywords(Keywords::of(&[
                    Keyword::DoubleStrike,
                ]))],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn drakuseth_maw_of_flames() -> CardDefinition {
    let text_box = "Flying\nWhenever Drakuseth attacks, Drakuseth deals 4 damage to any target and 3 damage to each of up to two other targets.";
    CardDefinition {
        key: "m20.drakuseth_maw_of_flames",
        scryfall_id: "d09af78f-efde-4107-8406-cb12fd11c686",
        printed: PrintedCharacteristics {
            name: "Drakuseth, Maw of Flames".into(),
            mana_cost: Some(ManaCost::parse("{4}{R}{R}{R}")),
            supertypes: vec!["Legendary".into()],
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(7),
            toughness: Some(7),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                effects: vec![
                    Effect::TargetClause {
                        requirement: TargetRequirement::one(TargetSpec::AnyTarget),
                        effects: vec![],
                    },
                    Effect::TargetClause {
                        requirement: TargetRequirement {
                            relation: Some(TargetRelation::DistinctFromEarlier),
                            ..TargetRequirement::up_to(2, TargetSpec::AnyTarget)
                        },
                        effects: vec![Effect::Simultaneously {
                            effects: vec![
                                Effect::deal_damage(4, ObjectSelector::ClauseTarget(0)),
                                Effect::deal_damage(3, ObjectSelector::ClauseTarget(1)),
                            ],
                        }],
                    },
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
