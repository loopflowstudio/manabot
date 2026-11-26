//! Throne of Eldraine.
use managym_state::{
    ability::{
        Ability, Condition, Effect, EffectValue, ObjectSelector, Per, TargetRelation,
        TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes,
        EntryChoiceKind, Keyword, Keywords, PrintedCharacteristics, ReplacementEffect,
        StaticAbility, StaticContinuous, StaticScope,
    },
    effect::Modification,
    mana::{Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        wishclaw_talisman(),
        run_away_together(),
        charming_prince(),
        heraldic_banner(),
        syr_alin_the_lions_claw(),
        bake_into_a_pie(),
        raging_redcap(),
        epic_downfall(),
        fabled_passage(),
        return_of_the_wildspeaker(),
        the_great_henge(),
        thrill_of_possibility(),
        wildborn_preserver(),
        scorching_dragonfire(),
    ]
}

fn epic_downfall() -> CardDefinition {
    let text_box = "Exile target creature with mana value 3 or greater.";

    CardDefinition {
        key: "eld.epic_downfall",
        scryfall_id: "63da83fe-fa59-40cb-a42e-e1b14b650bc8",
        printed: PrintedCharacteristics {
            name: "Epic Downfall".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::exile(ObjectSelector::Target(
                TargetSpec::Permanent {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        min_mana_value: Some(3),
                        ..Default::default()
                    },
                },
            ))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fabled_passage() -> CardDefinition {
    let text_box = "{T}, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield tapped, then shuffle. Then if you control four or more lands, untap that land.";

    CardDefinition {
        key: "eld.fabled_passage",
        scryfall_id: "b841bfa8-7c17-4df2-8466-780ab9a4a53a",
        printed: PrintedCharacteristics {
            name: "Fabled Passage".into(),
            types: CardTypes::new([CardType::Land]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                sacrifice_source: true,
                effects: vec![Effect::SelectCards {
                    from: ObjectSelector::Zone(ZoneType::Library),
                    predicate: CardPredicate::basic_land(),
                    max_mana_value: None,
                    min: 0,
                    max: EffectValue::Fixed(1),
                    effects: vec![
                        Effect::return_to_battlefield(ObjectSelector::Selected, true),
                        Effect::ShuffleLibrary,
                        Effect::If {
                            condition: Condition::ControlsAtLeast {
                                distinct_names: false,
                                count: 4,
                                predicate: CardPredicate {
                                    card_type: Some(CardType::Land),
                                    ..Default::default()
                                },
                            },
                            then: vec![Effect::Untap {
                                what: ObjectSelector::Moved,
                            }],
                            otherwise: vec![],
                        },
                    ],
                    if_declined: vec![Effect::ShuffleLibrary],
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn return_of_the_wildspeaker() -> CardDefinition {
    let text_box = "Choose one —\n• Draw cards equal to the greatest power among non-Human creatures you control.\n• Non-Human creatures you control get +3/+3 until end of turn.";
    let non_human = || CardPredicate {
        card_type: Some(CardType::Creature),
        not_subtypes: vec!["Human".into()],
        ..Default::default()
    };
    CardDefinition {
        key: "eld.return_of_the_wildspeaker",
        scryfall_id: "b88a4943-bd1b-4d10-9cd3-b2ab91b25c10",
        printed: PrintedCharacteristics {
            name: "Return of the Wildspeaker".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::DrawCards {
                        count: EffectValue::GreatestPowerYouControl {
                            predicate: non_human(),
                        },
                    }],
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::each_you_control(non_human()),
                        Modification::pump(3, 3),
                    )],
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn the_great_henge() -> CardDefinition {
    let text_box = "This spell costs {X} less to cast, where X is the greatest power among creatures you control.\n{T}: Add {G}{G}. You gain 2 life.\nWhenever a nontoken creature you control enters, put a +1/+1 counter on it and draw a card.";
    let creature = || CardPredicate {
        card_type: Some(CardType::Creature),
        ..Default::default()
    };
    CardDefinition {
        key: "eld.the_great_henge",
        scryfall_id: "af915ed2-1f34-43f6-85f5-2430325b720f",
        printed: PrintedCharacteristics {
            name: "The Great Henge".into(),
            mana_cost: Some(ManaCost::parse("{7}{G}{G}")),
            types: CardTypes::new([CardType::Artifact]),
            supertypes: vec!["Legendary".into()],
            statics: vec![StaticAbility::CostsLessByGreatestPower(creature())],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::add_mana(Mana::parse("GG")), Effect::gain_life(2)],
                ..Default::default()
            }],
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::you_control(CardPredicate {
                        token: Some(false),
                        ..CardPredicate::creature()
                    }),
                },
                effects: vec![
                    Effect::put_counters(
                        ObjectSelector::Triggering,
                        CounterKind::PlusOnePlusOne,
                        1,
                    ),
                    Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    },
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn wildborn_preserver() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nReach (This creature can block creatures with flying.)\nWhenever another non-Human creature you control enters, you may pay {X}. When you do, put X +1/+1 counters on this creature.";
    CardDefinition {
        key: "eld.wildborn_preserver",
        scryfall_id: "55f76830-369e-4224-9ded-7d1ce04c87e4",
        printed: PrintedCharacteristics {
            name: "Wildborn Preserver".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Archer".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Flash, Keyword::Reach]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::another_you_control(CardPredicate {
                        not_subtypes: vec!["Human".into()],
                        ..CardPredicate::creature()
                    }),
                },
                effects: vec![Effect::OptionalManaPaymentX {
                    if_paid: vec![Effect::WhenYouDo {
                        effects: vec![Effect::put_counters(
                            ObjectSelector::This,
                            CounterKind::PlusOnePlusOne,
                            EffectValue::X,
                        )],
                    }],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn thrill_of_possibility() -> CardDefinition {
    let text_box = "As an additional cost to cast this spell, discard a card.\nDraw two cards.";
    CardDefinition {
        key: "eld.thrill_of_possibility",
        scryfall_id: "c9021f85-7ab4-4a78-a398-1611fe09cd14",
        printed: PrintedCharacteristics {
            name: "Thrill of Possibility".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Instant]),
            additional_costs: vec![AdditionalCost::Discard],
            spell_effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(2),
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn raging_redcap() -> CardDefinition {
    let text_box =
        "Double strike (This creature deals both first-strike and regular combat damage.)";
    CardDefinition {
        key: "eld.raging_redcap",
        scryfall_id: "d9325398-41c3-4177-a64d-ea38cb7a8737",
        printed: PrintedCharacteristics {
            name: "Raging Redcap".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Knight".into()],
            power: Some(1),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::DoubleStrike]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bake_into_a_pie() -> CardDefinition {
    let text_box = "Destroy target creature. Create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")";
    CardDefinition {
        key: "eld.bake_into_a_pie",
        scryfall_id: "42a4d090-1bb7-4334-ab22-e2527391e79b",
        printed: PrintedCharacteristics {
            name: "Bake into a Pie".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}{B}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::destroy(ObjectSelector::Target(TargetSpec::Creature)),
                Effect::create_tokens("token.food", 1),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn syr_alin_the_lions_claw() -> CardDefinition {
    let text_box = "First strike (This creature deals combat damage before creatures without first strike.)\nWhenever Syr Alin attacks, other creatures you control get +1/+1 until end of turn.";
    CardDefinition {
        key: "eld.syr_alin_the_lions_claw",
        scryfall_id: "4cddb2d2-d813-4b83-a592-380ba4edf54f",
        printed: PrintedCharacteristics {
            name: "Syr Alin, the Lion's Claw".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Knight".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::FirstStrike]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::each_other_you_control(CardPredicate::creature()),
                    Modification::pump(1, 1),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn scorching_dragonfire() -> CardDefinition {
    let text_box = "Scorching Dragonfire deals 3 damage to target creature or planeswalker. If that creature or planeswalker would die this turn, exile it instead.";
    let target = || {
        ObjectSelector::Target(TargetSpec::Permanent {
            predicate: CardPredicate {
                card_types_any: vec![CardType::Creature, CardType::Planeswalker],
                ..Default::default()
            },
        })
    };
    CardDefinition {
        key: "eld.scorching_dragonfire",
        scryfall_id: "3b74a806-ed74-458e-8903-d3d084e9f507",
        printed: PrintedCharacteristics {
            name: "Scorching Dragonfire".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::deal_damage(3, target()),
                Effect::until_end_of_turn(target(), Modification::ExileIfWouldDie),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn heraldic_banner() -> CardDefinition {
    let text_box = "As this artifact enters, choose a color.\nCreatures you control of the chosen color get +1/+0.\n{T}: Add one mana of the chosen color.";
    CardDefinition {
        key: "eld.heraldic_banner",
        scryfall_id: "2e349af5-3f25-46d3-908e-83b2f6028b95",
        printed: PrintedCharacteristics {
            name: "Heraldic Banner".into(),
            types: CardTypes::new([CardType::Artifact]),
            mana_cost: Some(ManaCost::parse("{3}")),
            replacement_effects: vec![ReplacementEffect::AsEntersChoose {
                kind: EntryChoiceKind::Color,
                look_at_opponent_hand: false,
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::AddOneColorManaValue {
                    amount: 1.into(),
                    chosen_color: true,
                    creature_purposes: vec![],
                }],
                ..Default::default()
            }],
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::YouControl(CardPredicate {
                    chosen_color: true,
                    ..CardPredicate::creature()
                }),
                condition: None,
                characteristic_defining: false,
                modifications: vec![Modification::ModifyPt {
                    power: 1.into(),
                    toughness: 0.into(),
                }],
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn charming_prince() -> CardDefinition {
    let text_box = "When this creature enters, choose one —\n• Scry 2.\n• You gain 3 life.\n• Exile another target creature you own. Return it to the battlefield under your control at the beginning of the next end step.";
    CardDefinition {
        key: "eld.charming_prince",
        scryfall_id: "dcb94950-3f3e-4876-84f8-d5e4d9cfecee",
        printed: PrintedCharacteristics {
            name: "Charming Prince".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Noble".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                effects: vec![Effect::TriggerModes {
                    unchosen: false,
                    up_to: None,
                    modes: vec![
                        vec![Effect::Scry { count: 2 }],
                        vec![Effect::gain_life(3)],
                        vec![Effect::another_target(
                            TargetSpec::PermanentYouOwn {
                                predicate: CardPredicate::creature(),
                            },
                            |spec| {
                                vec![
                                    Effect::exile(ObjectSelector::Target(spec)),
                                    Effect::at_next_end_step(
                                        ObjectSelector::Moved,
                                        vec![Effect::return_to_battlefield(
                                            ObjectSelector::Remembered,
                                            false,
                                        )],
                                    ),
                                ]
                            },
                        )],
                    ],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn run_away_together() -> CardDefinition {
    let text_box = "Choose two target creatures controlled by different players. Return those creatures to their owners' hands.";
    CardDefinition {
        key: "eld.run_away_together",
        scryfall_id: "aeffc3c0-567c-442f-ba06-b7d9617c5789",
        printed: PrintedCharacteristics {
            name: "Run Away Together".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Instant]),
            targeting: vec![
                TargetRequirement::one(TargetSpec::Creature),
                TargetRequirement {
                    relation: Some(TargetRelation::DifferentControllerFrom(0)),
                    ..TargetRequirement::one(TargetSpec::Creature)
                },
            ],
            spell_effects: vec![Effect::Simultaneously {
                effects: vec![
                    Effect::return_to_hand(ObjectSelector::ClauseTarget(0)),
                    Effect::return_to_hand(ObjectSelector::ClauseTarget(1)),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn wishclaw_talisman() -> CardDefinition {
    use managym_state::{ability::PlayerSelector, card::EntryScope, effect::EffectDuration};
    CardDefinition {
key: "eld.wishclaw_talisman",
scryfall_id: "07c17b01-ee5d-491a-8403-b3f819b778c4",
printed: PrintedCharacteristics {
name: "Wishclaw Talisman".into(),
mana_cost: Some(ManaCost::parse("{1}{B}")),
types: CardTypes::new([CardType::Artifact]),
replacement_effects: vec![ReplacementEffect::EntersWithCounters { on: EntryScope::This, kind: CounterKind::Wish, count: 3.into(), condition: None }],
activated_abilities: vec![ActivatedAbilityDefinition {
own_turn_only: true, mana_cost: ManaCost::parse("{1}"), tap_source: true, remove_counters: Some((CounterKind::Wish, 1)),
effects: vec![Effect::search_for_card(ZoneType::Hand), Effect::GainControl {what: ObjectSelector::This, player: PlayerSelector::EachOpponent, duration: EffectDuration::Indefinite}], ..Default::default()
}],
text_box: "This artifact enters with three wish counters on it.\n{1}, {T}, Remove a wish counter from this artifact: Search your library for a card, put it into your hand, then shuffle. An opponent gains control of this artifact. Activate only during your turn.".into(),
..Default::default()
},
..Default::default()
}
}
