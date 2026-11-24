//! Foundations cards, expressed in the engine's shared rules vocabulary.
use managym_state::{
    ability::{
        Ability, Actor, Aggregate, AttackSize, CompareOp, Condition, DamageKind,
        DamageRecipientFilter, Effect, EffectValue, ObjectProperty, ObjectSelector, PaymentCost,
        Per, PlayWindow, PlayerSelector, Scope, TargetRequirement, TargetSpec, TriggerCondition,
        TriggerSubject, TurnStep,
    },
    card::{
        ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes, Keyword,
        Keywords, PrintedCharacteristics, ReplacementEffect, Restriction, SacrificeCost,
        SpellCostReduction, StaticAbility, StaticRestriction, StaticScope, TargetCostReduction,
    },
    effect::Modification,
    mana::{Color, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        tragic_banshee(),
        vengeful_bloodwitch(),
        needletooth_pack(),
        slumbering_cerberus(),
        wardens_of_the_cycle(),
        courageous_goblin(),
        bulk_up(),
        ashroot_animist(),
        alesha_who_laughs_at_fate(),
        searslicer_goblin(),
        gorehorn_raider(),
        goblin_boarders(),
        battlesong_berserker(),
        midnight_snack(),
        skyship_buccaneer(),
        armasaur_guide(),
        blasphemous_edict(),
        cackling_prowler(),
        cat_collector(),
        chart_a_course(),
        claws_out(),
        drake_hatcher(),
        erudite_wizard(),
        felling_blow(),
        fiendish_panda(),
        high_society_hunter(),
        homunculus_horde(),
        hungry_ghoul(),
        incinerating_blast(),
        herald_of_eternal_dawn(),
        kellan_planar_trailblazer(),
        linden_the_steadfast_queen(),
        luminous_rebuke(),
        lunar_insight(),
        mischievous_mystic(),
        leyline_axe(),
        sire_of_seven_deaths(),
        vanguard_seraph(),
        bloodthirsty_conqueror(),
        exemplar_of_light(),
        scrawling_crawler(),
        firespitter_whelp(),
        archmage_of_runes(),
        goblin_negotiation(),
        niv_mizzet_visionary(),
    ]
}

fn vanguard_seraph() -> CardDefinition {
    let text_box = "Flying\nWhenever you gain life for the first time each turn, surveil 1. (Look at the top card of your library. You may put it into your graveyard.)";
    CardDefinition {
        key: "fdn.vanguard_seraph",
        scryfall_id: "4329c861-fc16-4a96-9c03-25af6ac2adc8",
        printed: PrintedCharacteristics {
            name: "Vanguard Seraph".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Angel".into(), "Warrior".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::GainsLife {
                    who: Actor::You,
                    nth: Some(1),
                },
                intervening_if: None,
                effects: vec![Effect::Surveil { count: 1 }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bloodthirsty_conqueror() -> CardDefinition {
    let text_box = "Flying, deathtouch\nWhenever an opponent loses life, you gain that much life. (Damage causes loss of life.)";
    CardDefinition {
        key: "fdn.bloodthirsty_conqueror",
        scryfall_id: "ce860ed4-a5bd-4347-9eab-dd716ea84db1",
        printed: PrintedCharacteristics {
            name: "Bloodthirsty Conqueror".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Knight".into()],
            power: Some(5),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Deathtouch]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::LosesLife {
                    who: Actor::Opponent,
                },
                intervening_if: None,
                effects: vec![Effect::gain_life(EffectValue::TriggeringAmount)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn exemplar_of_light() -> CardDefinition {
    let text_box = "Flying\nWhenever you gain life, put a +1/+1 counter on this creature.\nWhenever you put one or more +1/+1 counters on this creature, draw a card. This ability triggers only once each turn.";
    CardDefinition {
        key: "fdn.exemplar_of_light",
        scryfall_id: "920c8fc5-fdd2-446a-a676-5c363f96928f",
        printed: PrintedCharacteristics {
            name: "Exemplar of Light".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Angel".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::GainsLife {
                        who: Actor::You,
                        nth: None,
                    },
                    intervening_if: None,
                    effects: vec![Effect::put_counters(
                        ObjectSelector::This,
                        CounterKind::PlusOnePlusOne,
                        1,
                    )],
                },
                Ability::Triggered {
                    condition: TriggerCondition::OnceEachTurn {
                        condition: Box::new(TriggerCondition::PutsCountersOn {
                            who: Actor::You,
                            kind: CounterKind::PlusOnePlusOne,
                            subject: TriggerSubject::This,
                        }),
                    },
                    intervening_if: None,
                    effects: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sire_of_seven_deaths() -> CardDefinition {
    let text_box = "Reach, first strike\nVigilance, menace\nTrample, lifelink\nWard—Pay 7 life.";
    CardDefinition {
        key: "fdn.sire_of_seven_deaths",
        scryfall_id: "8d8432a7-1c8a-4cfb-947c-ecf9791063eb",
        printed: PrintedCharacteristics {
            name: "Sire of Seven Deaths".into(),
            mana_cost: Some(ManaCost::parse("{7}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Eldrazi".into()],
            power: Some(7),
            toughness: Some(7),
            keywords: Keywords::of(&[
                Keyword::Reach,
                Keyword::FirstStrike,
                Keyword::Vigilance,
                Keyword::Menace,
                Keyword::Trample,
                Keyword::Lifelink,
            ]),
            statics: vec![StaticAbility::Ward(PaymentCost::Life(7))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn hungry_ghoul() -> CardDefinition {
    let text_box = "{1}, Sacrifice another creature: Put a +1/+1 counter on this creature.";
    CardDefinition {
        key: "fdn.hungry_ghoul",
        scryfall_id: "790f9433-7565-4f7f-88e8-8af762ea0296",
        printed: PrintedCharacteristics {
            name: "Hungry Ghoul".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Zombie".into()],
            power: Some(2),
            toughness: Some(2),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}"),
                sacrifice: Some(SacrificeCost {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        ..Default::default()
                    },
                    another: true,
                    or_mana: None,
                }),
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn kellan_planar_trailblazer() -> CardDefinition {
    let text_box = "{1}{R}: If Kellan is a Scout, it becomes a Human Faerie Detective and gains \"Whenever Kellan deals combat damage to a player, exile the top card of your library. You may play that card this turn.\"\n{2}{R}: If Kellan is a Detective, it becomes a 3/2 Human Faerie Rogue and gains double strike.";
    CardDefinition {
        key: "fdn.kellan_planar_trailblazer",
        scryfall_id: "f46a9329-7b91-441d-8653-50c1152c9120",
        printed: PrintedCharacteristics {
            name: "Kellan, Planar Trailblazer".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Faerie".into(), "Scout".into()],
            power: Some(2),
            toughness: Some(1),
            activated_abilities: vec![
                ActivatedAbilityDefinition {
                    mana_cost: ManaCost::parse("{1}{R}"),
                    effects: vec![Effect::If {
                        condition: Condition::SourceMatches(CardPredicate::subtype("Scout")),
                        then: vec![
                            Effect::indefinitely(
                                ObjectSelector::This,
                                Modification::SetCreatureTypes(vec![
                                    "Human".into(),
                                    "Faerie".into(),
                                    "Detective".into(),
                                ]),
                            ),
                            Effect::indefinitely(
                                ObjectSelector::This,
                                Modification::GrantTriggered(Box::new(Ability::Triggered {
                                    condition: TriggerCondition::DealsCombatDamage {
                                        subject: TriggerSubject::This,
                                        to: Actor::AnyPlayer,
                                        per: Per::Object,
                                    },
                                    intervening_if: None,
                                    effects: vec![Effect::ExileTopCardsToPlay {
                                        count: 1,
                                        until: PlayWindow::ThisTurn,
                                    }],
                                })),
                            ),
                        ],
                        otherwise: vec![],
                    }],
                    ..Default::default()
                },
                ActivatedAbilityDefinition {
                    mana_cost: ManaCost::parse("{2}{R}"),
                    effects: vec![Effect::If {
                        condition: Condition::SourceMatches(CardPredicate::subtype("Detective")),
                        then: vec![
                            Effect::indefinitely(
                                ObjectSelector::This,
                                Modification::SetCreatureTypes(vec![
                                    "Human".into(),
                                    "Faerie".into(),
                                    "Rogue".into(),
                                ]),
                            ),
                            Effect::indefinitely(
                                ObjectSelector::This,
                                Modification::base_pt(Some(3), Some(2)),
                            ),
                            Effect::indefinitely(
                                ObjectSelector::This,
                                Modification::AddKeywords(Keywords::of(&[Keyword::DoubleStrike])),
                            ),
                        ],
                        otherwise: vec![],
                    }],
                    ..Default::default()
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn drake_hatcher() -> CardDefinition {
    let text_box = "Vigilance, prowess\nWhenever this creature deals combat damage to a player, put that many incubation counters on it.\nRemove three incubation counters from this creature: Create a 2/2 blue Drake creature token with flying.";
    CardDefinition {
        key: "fdn.drake_hatcher",
        scryfall_id: "bcaf4196-6bf3-47fa-b5c7-0e77f45cf820",
        printed: PrintedCharacteristics {
            name: "Drake Hatcher".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(1),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Vigilance]),
            abilities: vec![
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::casts(
                        Actor::You,
                        CardPredicate {
                            not_card_types: vec![CardType::Creature],
                            ..Default::default()
                        },
                    ),
                    effects: vec![Effect::until_end_of_turn(
                        ObjectSelector::This,
                        Modification::pump(1, 1),
                    )],
                },
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::DealsCombatDamage {
                        subject: TriggerSubject::This,
                        to: Actor::AnyPlayer,
                        per: Per::Object,
                    },
                    effects: vec![Effect::put_counters(
                        ObjectSelector::This,
                        CounterKind::Incubation,
                        EffectValue::TriggeringAmount,
                    )],
                },
            ],
            activated_abilities: vec![ActivatedAbilityDefinition {
                remove_counters: Some((CounterKind::Incubation, 3)),
                effects: vec![Effect::create_tokens("token.drake", 1)],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn felling_blow() -> CardDefinition {
    CardDefinition {
        key: "fdn.felling_blow",
        scryfall_id: "96948ae3-b15d-4d6d-aa73-9f52084cd903",
        printed: PrintedCharacteristics {
            name: "Felling Blow".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            targeting: vec![
                TargetRequirement::one(TargetSpec::CreatureYouControl),
                TargetRequirement::one(TargetSpec::CreatureOpponentControls),
            ],
            spell_effects: vec![
                Effect::put_counters(ObjectSelector::Target(TargetSpec::CreatureYouControl), CounterKind::PlusOnePlusOne, 1),
                Effect::TargetCreaturesDealPowerDamageToLastTarget,
            ],
            text_box: "Put a +1/+1 counter on target creature you control. Then that creature deals damage equal to its power to target creature an opponent controls.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn lunar_insight() -> CardDefinition {
    CardDefinition {
        key: "fdn.lunar_insight",
        scryfall_id: "a9a159f6-fecf-4bdd-b2f8-a9665a5cc32d",
        printed: PrintedCharacteristics {
            name: "Lunar Insight".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::DrawCards {
                count: EffectValue::DistinctManaValues {
                    of: Box::new(ObjectSelector::each(
                        Scope::YouControl,
                        CardPredicate {
                            not_card_types: vec![CardType::Land],
                            ..Default::default()
                        },
                    )),
                },
            }],
            text_box:
                "Draw a card for each different mana value among nonland permanents you control."
                    .into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn luminous_rebuke() -> CardDefinition {
    let text_box = "This spell costs {3} less to cast if it targets a tapped creature.\nDestroy target creature.";
    CardDefinition {
        key: "fdn.luminous_rebuke",
        scryfall_id: "621839e1-2756-4cdc-a25c-5f76ea98dd87",
        printed: PrintedCharacteristics {
            name: "Luminous Rebuke".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::destroy(ObjectSelector::Target(
                TargetSpec::Creature,
            ))],
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            statics: vec![StaticAbility::CostsLessIfTargets(TargetCostReduction {
                predicate: CardPredicate {
                    tapped: true,
                    card_type: Some(CardType::Creature),
                    ..Default::default()
                },
                generic: 3,
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn blasphemous_edict() -> CardDefinition {
    let text_box = "You may pay {B} rather than pay this spell's mana cost if there are thirteen or more creatures on the battlefield.\nEach player sacrifices thirteen creatures of their choice.";
    CardDefinition {
        key: "fdn.blasphemous_edict",
        scryfall_id: "11040ecd-3153-4029-b42b-1441bc51ec34",
        printed: PrintedCharacteristics {
            name: "Blasphemous Edict".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            additional_costs: vec![AdditionalCost::AlternativeMana {
                cost: ManaCost::parse("{B}"),
                condition: Condition::BattlefieldAtLeast {
                    count: 13,
                    predicate: CardPredicate::creature(),
                },
            }],
            spell_effects: vec![Effect::EachPlayerSacrificesCreatures { count: 13 }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn claws_out() -> CardDefinition {
    let text_box = "Affinity for Cats (This spell costs {1} less to cast for each Cat you control.)\nCreatures you control get +2/+2 until end of turn.";
    CardDefinition {
        key: "fdn.claws_out",
        scryfall_id: "4396049c-b976-4b7f-8ecd-564e24ebd631",
        printed: PrintedCharacteristics {
            name: "Claws Out".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}{W}")),
            types: CardTypes::new([CardType::Instant]),
            statics: vec![StaticAbility::CostsLessPerPermanent(
                CardPredicate::subtype("Cat"),
            )],
            spell_effects: vec![Effect::until_end_of_turn(
                ObjectSelector::each_you_control(CardPredicate::creature()),
                Modification::pump(2, 2),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn cat_collector() -> CardDefinition {
    let text_box = "When this creature enters, create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")\nWhenever you gain life for the first time during each of your turns, create a 1/1 white Cat creature token.";
    CardDefinition {
        key: "fdn.cat_collector",
        scryfall_id: "526fe356-bff1-4211-9e88-bf913ac76b1d",
        printed: PrintedCharacteristics {
            name: "Cat Collector".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Citizen".into()],
            power: Some(3),
            toughness: Some(2),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: None,
                    effects: vec![Effect::create_tokens("token.food", 1)],
                },
                Ability::Triggered {
                    condition: TriggerCondition::GainsLife {
                        who: Actor::You,
                        nth: Some(1),
                    },
                    intervening_if: Some(Condition::YourTurn),
                    effects: vec![Effect::create_tokens("token.cat", 1)],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fiendish_panda() -> CardDefinition {
    let text_box = "Whenever you gain life, put a +1/+1 counter on this creature.\nWhen this creature dies, return another target non-Bear creature card with mana value less than or equal to this creature's power from your graveyard to the battlefield.";
    let spec = TargetSpec::GraveyardCardYouOwnWithinSourcePower {
        predicate: CardPredicate {
            not_subtypes: vec!["Bear".into()],
            ..CardPredicate::creature()
        },
    };
    CardDefinition {
        key: "fdn.fiendish_panda",
        scryfall_id: "4e434d74-cad0-45f5-bc8d-f34aa5e1d879",
        printed: PrintedCharacteristics {
            name: "Fiendish Panda".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bear".into(), "Demon".into()],
            power: Some(3),
            toughness: Some(2),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::GainsLife {
                        who: Actor::You,
                        nth: None,
                    },
                    intervening_if: None,
                    effects: vec![Effect::put_counters(
                        ObjectSelector::This,
                        CounterKind::PlusOnePlusOne,
                        1,
                    )],
                },
                Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: None,
                    effects: vec![Effect::TargetClause {
                        requirement: TargetRequirement::another(spec.clone()),
                        effects: vec![Effect::return_to_battlefield(
                            ObjectSelector::Target(spec),
                            false,
                        )],
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn cackling_prowler() -> CardDefinition {
    CardDefinition {
        key: "fdn.cackling_prowler",
        scryfall_id: "1bd8e971-c075-4203-8d83-c28f22d4f9b9",
        printed: PrintedCharacteristics {
            name: "Cackling Prowler".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Hyena".into(), "Rogue".into()],
            power: Some(4),
            toughness: Some(3),
            statics: vec![StaticAbility::Ward(PaymentCost::Mana(ManaCost::parse("{2}")))],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                intervening_if: Some(Condition::CreatureDiedThisTurn),
                effects: vec![Effect::put_counters(ObjectSelector::This, CounterKind::PlusOnePlusOne, 1)],
            }],
            text_box: "Ward {2} (Whenever this creature becomes the target of a spell or ability an opponent controls, counter it unless that player pays {2}.)\nMorbid — At the beginning of your end step, if a creature died this turn, put a +1/+1 counter on this creature.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn high_society_hunter() -> CardDefinition {
    let text_box = "Flying\nWhenever this creature attacks, you may sacrifice another creature. If you do, put a +1/+1 counter on this creature.\nWhenever another nontoken creature dies, draw a card.";
    CardDefinition {
        key: "fdn.high_society_hunter",
        scryfall_id: "51da4a4b-ea12-4169-a7cf-eb4427f13e84",
        printed: PrintedCharacteristics {
            name: "High-Society Hunter".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Noble".into()],
            power: Some(5),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            text_box: text_box.into(),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                    intervening_if: None,
                    effects: vec![
                        Effect::may_sacrifice(CardPredicate::creature(), true),
                        Effect::if_you_do(vec![Effect::put_counters(
                            ObjectSelector::This,
                            CounterKind::PlusOnePlusOne,
                            1,
                        )]),
                    ],
                },
                Ability::Triggered {
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::another(CardPredicate {
                            token: Some(false),
                            ..CardPredicate::creature()
                        }),
                    },
                    intervening_if: None,
                    effects: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                },
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn leyline_axe() -> CardDefinition {
    use managym_state::card::{StaticKeywordGrant, StaticPtBuff, StaticScope};
    CardDefinition {
        key: "fdn.leyline_axe",
        scryfall_id: "b9c03336-a321-4c06-94d1-809f328fabd8",
        printed: PrintedCharacteristics {
            name: "Leyline Axe".into(),
            mana_cost: Some(ManaCost::parse("{4}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            activated_abilities: vec![ActivatedAbilityDefinition {
                sorcery_timing: true,
                mana_cost: ManaCost::parse("{3}"),
                effects: vec![Effect::AttachSourceTo {
                    what: ObjectSelector::Target(TargetSpec::CreatureYouControl),
                }],
                ..Default::default()
            }],
            statics: vec![StaticAbility::BeginOnBattlefield,
                StaticAbility::PtBuff(StaticPtBuff {
                    scope: StaticScope::Attached,
                    condition: None,
                    power: 1,
                    toughness: 1,
                    per_creature_type_max: None,
                    per_color_among: None,
                    per_graveyard_matching: None,
                }),
                StaticAbility::KeywordGrant(StaticKeywordGrant {
                    scope: StaticScope::Attached,
                    condition: None,
                    keywords: Keywords::of(&[Keyword::DoubleStrike, Keyword::Trample]),
                }),
            ],
            text_box: "If this card is in your opening hand, you may begin the game with it on the battlefield.\nEquipped creature gets +1/+1 and has double strike and trample.\nEquip {3}".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn incinerating_blast() -> CardDefinition {
    CardDefinition {
        key: "fdn.incinerating_blast",
        scryfall_id: "d58e20ab-c5ca-4295-884d-78efdaa83243",
        printed: PrintedCharacteristics {
            name: "Incinerating Blast".into(),
            mana_cost: Some(ManaCost::parse("{4}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![
                Effect::deal_damage(6, ObjectSelector::Target(TargetSpec::Creature)),
                Effect::discard_draw(1, 1),
            ],
            text_box: "Incinerating Blast deals 6 damage to target creature.\nYou may discard a card. If you do, draw a card.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn on_second_draw(effect: Effect) -> Ability {
    Ability::Triggered {
        condition: TriggerCondition::Draws {
            who: Actor::You,
            nth: Some(2),
        },
        intervening_if: None,
        effects: vec![effect],
    }
}

fn erudite_wizard() -> CardDefinition {
    let text_box =
        "Whenever you draw your second card each turn, put a +1/+1 counter on this creature.";
    CardDefinition {
        key: "fdn.erudite_wizard",
        scryfall_id: "9273c417-0fcd-4273-b24e-afff76336d0c",
        printed: PrintedCharacteristics {
            name: "Erudite Wizard".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(3),
            abilities: vec![on_second_draw(Effect::put_counters(
                ObjectSelector::This,
                CounterKind::PlusOnePlusOne,
                1,
            ))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn homunculus_horde() -> CardDefinition {
    let text_box = "Whenever you draw your second card each turn, create a token that's a copy of this creature.";
    CardDefinition {
        key: "fdn.homunculus_horde",
        scryfall_id: "470c4d03-340a-4e0e-a59f-f19d05497785",
        printed: PrintedCharacteristics {
            name: "Homunculus Horde".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Homunculus".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![on_second_draw(Effect::CreateTokenCopy {
                of: ObjectSelector::This,
                exceptions: vec![],
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mischievous_mystic() -> CardDefinition {
    let text_box = "Flying\nWhenever you draw your second card each turn, create a 1/1 blue Faerie creature token with flying.";
    CardDefinition {
        key: "fdn.mischievous_mystic",
        scryfall_id: "20d89cec-528b-4b2a-87db-e11ce0000622",
        printed: PrintedCharacteristics {
            name: "Mischievous Mystic".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![on_second_draw(Effect::create_tokens("token.faerie", 1))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn chart_a_course() -> CardDefinition {
    CardDefinition {
        key: "fdn.chart_a_course",
        scryfall_id: "7599d459-fe71-48f4-8c65-0f519bb63a68",
        printed: PrintedCharacteristics {
            name: "Chart a Course".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::DrawCards {
                    count: EffectValue::Fixed(2),
                },
                Effect::If {
                    condition: Condition::AttackedThisTurn,
                    then: vec![],
                    otherwise: vec![Effect::discard()],
                },
            ],
            text_box: "Draw two cards. Then discard a card unless you attacked this turn.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn linden_the_steadfast_queen() -> CardDefinition {
    let text_box = "Vigilance (Attacking doesn't cause this creature to tap.)\nWhenever a white creature you control attacks, you gain 1 life.";
    CardDefinition {
        key: "fdn.linden_the_steadfast_queen",
        scryfall_id: "e0bb07cd-9b74-4ba7-8089-27a565d74197",
        printed: PrintedCharacteristics {
            name: "Linden, the Steadfast Queen".into(),
            mana_cost: Some(ManaCost::parse("{W}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Noble".into()],
            power: Some(3),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Vigilance]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(
                    TriggerSubject::you_control(CardPredicate {
                        colors_any: vec![Color::White],
                        ..CardPredicate::creature()
                    }),
                    Per::Object,
                ),
                intervening_if: None,
                effects: vec![Effect::gain_life(1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn herald_of_eternal_dawn() -> CardDefinition {
    CardDefinition {
        key: "fdn.herald_of_eternal_dawn",
        scryfall_id: "c9fdfebf-98e0-4718-bac3-6eee1cd0623d",
        printed: PrintedCharacteristics {
            name: "Herald of Eternal Dawn".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}{W}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Angel".into()],
            power: Some(6),
            toughness: Some(6),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Flash]),
            statics: vec![StaticAbility::CantLoseAndOpponentsCantWin],
            text_box: "Flash (You may cast this spell any time you could cast an instant.)\nFlying\nYou can't lose the game and your opponents can't win the game.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn armasaur_guide() -> CardDefinition {
    let text_box = "Vigilance (Attacking doesn't cause this creature to tap.)\nWhenever you attack with three or more creatures, put a +1/+1 counter on target creature you control.";
    CardDefinition {
        key: "fdn.armasaur_guide",
        scryfall_id: "c80fc380-0499-4499-8a60-c43844c02c9b",
        printed: PrintedCharacteristics {
            name: "Armasaur Guide".into(),
            mana_cost: Some(ManaCost::parse("{4}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dinosaur".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Vigilance]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::you_control(CardPredicate::creature()),
                    per: Per::Batch,
                    attackers: AttackSize::AtLeast(3),
                    nth: None,
                },
                intervening_if: None,
                effects: vec![Effect::put_counters(
                    ObjectSelector::Target(TargetSpec::CreatureYouControl),
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

fn skyship_buccaneer() -> CardDefinition {
    let text_box =
        "Flying\nRaid — When this creature enters, if you attacked this turn, draw a card.";
    CardDefinition {
        key: "fdn.skyship_buccaneer",
        scryfall_id: "62958fc3-55dc-4b97-a070-490d6ed27820",
        printed: PrintedCharacteristics {
            name: "Skyship Buccaneer".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Pirate".into()],
            power: Some(4),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: Some(Condition::AttackedThisTurn),
                effects: vec![Effect::DrawCards { count: 1.into() }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn midnight_snack() -> CardDefinition {
    let text_box = "Raid — At the beginning of your end step, if you attacked this turn, create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")\n{2}{B}, Sacrifice this enchantment: Target opponent loses X life, where X is the amount of life you gained this turn.";
    CardDefinition {
        key: "fdn.midnight_snack",
        scryfall_id: "c9b7543f-2a45-4db6-b560-d15507a58c91",
        printed: PrintedCharacteristics {
            name: "Midnight Snack".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                intervening_if: Some(Condition::AttackedThisTurn),
                effects: vec![Effect::create_tokens("token.food", 1)],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{2}{B}"),
                sacrifice_source: true,
                effects: vec![Effect::LoseLife {
                    player: PlayerSelector::Target(Box::new(TargetSpec::Opponent)),
                    amount: EffectValue::LifeGainedThisTurn(PlayerSelector::You),
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn battlesong_berserker() -> CardDefinition {
    let text_box = "Whenever you attack, target creature you control gets +1/+0 and gains menace until end of turn. (It can't be blocked except by two or more creatures.)";
    CardDefinition {
        key: "fdn.battlesong_berserker",
        scryfall_id: "a1f8b199-5d62-485f-b1c3-b30aa550595b",
        printed: PrintedCharacteristics {
            name: "Battlesong Berserker".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Berserker".into()],
            power: Some(3),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(
                    TriggerSubject::you_control(CardPredicate::creature()),
                    Per::Batch,
                ),
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::CreatureYouControl),
                    effects: vec![
                        Effect::until_end_of_turn(
                            ObjectSelector::Target(TargetSpec::CreatureYouControl),
                            Modification::pump(1, 0),
                        ),
                        Effect::until_end_of_turn(
                            ObjectSelector::Target(TargetSpec::CreatureYouControl),
                            Modification::AddKeywords(Keywords::of(&[Keyword::Menace])),
                        ),
                    ],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn goblin_boarders() -> CardDefinition {
    let text_box =
        "Raid — This creature enters with a +1/+1 counter on it if you attacked this turn.";
    CardDefinition {
        key: "fdn.goblin_boarders",
        scryfall_id: "4409a063-bf2a-4a49-803e-3ce6bd474353",
        printed: PrintedCharacteristics {
            name: "Goblin Boarders".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Pirate".into()],
            power: Some(3),
            toughness: Some(2),
            replacement_effects: vec![ReplacementEffect::EntersWithPlusOneCounters {
                count: 1,
                condition: Some(Condition::AttackedThisTurn),
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gorehorn_raider() -> CardDefinition {
    let text_box = "Raid — When this creature enters, if you attacked this turn, this creature deals 2 damage to any target.";
    CardDefinition {
        key: "fdn.gorehorn_raider",
        scryfall_id: "78ce6c40-3452-4aa0-a45b-dbfd70f8d220",
        printed: PrintedCharacteristics {
            name: "Gorehorn Raider".into(),
            mana_cost: Some(ManaCost::parse("{4}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Minotaur".into(), "Pirate".into()],
            power: Some(4),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: Some(Condition::AttackedThisTurn),
                effects: vec![Effect::deal_damage(
                    2,
                    ObjectSelector::Target(TargetSpec::AnyTarget),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn searslicer_goblin() -> CardDefinition {
    let text_box = "Raid — At the beginning of your end step, if you attacked this turn, create a 1/1 red Goblin creature token.";
    CardDefinition {
        key: "fdn.searslicer_goblin",
        scryfall_id: "94ad0b97-a318-4e76-ac79-b3e83417c333",
        printed: PrintedCharacteristics {
            name: "Searslicer Goblin".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Warrior".into()],
            power: Some(2),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                intervening_if: Some(Condition::AttackedThisTurn),
                effects: vec![Effect::create_tokens("token.goblin", 1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn alesha_who_laughs_at_fate() -> CardDefinition {
    let text_box = "First strike\nWhenever Alesha attacks, put a +1/+1 counter on it.\nRaid — At the beginning of your end step, if you attacked this turn, return target creature card with mana value less than or equal to Alesha's power from your graveyard to the battlefield.";
    CardDefinition {
        key: "fdn.alesha_who_laughs_at_fate",
        scryfall_id: "a93e3406-4e29-4bc0-ae52-cbd2ac1f99a4",
        printed: PrintedCharacteristics {
            name: "Alesha, Who Laughs at Fate".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Warrior".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::FirstStrike]),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                    intervening_if: None,
                    effects: vec![Effect::put_counters(
                        ObjectSelector::This,
                        CounterKind::PlusOnePlusOne,
                        1,
                    )],
                },
                Ability::Triggered {
                    condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                    intervening_if: Some(Condition::AttackedThisTurn),
                    effects: vec![Effect::return_to_battlefield(
                        ObjectSelector::Target(TargetSpec::GraveyardCardYouOwnWithinSourcePower {
                            predicate: CardPredicate::creature(),
                        }),
                        false,
                    )],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ashroot_animist() -> CardDefinition {
    let text_box = "Trample\nWhenever this creature attacks, another target creature you control gains trample and gets +X/+X until end of turn, where X is this creature's power.";
    CardDefinition {
        key: "fdn.ashroot_animist",
        scryfall_id: "dece147f-d71a-4c95-9fe5-f5c862ef14ac",
        printed: PrintedCharacteristics {
            name: "Ashroot Animist".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Lizard".into(), "Druid".into()],
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Trample]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::another(TargetSpec::CreatureYouControl),
                    effects: vec![
                        Effect::until_end_of_turn(
                            ObjectSelector::Target(TargetSpec::CreatureYouControl),
                            Modification::AddKeywords(Keywords::of(&[Keyword::Trample])),
                        ),
                        Effect::until_end_of_turn(
                            ObjectSelector::Target(TargetSpec::CreatureYouControl),
                            Modification::ModifyPt {
                                power: EffectValue::SourcePower,
                                toughness: EffectValue::SourcePower,
                            },
                        ),
                    ],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bulk_up() -> CardDefinition {
    let text_box = "Double target creature's power until end of turn.\nFlashback {4}{R}{R} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";
    CardDefinition {
        key: "fdn.bulk_up",
        scryfall_id: "977dcc50-da10-4281-b522-9240c1204f5d",
        printed: PrintedCharacteristics {
            name: "Bulk Up".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Instant]),
            statics: vec![StaticAbility::Flashback(ManaCost::parse("{4}{R}{R}"))],
            spell_effects: vec![Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::ModifyPt {
                    power: ObjectSelector::Target(TargetSpec::Creature)
                        .property(ObjectProperty::Power, Aggregate::Sum),
                    toughness: 0.into(),
                },
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn courageous_goblin() -> CardDefinition {
    let text_box = "Whenever this creature attacks while you control a creature with power 4 or greater, this creature gets +1/+0 and gains menace until end of turn. (It can't be blocked except by two or more creatures.)";
    CardDefinition {
        key: "fdn.courageous_goblin",
        scryfall_id: "8db6819c-666a-409d-85a5-b9ac34d8dd2f",
        printed: PrintedCharacteristics {
            name: "Courageous Goblin".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::ActiveIf {
                    active_if: Condition::compare(
                        ObjectSelector::each_you_control(CardPredicate {
                            min_power: Some(4),
                            ..CardPredicate::creature()
                        })
                        .count(),
                        CompareOp::Ge,
                        1,
                    ),
                    condition: Box::new(TriggerCondition::attacks(
                        TriggerSubject::This,
                        Per::Object,
                    )),
                },
                intervening_if: None,
                effects: vec![
                    Effect::until_end_of_turn(ObjectSelector::This, Modification::pump(1, 0)),
                    Effect::until_end_of_turn(
                        ObjectSelector::This,
                        Modification::AddKeywords(Keywords::of(&[Keyword::Menace])),
                    ),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn tragic_banshee() -> CardDefinition {
    let text_box = "Morbid — When this creature enters, target creature an opponent controls gets -1/-1 until end of turn. If a creature died this turn, that creature gets -13/-13 until end of turn instead.";
    let victim = || ObjectSelector::Target(TargetSpec::CreatureOpponentControls);
    CardDefinition {
        key: "fdn.tragic_banshee",
        scryfall_id: "30df3e33-2f17-4067-99f1-5db6b0f41fd4",
        printed: PrintedCharacteristics {
            name: "Tragic Banshee".into(),
            mana_cost: Some(ManaCost::parse("{4}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Spirit".into()],
            power: Some(5),
            toughness: Some(3),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                // Death history is read as the ability resolves (CR 608.2h).
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::CreatureOpponentControls),
                    effects: vec![Effect::If {
                        condition: Condition::CreatureDiedThisTurn,
                        then: vec![Effect::until_end_of_turn(
                            victim(),
                            Modification::pump(-13, -13),
                        )],
                        otherwise: vec![Effect::until_end_of_turn(
                            victim(),
                            Modification::pump(-1, -1),
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

fn vengeful_bloodwitch() -> CardDefinition {
    let text_box = "Whenever this creature or another creature you control dies, target opponent loses 1 life and you gain 1 life.";
    CardDefinition {
        key: "fdn.vengeful_bloodwitch",
        scryfall_id: "bd0c12dd-f138-45c0-9614-d83a1d8e8399",
        printed: PrintedCharacteristics {
            name: "Vengeful Bloodwitch".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vampire".into(), "Warlock".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::you_control(CardPredicate::creature()),
                },
                intervening_if: None,
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::Opponent),
                    effects: vec![
                        Effect::LoseLife {
                            player: PlayerSelector::Target(Box::new(TargetSpec::Opponent)),
                            amount: EffectValue::Fixed(1),
                        },
                        Effect::gain_life(1),
                    ],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn needletooth_pack() -> CardDefinition {
    let text_box = "Morbid — At the beginning of your end step, if a creature died this turn, put two +1/+1 counters on target creature you control.";
    CardDefinition {
        key: "fdn.needletooth_pack",
        scryfall_id: "993c1679-e02b-44f2-b34e-12fd6b5142e9",
        printed: PrintedCharacteristics {
            name: "Needletooth Pack".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dinosaur".into()],
            power: Some(4),
            toughness: Some(5),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                intervening_if: Some(Condition::CreatureDiedThisTurn),
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::CreatureYouControl),
                    effects: vec![Effect::put_counters(
                        ObjectSelector::Target(TargetSpec::CreatureYouControl),
                        CounterKind::PlusOnePlusOne,
                        2,
                    )],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn slumbering_cerberus() -> CardDefinition {
    let text_box = "This creature doesn't untap during your untap step.\nMorbid — At the beginning of each end step, if a creature died this turn, untap this creature.";
    CardDefinition {
        key: "fdn.slumbering_cerberus",
        scryfall_id: "9d06faa8-201d-45db-b398-ad56f7b01848",
        printed: PrintedCharacteristics {
            name: "Slumbering Cerberus".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dog".into()],
            power: Some(4),
            toughness: Some(2),
            statics: vec![StaticAbility::Restriction(StaticRestriction {
                scope: StaticScope::This,
                kind: Restriction::DoesntUntap,
                unless: None,
            })],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::AnyPlayer),
                intervening_if: Some(Condition::CreatureDiedThisTurn),
                effects: vec![Effect::untap(ObjectSelector::This)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn wardens_of_the_cycle() -> CardDefinition {
    let text_box = "Morbid — At the beginning of your end step, if a creature died this turn, choose one —\n• You gain 2 life.\n• You draw a card and you lose 1 life.";
    CardDefinition {
        key: "fdn.wardens_of_the_cycle",
        scryfall_id: "83ea9b2c-5723-4eff-88ac-6669975939e3",
        printed: PrintedCharacteristics {
            name: "Wardens of the Cycle".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Warlock".into()],
            power: Some(3),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
                intervening_if: Some(Condition::CreatureDiedThisTurn),
                effects: vec![Effect::TriggerModes {
                    modes: vec![
                        vec![Effect::gain_life(2)],
                        vec![
                            Effect::DrawCards {
                                count: EffectValue::Fixed(1),
                            },
                            Effect::LoseLife {
                                player: PlayerSelector::You,
                                amount: EffectValue::Fixed(1),
                            },
                        ],
                    ],
                    up_to: None,
                    unchosen: false,
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn scrawling_crawler() -> CardDefinition {
    let text_box = "At the beginning of your upkeep, each player draws a card.\nWhenever an opponent draws a card, that player loses 1 life.";
    CardDefinition {
        key: "fdn.scrawling_crawler",
        scryfall_id: "a1176dcf-40ee-4342-aa74-791b8352e99a",
        printed: PrintedCharacteristics {
            name: "Scrawling Crawler".into(),
            mana_cost: Some(ManaCost::parse("{3}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Phyrexian".into(), "Construct".into()],
            power: Some(3),
            toughness: Some(2),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::beginning_of(TurnStep::Upkeep, Actor::You),
                    intervening_if: None,
                    // Each player draws in APNAP order (CR 101.4).
                    effects: vec![Effect::ForPlayer {
                        player: PlayerSelector::EachPlayer,
                        effects: vec![Effect::DrawCards {
                            count: EffectValue::Fixed(1),
                        }],
                    }],
                },
                Ability::Triggered {
                    condition: TriggerCondition::Draws {
                        who: Actor::Opponent,
                        nth: None,
                    },
                    intervening_if: None,
                    effects: vec![Effect::LoseLife {
                        player: PlayerSelector::TriggeringPlayer,
                        amount: EffectValue::Fixed(1),
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn firespitter_whelp() -> CardDefinition {
    let text_box = "Flying\nWhenever you cast a noncreature or Dragon spell, this creature deals 1 damage to each opponent.";
    CardDefinition {
        key: "fdn.firespitter_whelp",
        scryfall_id: "4b3a4c7d-3126-4bde-9dca-cb6a1e2f37c9",
        printed: PrintedCharacteristics {
            name: "Firespitter Whelp".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                // One predicate: a noncreature Dragon spell triggers it once.
                condition: TriggerCondition::casts(
                    Actor::You,
                    CardPredicate {
                        any_of: vec![
                            CardPredicate {
                                not_card_types: vec![CardType::Creature],
                                ..Default::default()
                            },
                            CardPredicate::subtype("Dragon"),
                        ],
                        ..Default::default()
                    },
                ),
                intervening_if: None,
                effects: vec![Effect::DealDamageToPlayer {
                    player: PlayerSelector::EachOpponent,
                    amount: EffectValue::Fixed(1),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn archmage_of_runes() -> CardDefinition {
    let text_box = "Instant and sorcery spells you cast cost {1} less to cast.\nWhenever you cast an instant or sorcery spell, draw a card.";
    let instant_or_sorcery = || CardPredicate {
        card_types_any: vec![CardType::Instant, CardType::Sorcery],
        ..Default::default()
    };
    CardDefinition {
        key: "fdn.archmage_of_runes",
        scryfall_id: "334b5018-2da9-49f1-9d09-83d312ecfb02",
        printed: PrintedCharacteristics {
            name: "Archmage of Runes".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Giant".into(), "Wizard".into()],
            power: Some(3),
            toughness: Some(6),
            statics: vec![StaticAbility::SpellsCostLess(SpellCostReduction {
                first_on_own_turn: false,
                predicate: instant_or_sorcery(),
                condition: Condition::Always,
                generic: 1,
            })],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::casts(Actor::You, instant_or_sorcery()),
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

fn goblin_negotiation() -> CardDefinition {
    let text_box = "Goblin Negotiation deals X damage to target creature. Create a number of 1/1 red Goblin creature tokens equal to the amount of excess damage dealt to that creature this way.";
    CardDefinition {
        key: "fdn.goblin_negotiation",
        scryfall_id: "f2016585-e26c-4d13-b09f-af6383c192f7",
        printed: PrintedCharacteristics {
            name: "Goblin Negotiation".into(),
            mana_cost: Some(ManaCost::parse("{X}{R}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![
                Effect::deal_damage(EffectValue::X, ObjectSelector::Target(TargetSpec::Creature)),
                // Excess is measured as the damage is dealt (CR 120.4a):
                // prevented damage leaves none.
                Effect::create_tokens("token.goblin", EffectValue::ExcessDamage),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn niv_mizzet_visionary() -> CardDefinition {
    let text_box = "Flying\nYou have no maximum hand size.\nWhenever a source you control deals noncombat damage to an opponent, you draw that many cards.";
    CardDefinition {
        key: "fdn.niv_mizzet_visionary",
        scryfall_id: "7a69a618-d588-4745-8ede-0ff0a9f356f1",
        printed: PrintedCharacteristics {
            name: "Niv-Mizzet, Visionary".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Dragon".into(), "Wizard".into()],
            power: Some(5),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying]),
            statics: vec![StaticAbility::NoMaximumHandSize],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::DealsDamage {
                    kind: DamageKind::Noncombat,
                    source: TriggerSubject::you_control(CardPredicate::default()),
                    to: DamageRecipientFilter::Player(Actor::Opponent),
                },
                intervening_if: None,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::TriggeringAmount,
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
