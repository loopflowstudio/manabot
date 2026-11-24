//! Foundations cards, expressed in the engine's shared rules vocabulary.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, PaymentCost, Per,
        PlayWindow, Scope, TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes,
        DamageTransform, Keyword, Keywords, LayerValue, PrintedCharacteristics, ReplacementEffect,
        SacrificeCost, StaticAbility, StaticContinuous, StaticScope, TargetCostReduction,
    },
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        elvish_regrower(),
        clinquant_skymage(),
        seekers_folly(),
        goblin_surprise(),
        stab(),
        quakestrider_ceratops(),
        eager_trufflesnout(),
        infestation_sage(),
        elfsworn_giant(),
        rite_of_the_dragoncaller(),
        dragon_trainer(),
        guarded_heir(),
        anthem_of_champions(),
        arahbo_the_first_fang(),
        blasphemous_edict(),
        claws_out(),
        helpful_hunter(),
        hungry_ghoul(),
        herald_of_eternal_dawn(),
        kellan_planar_trailblazer(),
        lightshell_duo(),
        luminous_rebuke(),
        prideful_parent(),
        leyline_axe(),
        sire_of_seven_deaths(),
        twinflame_tyrant(),
    ]
}

fn twinflame_tyrant() -> CardDefinition {
    CardDefinition {
        key: "fdn.twinflame_tyrant",
        scryfall_id: "1eb34f51-0bd2-43c3-af95-2ce8dabcc7bb",
        printed: PrintedCharacteristics {
            name: "Twinflame Tyrant".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(3),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying]),
            replacement_effects: vec![ReplacementEffect::ModifyDamage {
                creature_source: false,
                source_by: Actor::You,
                recipient_by: Actor::Opponent,
                transform: DamageTransform::Double,
            }],
            text_box: "Flying\nIf a source you control would deal damage to an opponent or a permanent an opponent controls, it deals double that damage instead.".into(),
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

fn lightshell_duo() -> CardDefinition {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)\nWhen this creature enters, surveil 2. (Look at the top two cards of your library, then put any number of them into your graveyard and the rest on top of your library in any order.)";
    CardDefinition {
        key: "fdn.lightshell_duo",
        scryfall_id: "bb75315c-ea8f-4eb0-899e-c73ef75fc396",
        printed: PrintedCharacteristics {
            name: "Lightshell Duo".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Rat".into(), "Otter".into()],
            power: Some(3),
            toughness: Some(4),
            abilities: vec![
                Ability::prowess(),
                Ability::when_enters(TriggerSubject::This, vec![Effect::Surveil { count: 2 }]),
            ],
            text_box: text_box.into(),
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

fn arahbo_the_first_fang() -> CardDefinition {
    let cat = CardPredicate::subtype("Cat");
    let other_nontoken_cat = TriggerSubject::another_you_control(CardPredicate {
        token: Some(false),
        ..cat.clone()
    });
    let text_box = "Other Cats you control get +1/+1.\nWhenever Arahbo or another nontoken Cat you control enters, create a 1/1 white Cat creature token.";
    CardDefinition {
        key: "fdn.arahbo_the_first_fang",
        scryfall_id: "524a5d93-26ed-436d-a437-dc9460acce98",
        printed: PrintedCharacteristics {
            name: "Arahbo, the First Fang".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into(), "Avatar".into()],
            power: Some(2),
            toughness: Some(2),
            supertypes: vec!["Legendary".into()],
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::OtherYouControl(cat),
                condition: None,
                characteristic_defining: false,
                modifications: vec![Modification::ModifyPt {
                    power: LayerValue::Fixed(1),
                    toughness: LayerValue::Fixed(1),
                }],
            })],
            abilities: vec![Ability::when_enters(
                TriggerSubject::ThisOr(Box::new(other_nontoken_cat)),
                vec![Effect::create_tokens("token.cat_1_1_w", 1)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn helpful_hunter() -> CardDefinition {
    CardDefinition {
        key: "fdn.helpful_hunter",
        scryfall_id: "1b9a0e91-80b5-428f-8f08-931d0631be14",
        printed: PrintedCharacteristics {
            name: "Helpful Hunter".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            )],
            text_box: "When this creature enters, draw a card.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn prideful_parent() -> CardDefinition {
    CardDefinition {
        key: "fdn.prideful_parent",
        scryfall_id: "b742117a-8a72-43b9-b05d-274829d138a2",
        printed: PrintedCharacteristics {
            name: "Prideful Parent".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Vigilance]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::create_tokens("token.cat_1_1_w", 1)],
            )],
            text_box: "Vigilance (Attacking doesn't cause this creature to tap.)\nWhen this creature enters, create a 1/1 white Cat creature token.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn anthem_of_champions() -> CardDefinition {
    CardDefinition {
        key: "fdn.anthem_of_champions",
        scryfall_id: "42fe3a40-9cbe-4235-86f9-32576aaebba8",
        printed: PrintedCharacteristics {
            name: "Anthem of Champions".into(),
            mana_cost: Some(ManaCost::parse("{G}{W}")),
            types: CardTypes::new([CardType::Enchantment]),
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::YouControl(CardPredicate::creature()),
                condition: None,
                characteristic_defining: false,
                modifications: vec![Modification::ModifyPt {
                    power: LayerValue::Fixed(1),
                    toughness: LayerValue::Fixed(1),
                }],
            })],
            text_box: "Creatures you control get +1/+1.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn guarded_heir() -> CardDefinition {
    let text_box = "Lifelink (Damage dealt by this creature also causes you to gain that much life.)\nWhen this creature enters, create two 3/3 white Knight creature tokens.";
    CardDefinition {
        key: "fdn.guarded_heir",
        scryfall_id: "525ba5c7-3ce5-4e52-b8b5-96c9040a6738",
        printed: PrintedCharacteristics {
            name: "Guarded Heir".into(),
            mana_cost: Some(ManaCost::parse("{5}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Noble".into()],
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Lifelink]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::create_tokens("token.knight_3_3_w", 2)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dragon_trainer() -> CardDefinition {
    let text_box = "When this creature enters, create a 4/4 red Dragon creature token with flying.";
    CardDefinition {
        key: "fdn.dragon_trainer",
        scryfall_id: "91bd75a1-cb54-4e38-9ce1-e8f32a73c6eb",
        printed: PrintedCharacteristics {
            name: "Dragon Trainer".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::create_tokens("token.dragon_4_4_r_flying", 1)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rite_of_the_dragoncaller() -> CardDefinition {
    let text_box = "Whenever you cast an instant or sorcery spell, create a 5/5 red Dragon creature token with flying.";
    CardDefinition {
        key: "fdn.rite_of_the_dragoncaller",
        scryfall_id: "673e4561-8dfd-46db-b492-878009666ac7",
        printed: PrintedCharacteristics {
            name: "Rite of the Dragoncaller".into(),
            mana_cost: Some(ManaCost::parse("{4}{R}{R}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::casts(
                    Actor::You,
                    CardPredicate {
                        card_types_any: vec![CardType::Instant, CardType::Sorcery],
                        ..Default::default()
                    },
                ),
                effects: vec![Effect::create_tokens("token.dragon_5_5_r_flying", 1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn elfsworn_giant() -> CardDefinition {
    let text_box = "Reach (This creature can block creatures with flying.)\nLandfall — Whenever a land you control enters, create a 1/1 green Elf Warrior creature token.";
    CardDefinition {
        key: "fdn.elfsworn_giant",
        scryfall_id: "5128a5be-ffa6-4998-8488-872d80b24cb2",
        printed: PrintedCharacteristics {
            name: "Elfsworn Giant".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Giant".into()],
            power: Some(5),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Reach]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::you_control(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
                vec![Effect::create_tokens("token.elf_warrior_1_1_g", 1)],
            )],
            text_box: text_box.into(),
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

fn infestation_sage() -> CardDefinition {
    let text_box =
        "When this creature dies, create a 1/1 black and green Insect creature token with flying.";
    CardDefinition {
        key: "fdn.infestation_sage",
        scryfall_id: "d40c73de-7a5f-46f2-a70b-449bc8ecfe24",
        printed: PrintedCharacteristics {
            name: "Infestation Sage".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Warlock".into()],
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::create_tokens("token.insect_1_1_bg_flying", 1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn eager_trufflesnout() -> CardDefinition {
    let text_box = "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nWhenever this creature deals combat damage to a player, create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")";
    CardDefinition {
        key: "fdn.eager_trufflesnout",
        scryfall_id: "a6e8433d-eb2a-43d1-b59b-7d70ff97c8e7",
        printed: PrintedCharacteristics {
            name: "Eager Trufflesnout".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Boar".into()],
            power: Some(4),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Trample]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::DealsCombatDamage {
                    subject: TriggerSubject::This,
                    to: Actor::AnyPlayer,
                    per: Per::Object,
                },
                effects: vec![Effect::create_tokens("token.food", 1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn quakestrider_ceratops() -> CardDefinition {
    CardDefinition {
        key: "fdn.quakestrider_ceratops",
        scryfall_id: "067f72c2-ead6-4879-bc9d-696c9f87c0b2",
        printed: PrintedCharacteristics {
            name: "Quakestrider Ceratops".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dinosaur".into()],
            power: Some(12),
            toughness: Some(8),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn stab() -> CardDefinition {
    CardDefinition {
        key: "fdn.stab",
        scryfall_id: "6859a5ba-1c1c-4631-bba8-f9900b827178",
        printed: PrintedCharacteristics {
            name: "Stab".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::pump(-2, -2),
            )],
            text_box: "Target creature gets -2/-2 until end of turn.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn goblin_surprise() -> CardDefinition {
    let text_box = "Choose one —\n• Creatures you control get +2/+0 until end of turn.\n• Create two 1/1 red Goblin creature tokens.";
    CardDefinition {
        key: "fdn.goblin_surprise",
        scryfall_id: "527dd5d4-5f72-40bb-8a9d-1f5ac3f81e2e",
        printed: PrintedCharacteristics {
            name: "Goblin Surprise".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::each(Scope::YouControl, CardPredicate::creature()),
                        Modification::pump(2, 0),
                    )],
                    vec![Effect::create_tokens("token.goblin_1_1_r", 2)],
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn seekers_folly() -> CardDefinition {
    let text_box = "Choose one —\n• Target opponent discards two cards.\n• Creatures your opponents control get -1/-1 until end of turn.";
    CardDefinition {
        key: "fdn.seekers_folly",
        scryfall_id: "bc359da6-8b7f-45ec-b530-ce159fc35953",
        printed: PrintedCharacteristics {
            name: "Seeker's Folly".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::Modal {
                modes: vec![
                    vec![Effect::Discard {
                        what: ObjectSelector::Target(TargetSpec::Opponent),
                        count: 2,
                    }],
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::each(Scope::OpponentsControl, CardPredicate::creature()),
                        Modification::pump(-1, -1),
                    )],
                ],
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

fn clinquant_skymage() -> CardDefinition {
    CardDefinition {
        key: "fdn.clinquant_skymage",
        scryfall_id: "36012810-0e83-4640-8ba7-7262229f1b84",
        printed: PrintedCharacteristics {
            name: "Clinquant Skymage".into(),
            subtypes: vec!["Bird".into(), "Wizard".into()],
            types: CardTypes::new([CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            power: Some(1),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Draws {
                    who: Actor::You,
                    nth: None,
                },
                intervening_if: None,
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
            }],
            text_box: "Flying\nWhenever you draw a card, put a +1/+1 counter on this creature."
                .into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn elvish_regrower() -> CardDefinition {
    let text_box =
        "When this creature enters, return target permanent card from your graveyard to your hand.";
    CardDefinition {
        key: "fdn.elvish_regrower",
        scryfall_id: "2694e3cd-26ed-4a10-ae55-fb84d7800253",
        printed: PrintedCharacteristics {
            name: "Elvish Regrower".into(),
            subtypes: vec!["Elf".into(), "Druid".into()],
            types: CardTypes::new([CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{2}{G}{G}")),
            power: Some(4),
            toughness: Some(3),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::return_to_hand(ObjectSelector::Target(
                    TargetSpec::GraveyardCardYouOwn {
                        predicate: CardPredicate {
                            card_types_any: vec![
                                CardType::Artifact,
                                CardType::Battle,
                                CardType::Creature,
                                CardType::Enchantment,
                                CardType::Land,
                                CardType::Planeswalker,
                            ],
                            ..Default::default()
                        },
                    },
                ))],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
