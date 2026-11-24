//! Foundations cards, expressed in the engine's shared rules vocabulary.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, ObjectSelector, PaymentCost, Per, PlayWindow,
        TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject, TurnStep,
    },
    card::ManaAbility,
    card::{
        ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes, Keyword,
        Keywords, PrintedCharacteristics, SacrificeCost, StaticAbility, StaticKeywordGrant,
        StaticPtBuff, StaticScope, TargetCostReduction,
    },
    effect::{CopyException, Modification},
    mana::{Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
};
use managym_state::{
    ability::{EffectValue, EntryCounters},
    card::ReplacementEffect,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        fiendish_panda(),
        sun_blessed_healer(),
        kykar_zephyr_awakener(),
        infernal_vessel(),
        nine_lives_familiar(),
        valkyrie_s_call(),
        electroduplicate(),
        blasphemous_edict(),
        claws_out(),
        homunculus_horde(),
        hungry_ghoul(),
        herald_of_eternal_dawn(),
        kellan_planar_trailblazer(),
        luminous_rebuke(),
        leyline_axe(),
        sire_of_seven_deaths(),
        raise_the_past(),
        soulstone_sanctuary(),
        dropkick_bomber(),
        celestial_armor(),
    ]
}

fn celestial_armor() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)\nWhen this Equipment enters, attach it to target creature you control. That creature gains hexproof and indestructible until end of turn.\nEquipped creature gets +2/+0 and has flying.\nEquip {3}{W} ({3}{W}: Attach to target creature you control. Equip only as a sorcery.)";
    CardDefinition {
        key: "fdn.celestial_armor",
        scryfall_id: "809ee8ad-1573-49e5-9e84-b7cdd29efcae",
        printed: PrintedCharacteristics {
            name: "Celestial Armor".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            keywords: Keywords::of(&[Keyword::Flash]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![
                    Effect::AttachSourceTo {
                        what: ObjectSelector::Target(TargetSpec::CreatureYouControl),
                    },
                    Effect::until_end_of_turn(
                        ObjectSelector::Target(TargetSpec::CreatureYouControl),
                        Modification::AddKeywords(Keywords::of(&[
                            Keyword::Hexproof,
                            Keyword::Indestructible,
                        ])),
                    ),
                ],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                sorcery_timing: true,
                mana_cost: ManaCost::parse("{3}{W}"),
                effects: vec![Effect::AttachSourceTo {
                    what: ObjectSelector::Target(TargetSpec::CreatureYouControl),
                }],
                ..Default::default()
            }],
            statics: vec![
                StaticAbility::PtBuff(StaticPtBuff {
                    scope: StaticScope::Attached,
                    condition: None,
                    power: 2,
                    toughness: 0,
                    per_creature_type_max: None,
                    per_color_among: None,
                    per_graveyard_matching: None,
                }),
                StaticAbility::KeywordGrant(StaticKeywordGrant {
                    scope: StaticScope::Attached,
                    condition: None,
                    keywords: Keywords::of(&[Keyword::Flying]),
                }),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dropkick_bomber() -> CardDefinition {
    let text_box = "Other Goblins you control get +1/+1.\n{R}: Until end of turn, another target Goblin you control gains flying and \"When this creature deals combat damage, sacrifice it.\"";
    let goblin = TargetSpec::PermanentYouControl {
        predicate: CardPredicate::subtype("Goblin"),
    };
    CardDefinition {
        key: "fdn.dropkick_bomber",
        scryfall_id: "b44f758e-716a-408e-96d4-b58403591c2a",
        printed: PrintedCharacteristics {
            name: "Dropkick Bomber".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Warrior".into()],
            power: Some(2),
            toughness: Some(3),
            statics: vec![StaticAbility::PtBuff(StaticPtBuff {
                scope: StaticScope::OtherYouControl(CardPredicate::subtype("Goblin")),
                condition: None,
                power: 1,
                toughness: 1,
                per_creature_type_max: None,
                per_color_among: None,
                per_graveyard_matching: None,
            })],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{R}"),
                effects: vec![Effect::TargetClause {
                    requirement: TargetRequirement::another(goblin.clone()),
                    effects: vec![
                        Effect::until_end_of_turn(
                            ObjectSelector::Target(goblin.clone()),
                            Modification::AddKeywords(Keywords::of(&[Keyword::Flying])),
                        ),
                        Effect::until_end_of_turn(
                            ObjectSelector::Target(goblin),
                            Modification::GrantTriggered(Box::new(Ability::Triggered {
                                condition: TriggerCondition::SourceDealsCombatDamage,
                                intervening_if: None,
                                effects: vec![Effect::sacrifice_this()],
                            })),
                        ),
                    ],
                }],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn soulstone_sanctuary() -> CardDefinition {
    let text_box = "{T}: Add {C}.\n{4}: This land becomes a 3/3 creature with vigilance and all creature types. It's still a land.";
    CardDefinition {
        key: "fdn.soulstone_sanctuary",
        scryfall_id: "642553a7-6d0f-483d-a873-3a703786db42",
        printed: PrintedCharacteristics {
            name: "Soulstone Sanctuary".into(),
            types: CardTypes::new([CardType::Land]),
            mana_abilities: vec![ManaAbility {
                mana: Mana::parse("C"),
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{4}"),
                effects: [
                    Modification::BecomeCreature,
                    Modification::base_pt(Some(3), Some(3)),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Vigilance])),
                    Modification::AllCreatureTypes,
                ]
                .into_iter()
                .map(|modification| Effect::indefinitely(ObjectSelector::This, modification))
                .collect(),
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fiendish_panda() -> CardDefinition {
    let text_box = "Whenever you gain life, put a +1/+1 counter on this creature.\nWhen this creature dies, return another target non-Bear creature card with mana value less than or equal to this creature's power from your graveyard to the battlefield.";
    let target = TargetSpec::GraveyardCardYouOwnWithinSourcePower {
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
                    condition: TriggerCondition::GainsLife { who: Actor::You },
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
                        requirement: TargetRequirement::another(target.clone()),
                        effects: vec![Effect::return_to_battlefield(
                            ObjectSelector::Target(target),
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

fn homunculus_horde() -> CardDefinition {
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
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Draws { who: Actor::You, nth: Some(2) },
                effects: vec![Effect::create_token_copies(ObjectSelector::This, 1)],
            }],
            text_box: "Whenever you draw your second card each turn, create a token that's a copy of this creature.".into(),
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

fn electroduplicate() -> CardDefinition {
    let text_box = "Create a token that's a copy of target creature you control, except it has haste and \"At the beginning of the end step, sacrifice this token.\"\nFlashback {2}{R}{R} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";
    CardDefinition {
        key: "fdn.electroduplicate",
        scryfall_id: "abb06b1c-5d4e-49b9-9c4a-e60ab656a257",
        printed: PrintedCharacteristics {
            name: "Electroduplicate".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            statics: vec![StaticAbility::Flashback(ManaCost::parse("{2}{R}{R}"))],
            spell_effects: vec![Effect::CreateTokenCopy {
                count: 1.into(),
                of: ObjectSelector::Target(TargetSpec::CreatureYouControl),
                exceptions: vec![
                    CopyException::AddKeywords(Keywords::of(&[Keyword::Haste])),
                    CopyException::GrantTriggered(Box::new(Ability::Triggered {
                        condition: TriggerCondition::beginning_of(TurnStep::End, Actor::AnyPlayer),
                        intervening_if: None,
                        effects: vec![Effect::sacrifice_this()],
                    })),
                ],
            }],
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

fn valkyrie_s_call() -> CardDefinition {
    let text_box = "Whenever a nontoken, non-Angel creature you control dies, return that card to the battlefield under its owner's control with a +1/+1 counter on it. It has flying and is an Angel in addition to its other types.";
    CardDefinition {
        key: "fdn.valkyrie_s_call",
        scryfall_id: "0e1f1ff2-fa8f-4d38-b631-2d6e08e614c8",
        printed: PrintedCharacteristics {
            name: "Valkyrie's Call".into(),
            mana_cost: Some(ManaCost::parse("{3}{W}{W}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::you_control(CardPredicate {
                        token: Some(false),
                        not_subtypes: vec!["Angel".into()],
                        ..CardPredicate::creature()
                    }),
                },
                intervening_if: None,
                effects: vec![
                    Effect::return_to_owner(
                        ObjectSelector::Triggering,
                        false,
                        Some(CounterKind::PlusOnePlusOne.into()),
                    ),
                    Effect::indefinitely(
                        ObjectSelector::Moved,
                        Modification::AddKeywords(Keywords::of(&[Keyword::Flying])),
                    ),
                    Effect::indefinitely(
                        ObjectSelector::Moved,
                        Modification::AddSubtypes(vec!["Angel".into()]),
                    ),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn nine_lives_familiar() -> CardDefinition {
    let text_box = "This creature enters with eight revival counters on it if you cast it.\nWhen this creature dies, if it had a revival counter on it, return it to the battlefield with one fewer revival counter on it at the beginning of the next end step.";
    CardDefinition {
        key: "fdn.nine_lives_familiar",
        scryfall_id: "988c23f6-59fe-49f9-a9ce-9881dccb7033",
        printed: PrintedCharacteristics {
            name: "Nine-Lives Familiar".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Cat".into()],
            power: Some(1),
            toughness: Some(1),
            replacement_effects: vec![ReplacementEffect::EntersWithCounters {
                kind: CounterKind::Revival,
                count: 8,
                if_cast: true,
            }],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                intervening_if: Some(Condition::SourceCountersAtLeast {
                    kind: CounterKind::Revival,
                    count: 1,
                }),
                effects: vec![Effect::at_next_end_step(
                    ObjectSelector::This,
                    vec![Effect::Move {
                        what: ObjectSelector::Remembered,
                        to: ZoneType::Battlefield,
                        tapped: false,
                        transformed: false,
                        under: None,
                        with_counter: Some(EntryCounters {
                            kind: CounterKind::Revival,
                            count: EffectValue::DeathCountersMinus {
                                kind: CounterKind::Revival,
                                subtract: 1,
                            },
                        }),
                        linked: None,
                        library_end: None,
                    }],
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn infernal_vessel() -> CardDefinition {
    let text_box = "When this creature dies, if it wasn't a Demon, return it to the battlefield under its owner's control with two +1/+1 counters on it. It's a Demon in addition to its other types.";
    CardDefinition {
        key: "fdn.infernal_vessel",
        scryfall_id: "877b6330-2d0b-4f2f-a848-f10b06fb4ef5",
        printed: PrintedCharacteristics {
            name: "Infernal Vessel".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Cleric".into()],
            power: Some(2),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                intervening_if: Some(Condition::SourceMatches(CardPredicate {
                    not_subtypes: vec!["Demon".into()],
                    ..Default::default()
                })),
                effects: vec![
                    Effect::return_to_owner(
                        ObjectSelector::This,
                        false,
                        Some(EntryCounters {
                            kind: CounterKind::PlusOnePlusOne,
                            count: 2.into(),
                        }),
                    ),
                    Effect::indefinitely(
                        ObjectSelector::Moved,
                        Modification::AddSubtypes(vec!["Demon".into()]),
                    ),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn kykar_zephyr_awakener() -> CardDefinition {
    let text_box = "Flying\nWhenever you cast a noncreature spell, choose one —\n• Exile another target creature you control. Return that card to the battlefield under its owner's control at the beginning of the next end step.\n• Create a 1/1 white Spirit creature token with flying.";
    CardDefinition {
        key: "fdn.kykar_zephyr_awakener",
        scryfall_id: "6c980998-7124-4ec6-a0b6-e8d9a2364925",
        printed: PrintedCharacteristics {
            name: "Kykar, Zephyr Awakener".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}{U}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Bird".into(), "Wizard".into()],
            power: Some(3),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Casts {
                    who: Actor::You,
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Creature],
                        ..Default::default()
                    },
                    nth: None,
                },
                intervening_if: None,
                effects: vec![Effect::TriggerModes {
                    unchosen: false,
                    up_to: None,
                    modes: vec![
                        vec![Effect::TargetClause {
                            requirement: TargetRequirement::another(TargetSpec::CreatureYouControl),
                            effects: vec![
                                Effect::exile(ObjectSelector::Target(
                                    TargetSpec::CreatureYouControl,
                                )),
                                Effect::at_next_end_step(
                                    ObjectSelector::Moved,
                                    vec![Effect::return_to_owner(
                                        ObjectSelector::Remembered,
                                        false,
                                        None,
                                    )],
                                ),
                            ],
                        }],
                        vec![Effect::create_tokens("token.flying_spirit", 1)],
                    ],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sun_blessed_healer() -> CardDefinition {
    let text_box = "Kicker {1}{W} (You may pay an additional {1}{W} as you cast this spell.)\nLifelink (Damage dealt by this creature also causes you to gain that much life.)\nWhen this creature enters, if it was kicked, return target nonland permanent card with mana value 2 or less from your graveyard to the battlefield.";
    CardDefinition {
        key: "fdn.sun_blessed_healer",
        scryfall_id: "323d029e-9a88-4188-b3a4-38ef32cffc9f",
        printed: PrintedCharacteristics {
            name: "Sun-Blessed Healer".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Cleric".into()],
            power: Some(3),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Lifelink]),
            additional_costs: vec![AdditionalCost::Kicker(ManaCost::parse("{1}{W}"))],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: Some(Condition::Kicked),
                effects: vec![Effect::return_to_battlefield(
                    ObjectSelector::Target(TargetSpec::GraveyardCardYouOwn {
                        predicate: CardPredicate {
                            not_card_types: vec![CardType::Land],
                            max_mana_value: Some(2),
                            ..CardPredicate::permanent()
                        },
                    }),
                    false,
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn raise_the_past() -> CardDefinition {
    let text_box = "Return all creature cards with mana value 2 or less from your graveyard to the battlefield.";
    CardDefinition {
        key: "fdn.raise_the_past",
        scryfall_id: "6c6be129-56da-4fe7-a6bd-6a1d402c09e1",
        printed: PrintedCharacteristics {
            name: "Raise the Past".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            // One move of every card: the creatures enter together and see
            // each other enter (CR 603.6a).
            spell_effects: vec![Effect::return_to_battlefield(
                ObjectSelector::ZoneMatching {
                    zone: ZoneType::Graveyard,
                    owner: Actor::You,
                    predicate: CardPredicate {
                        max_mana_value: Some(2),
                        ..CardPredicate::creature()
                    },
                },
                false,
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
