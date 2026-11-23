//! Foundations cards, expressed in the engine's shared rules vocabulary.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, PaymentCost, Per,
        PlayWindow, TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes, Keyword,
        Keywords, LayerValue, PrintedCharacteristics, SacrificeCost, StaticAbility,
        StaticContinuous, StaticScope, TargetCostReduction,
    },
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        anthem_of_champions(),
        arahbo_the_first_fang(),
        blasphemous_edict(),
        claws_out(),
        helpful_hunter(),
        hungry_ghoul(),
        kellan_planar_trailblazer(),
        lightshell_duo(),
        luminous_rebuke(),
        prideful_parent(),
        sire_of_seven_deaths(),
    ]
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
