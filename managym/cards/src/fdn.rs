//! Foundations cards, expressed in the engine's shared rules vocabulary.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, ObjectSelector, Per, PlayWindow, TargetRequirement,
        TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, SacrificeCost, StaticAbility, TargetCostReduction,
    },
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        hungry_ghoul(),
        kellan_planar_trailblazer(),
        luminous_rebuke(),
    ]
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
