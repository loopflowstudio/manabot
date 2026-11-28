//! Core Set 2021.
use managym_state::ability::{
    Ability, Actor, CompareOp, Condition, Effect, EffectValue, ObjectSelector, PlayerSelector,
    TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject,
};
use managym_state::card::{
    ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, EntryScope, Keyword, Keywords,
    PrintedCharacteristics, ReplacementEffect, StaticAbility, StaticKeywordGrant, StaticScope,
};
use managym_state::effect::Modification;
use managym_state::mana::ManaCost;
use managym_state::permanent::CounterKind;
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        wildwood_scourge(),
        garruks_uprising(),
        heartfire_immolator(),
        mazemind_tome(),
        primal_might(),
        sanguine_indulgence(),
        shipwreck_dowser(),
    ]
}

fn primal_might() -> CardDefinition {
    let text_box = "Target creature you control gets +X/+X until end of turn. Then it fights up to one target creature you don't control. (Each deals damage equal to its power to the other.)";
    CardDefinition {
        key: "m21.primal_might",
        scryfall_id: "1cd8cee8-7ea0-4037-8a3b-39334dc064fb",
        printed: PrintedCharacteristics {
            name: "Primal Might".into(),
            mana_cost: Some(ManaCost::parse("{X}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::TargetClause {
                    requirement: TargetRequirement::one(TargetSpec::CreatureYouControl),
                    effects: vec![Effect::until_end_of_turn(
                        ObjectSelector::ClauseTarget(0),
                        Modification::ModifyPt {
                            power: EffectValue::X,
                            toughness: EffectValue::X,
                        },
                    )],
                },
                Effect::TargetClause {
                    requirement: TargetRequirement::up_to(1, TargetSpec::CreatureOpponentControls),
                    effects: vec![Effect::fight_last_target()],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mazemind_tome() -> CardDefinition {
    let text_box = "{T}, Put a page counter on this artifact: Scry 1. (Look at the top card of your library. You may put that card on the bottom.)\n{2}, {T}, Put a page counter on this artifact: Draw a card.\nWhen there are four or more page counters on this artifact, exile it. If you do, you gain 4 life.";
    let page = Some((CounterKind::Page, 1));
    CardDefinition {
        key: "m21.mazemind_tome",
        scryfall_id: "9fd761f3-6b43-4150-8595-dc3abd85b06c",
        printed: PrintedCharacteristics {
            name: "Mazemind Tome".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Book".into()],
            activated_abilities: vec![
                ActivatedAbilityDefinition {
                    tap_source: true,
                    add_counters: page,
                    effects: vec![Effect::Scry { count: 1 }],
                    ..Default::default()
                },
                ActivatedAbilityDefinition {
                    mana_cost: ManaCost::parse("{2}"),
                    tap_source: true,
                    add_counters: page,
                    effects: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                    ..Default::default()
                },
            ],
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::State(Condition::SourceCountersAtLeast {
                    kind: CounterKind::Page,
                    count: 4,
                }),
                effects: vec![
                    Effect::exile(ObjectSelector::This),
                    Effect::if_you_do(vec![Effect::gain_life(4)]),
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sanguine_indulgence() -> CardDefinition {
    let text_box = "This spell costs {3} less to cast if you've gained 3 or more life this turn.\nReturn up to two target creature cards from your graveyard to your hand.";
    let spec = TargetSpec::GraveyardCardYouOwn {
        predicate: CardPredicate::creature(),
    };
    CardDefinition {
        key: "m21.sanguine_indulgence",
        scryfall_id: "abfcd08a-cfb5-4d34-b950-f57a88c5cb8e",
        printed: PrintedCharacteristics {
            name: "Sanguine Indulgence".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            statics: vec![StaticAbility::CostsLess {
                amount: EffectValue::Fixed(3),
                condition: Some(Condition::compare(
                    EffectValue::LifeGainedThisTurn(PlayerSelector::You),
                    CompareOp::Ge,
                    3,
                )),
            }],
            spell_effects: vec![Effect::return_to_hand(ObjectSelector::Targets(
                spec.clone(),
            ))],
            targeting: vec![TargetRequirement::up_to(2, spec)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn shipwreck_dowser() -> CardDefinition {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)\nWhen this creature enters, return target instant or sorcery card from your graveyard to your hand.";
    CardDefinition {
        key: "m21.shipwreck_dowser",
        scryfall_id: "59d38ef7-5017-4ea3-b97f-a8fe12d03e98",
        printed: PrintedCharacteristics {
            name: "Shipwreck Dowser".into(),
            subtypes: vec!["Merfolk".into(), "Wizard".into()],
            types: CardTypes::new([CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{3}{U}{U}")),
            power: Some(3),
            toughness: Some(3),
            abilities: vec![
                Ability::prowess(),
                Ability::when_enters(
                    TriggerSubject::This,
                    vec![Effect::return_to_hand(ObjectSelector::Target(
                        TargetSpec::GraveyardCardYouOwn {
                            predicate: CardPredicate::instant_or_sorcery(),
                        },
                    ))],
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn heartfire_immolator() -> CardDefinition {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)\n{R}, Sacrifice this creature: It deals damage equal to its power to target creature or planeswalker.";

    CardDefinition {
        key: "m21.heartfire_immolator",
        scryfall_id: "869fb9f1-0d59-4874-aa52-ac665c3cc0e8",
        printed: PrintedCharacteristics {
            name: "Heartfire Immolator".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
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
            }],
            // "Its power": the sacrificed appearance's last known power
            // (CR 608.2h), read through the exact ability source.
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{R}"),
                sacrifice_source: true,
                effects: vec![Effect::deal_damage(
                    EffectValue::SourcePower,
                    ObjectSelector::Target(TargetSpec::Permanent {
                        predicate: CardPredicate {
                            card_types_any: vec![CardType::Creature, CardType::Planeswalker],
                            ..Default::default()
                        },
                    }),
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn garruks_uprising() -> CardDefinition {
    let powerful = CardPredicate {
        min_power: Some(4),
        ..CardPredicate::creature()
    };
    let draw = Effect::DrawCards { count: 1.into() };
    let text_box = "When this enchantment enters, if you control a creature with power 4 or greater, draw a card.\nCreatures you control have trample. (Each of those creatures can deal excess combat damage to the player or planeswalker it's attacking.)\nWhenever a creature you control with power 4 or greater enters, draw a card.";
    CardDefinition {
        key: "m21.garruks_uprising",
        scryfall_id: "71a4860a-8bb6-45c0-b00a-b4a42da33ab9",
        printed: PrintedCharacteristics {
            name: "Garruk's Uprising".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::This,
                    },
                    intervening_if: Some(Condition::ControlsAtLeast {
                        count: 1,
                        predicate: powerful.clone(),
                        distinct_names: false,
                    }),
                    effects: vec![draw.clone()],
                },
                Ability::when_enters(TriggerSubject::you_control(powerful), vec![draw]),
            ],
            statics: vec![StaticAbility::KeywordGrant(StaticKeywordGrant {
                scope: StaticScope::YouControl(CardPredicate::creature()),
                condition: None,
                keywords: Keywords::of(&[Keyword::Trample]),
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn wildwood_scourge() -> CardDefinition {
    let text_box = "This creature enters with X +1/+1 counters on it.\nWhenever one or more +1/+1 counters are put on another non-Hydra creature you control, put a +1/+1 counter on this creature.";
    CardDefinition {
        key: "m21.wildwood_scourge",
        scryfall_id: "46ff0b33-d153-4b0e-ac48-7e5ed70dea09",
        printed: PrintedCharacteristics {
            name: "Wildwood Scourge".into(),
            mana_cost: Some(ManaCost::parse("{X}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Hydra".into()],
            power: Some(0),
            toughness: Some(0),
            replacement_effects: vec![ReplacementEffect::EntersWithCounters {
                on: EntryScope::This,
                kind: CounterKind::PlusOnePlusOne,
                count: EffectValue::X,
                condition: None,
            }],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::PutsCountersOn {
                    who: Actor::AnyPlayer,
                    kind: CounterKind::PlusOnePlusOne,
                    subject: TriggerSubject::another_you_control(CardPredicate {
                        not_subtypes: vec!["Hydra".into()],
                        ..CardPredicate::creature()
                    }),
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
