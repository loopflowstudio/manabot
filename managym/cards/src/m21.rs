//! Core Set 2021.
use managym_state::{
    ability::{
        Ability, Actor, CompareOp, Condition, Effect, EffectValue, ObjectSelector, PlayerSelector,
        TargetRequirement, TargetSpec, TriggerCondition,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, PrintedCharacteristics,
        StaticAbility,
    },
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        heartfire_immolator(),
        mazemind_tome(),
        sanguine_indulgence(),
    ]
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
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
