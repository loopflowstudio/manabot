//! War of the Spark.
use managym_state::{
    ability::{
        Ability, CompareOp, Condition, Effect, EffectValue, ObjectSelector, Scope, TargetSpec,
        TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics, StaticAbility},
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![bolt_bend(), finale_of_revelation(), mayhem_devil()]
}

fn finale_of_revelation() -> CardDefinition {
    let text_box = "Draw X cards. If X is 10 or more, instead shuffle your graveyard into your library, draw X cards, untap up to five lands, and you have no maximum hand size for the rest of the game.\nExile Finale of Revelation.";
    let land = CardPredicate {
        card_type: Some(CardType::Land),
        ..Default::default()
    };
    CardDefinition {
        key: "war.finale_of_revelation",
        scryfall_id: "6630c34a-1a97-4e31-9d2c-1150b0aa903e",
        printed: PrintedCharacteristics {
            name: "Finale of Revelation".into(),
            mana_cost: Some(ManaCost::parse("{X}{U}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::If {
                    condition: Condition::compare(EffectValue::X, CompareOp::Ge, 10),
                    then: vec![
                        Effect::ShuffleGraveyardIntoLibrary,
                        Effect::DrawCards {
                            count: EffectValue::X,
                        },
                        Effect::SelectCards {
                            from: ObjectSelector::each(Scope::Battlefield, land),
                            predicate: CardPredicate::default(),
                            max_mana_value: None,
                            min: 0,
                            max: 5.into(),
                            effects: vec![Effect::Untap {
                                what: ObjectSelector::Selected,
                            }],
                            if_declined: vec![],
                        },
                        Effect::indefinitely(ObjectSelector::You, Modification::NoMaximumHandSize),
                    ],
                    otherwise: vec![Effect::DrawCards {
                        count: EffectValue::X,
                    }],
                },
                Effect::exile(ObjectSelector::This),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mayhem_devil() -> CardDefinition {
    let text_box =
        "Whenever a player sacrifices a permanent, this creature deals 1 damage to any target.";

    CardDefinition {
        key: "war.mayhem_devil",
        scryfall_id: "17416926-168b-49b3-9231-acbb8f8a1d13",
        printed: PrintedCharacteristics {
            name: "Mayhem Devil".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Devil".into()],
            power: Some(3),
            toughness: Some(3),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Sacrificed {
                    subject: TriggerSubject::any(CardPredicate::default()),
                },
                effects: vec![Effect::deal_damage(
                    1,
                    ObjectSelector::Target(TargetSpec::AnyTarget),
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn bolt_bend() -> CardDefinition {
    CardDefinition {
        key: "war.bolt_bend",
        scryfall_id: "39b35408-3728-4e1b-9f58-b0775df914d6",
        printed: PrintedCharacteristics {
            name: "Bolt Bend".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Instant]),
            statics: vec![StaticAbility::CostsLess {
                amount: 3.into(),
                condition: Some(Condition::ControlsAtLeast {
                    count: 1,
                    distinct_names: false,
                    predicate: CardPredicate { min_power: Some(4), ..CardPredicate::creature() },
                }),
            }],
            spell_effects: vec![Effect::ChangeTarget { target: TargetSpec::SpellOrAbilityWithSingleTarget }],
            text_box: "This spell costs {3} less to cast if you control a creature with power 4 or greater.\nChange the target of target spell or ability with a single target.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
