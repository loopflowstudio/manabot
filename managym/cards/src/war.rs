//! War of the Spark.
use managym_state::{
    ability::{
        Ability, Condition, Effect, ObjectSelector, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics, StaticAbility},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![bolt_bend(), mayhem_devil()]
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
