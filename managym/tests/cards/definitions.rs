//! Definition checks that the type system leaves open: a resolution-only
//! `Condition` (one that reads the resolving spell or ability) appears only
//! as the condition of an `Effect::If`, never in a static, an `ActiveIf` or an
//! intervening-if clause, where no resolution exists to ask.

use managym::{
    cardsets::catalog,
    state::{
        ability::{Ability, Condition, Effect, TriggerCondition},
        card::CardDefinition,
        predicate::CardPredicate,
    },
};
use serde_json::Value;

/// Paths (JSON pointers) of resolution-only conditions outside an
/// `Effect::If` condition.
fn misplaced_resolution_conditions(definition: &CardDefinition) -> Vec<String> {
    let value = serde_json::to_value(definition).expect("definitions serialize");
    let mut found = Vec::new();
    walk(&value, String::new(), false, &mut found);
    found
}

fn walk(value: &Value, path: String, in_if_condition: bool, found: &mut Vec<String>) {
    match value {
        Value::String(name) if matches!(name.as_str(), "Kicked" | "TargetDealtDamageThisTurn") => {
            if !in_if_condition {
                found.push(path);
            }
        }
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                walk(item, format!("{path}/{index}"), in_if_condition, found);
            }
        }
        Value::Object(fields) => {
            for (key, field) in fields {
                let child = format!("{path}/{key}");
                if key == "TargetMatches" && !in_if_condition {
                    found.push(child.clone());
                }
                match (key.as_str(), field) {
                    ("If", Value::Object(branch)) => {
                        for (part, inner) in branch {
                            walk(inner, format!("{child}/{part}"), part == "condition", found);
                        }
                    }
                    _ => walk(field, child, in_if_condition, found),
                }
            }
        }
        _ => {}
    }
}

#[test]
fn resolution_only_conditions_appear_only_in_effect_if() {
    for definition in catalog() {
        let misplaced = misplaced_resolution_conditions(definition);
        assert!(misplaced.is_empty(), "{}: {misplaced:?}", definition.key);
    }
}

#[test]
fn the_check_rejects_a_resolution_only_intervening_if_and_active_if() {
    let allowed = CardDefinition {
        name: "Witness".into(),
        spell_effects: vec![Effect::If {
            condition: Condition::AnyOf(vec![
                Condition::Kicked,
                Condition::TargetMatches(CardPredicate::creature()),
            ]),
            then: vec![Effect::Discard],
            otherwise: vec![],
        }],
        ..Default::default()
    };
    assert!(misplaced_resolution_conditions(&allowed).is_empty());

    let intervening = CardDefinition {
        name: "Witness".into(),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::BeginningOfYourEndStep,
            intervening_if: Some(Condition::TargetDealtDamageThisTurn),
            effects: vec![Effect::Discard],
        }],
        ..Default::default()
    };
    assert_eq!(misplaced_resolution_conditions(&intervening).len(), 1);

    let active = CardDefinition {
        name: "Witness".into(),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::ActiveIf {
                active_if: Condition::Kicked,
                condition: Box::new(TriggerCondition::BeginningOfYourEndStep),
            },
            intervening_if: None,
            effects: vec![Effect::Discard],
        }],
        ..Default::default()
    };
    assert_eq!(misplaced_resolution_conditions(&active).len(), 1);
}
