//! Definition checks that the type system leaves open: a resolution-only
//! `Condition` (one that reads the resolving spell or ability) appears only
//! as the condition of an `Effect::If`, never in a static, an `ActiveIf` or an
//! intervening-if clause, where no resolution exists to ask. A source
//! question (one that reads the ability's source permanent) may also be an
//! intervening-if, never a static or an `ActiveIf`.

use managym::{
    cardsets::catalog,
    state::{
        ability::{Ability, Condition, Effect, TriggerCondition},
        card::CardDefinition,
        predicate::CardPredicate,
    },
};
use serde_json::Value;

/// Where a condition sits: an `Effect::If` condition may ask anything; an
/// intervening-if may also ask about its ability's source.
#[derive(Clone, Copy, PartialEq)]
enum Place {
    Elsewhere,
    InterveningIf,
    IfCondition,
}

/// Paths (JSON pointers) of resolution-only conditions outside an
/// `Effect::If` condition, and of source questions outside an `Effect::If`
/// condition or an intervening-if clause.
fn misplaced_resolution_conditions(definition: &CardDefinition) -> Vec<String> {
    let value = serde_json::to_value(definition).expect("definitions serialize");
    let mut found = Vec::new();
    walk(&value, String::new(), Place::Elsewhere, &mut found);
    found
}

fn walk(value: &Value, path: String, place: Place, found: &mut Vec<String>) {
    let source_allowed = place != Place::Elsewhere;
    match value {
        Value::String(name) if matches!(name.as_str(), "Kicked" | "TargetDealtDamageThisTurn") => {
            if place != Place::IfCondition {
                found.push(path);
            }
        }
        Value::String(name) if name == "SourceTapped" && !source_allowed => found.push(path),
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                walk(item, format!("{path}/{index}"), place, found);
            }
        }
        Value::Object(fields) => {
            for (key, field) in fields {
                let child = format!("{path}/{key}");
                let misplaced = match key.as_str() {
                    "TargetMatches" => place != Place::IfCondition,
                    "SourceCountersAtLeast" => !source_allowed,
                    _ => false,
                };
                if misplaced {
                    found.push(child.clone());
                }
                match (key.as_str(), field) {
                    ("If", Value::Object(branch)) => {
                        for (part, inner) in branch {
                            let place = if part == "condition" {
                                Place::IfCondition
                            } else {
                                place
                            };
                            walk(inner, format!("{child}/{part}"), place, found);
                        }
                    }
                    ("intervening_if", _) => walk(field, child, Place::InterveningIf, found),
                    _ => walk(field, child, place, found),
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

    let source_intervening = CardDefinition {
        name: "Witness".into(),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::BeginningOfYourEndStep,
            intervening_if: Some(Condition::SourceTapped),
            effects: vec![Effect::Discard],
        }],
        ..Default::default()
    };
    assert!(misplaced_resolution_conditions(&source_intervening).is_empty());

    let source_active = CardDefinition {
        name: "Witness".into(),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::ActiveIf {
                active_if: Condition::SourceTapped,
                condition: Box::new(TriggerCondition::BeginningOfYourEndStep),
            },
            intervening_if: None,
            effects: vec![Effect::Discard],
        }],
        ..Default::default()
    };
    assert_eq!(misplaced_resolution_conditions(&source_active).len(), 1);
}
