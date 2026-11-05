//! Definition checks that the type system leaves open: a resolution-only
//! `Condition` (one that reads the resolving spell or ability) appears only
//! as the condition of an `Effect::If`, never in a static, an `ActiveIf` or an
//! intervening-if clause, where no resolution exists to ask. A source
//! question (one that reads the ability's source permanent) may also be an
//! intervening-if or the "as long as" of a static the source projects on
//! itself (a type grant, a land-type overwrite), never a scoped static's
//! condition or an `ActiveIf`.

use managym::state::{
    ability::{Ability, Condition, Effect, TriggerCondition},
    card::{CardDefinition, CardType, StaticAbility, StaticPtBuff, StaticScope, StaticTypeGrant},
    permanent::CounterKind,
    predicate::CardPredicate,
    subtype::subtype_category,
};
use managym_cards::catalog;
use serde_json::Value;

/// Where a condition sits: an `Effect::If` condition may ask anything; an
/// intervening-if and a source-projected static's "as long as" may also ask
/// about the source.
#[derive(Clone, Copy, PartialEq)]
enum Place {
    Elsewhere,
    InterveningIf,
    SourceStatic,
    IfCondition,
}

/// Paths (JSON pointers) of resolution-only conditions outside an
/// `Effect::If` condition, and of source questions outside an `Effect::If`
/// condition, an intervening-if clause or a type grant's / land-type
/// overwrite's own condition.
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
                    "SourceCountersAtLeast" | "namesake" => !source_allowed,
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
                    ("TypeGrant" | "SetsLandType", Value::Object(ability)) => {
                        for (part, inner) in ability {
                            let place = if part == "condition" {
                                Place::SourceStatic
                            } else {
                                place
                            };
                            walk(inner, format!("{child}/{part}"), place, found);
                        }
                    }
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
            then: vec![Effect::discard()],
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
            effects: vec![Effect::discard()],
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
            effects: vec![Effect::discard()],
        }],
        ..Default::default()
    };
    assert_eq!(misplaced_resolution_conditions(&active).len(), 1);

    let source_intervening = CardDefinition {
        name: "Witness".into(),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::BeginningOfYourEndStep,
            intervening_if: Some(Condition::SourceTapped),
            effects: vec![Effect::discard()],
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
            effects: vec![Effect::discard()],
        }],
        ..Default::default()
    };
    assert_eq!(misplaced_resolution_conditions(&source_active).len(), 1);

    let fire = Condition::SourceCountersAtLeast {
        kind: CounterKind::Fire,
        count: 3,
    };
    let source_type_grant = CardDefinition {
        name: "Witness".into(),
        statics: vec![StaticAbility::TypeGrant(StaticTypeGrant {
            scope: StaticScope::This,
            types: vec![CardType::Creature],
            subtypes: Vec::new(),
            condition: Some(fire.clone()),
        })],
        ..Default::default()
    };
    assert!(misplaced_resolution_conditions(&source_type_grant).is_empty());

    let source_buff = CardDefinition {
        name: "Witness".into(),
        statics: vec![StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::This,
            condition: Some(fire),
            power: 1,
            toughness: 1,
            per_creature_type_max: None,
            per_color_among: None,
            per_graveyard_matching: None,
        })],
        ..Default::default()
    };
    assert_eq!(misplaced_resolution_conditions(&source_buff).len(), 1);
}

/// Planning inventory is audited against the catalog, never used to admit a
/// card. Back faces and tokens exist but are not independent deck entries.
#[test]
fn tla_allocation_has_every_face_and_fixed_token_in_the_catalog() {
    use managym::cards::{content_pack::default_content_pack, world_pool, Eligibility};
    use std::collections::BTreeSet;
    let catalog = default_content_pack();
    let pool = world_pool();
    let mut identities = BTreeSet::new();
    let (mut faces, mut tokens, mut markers, mut excluded) = (0, 0, 0, 0);
    for line in include_str!("../../../../docs/rules/tla/allocation.jsonl").lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        let name = row["face_name"].as_str().unwrap();
        match row["kind"].as_str().unwrap() {
            "card_face" => {
                faces += 1;
                identities.insert(row["oracle_id"].as_str().unwrap().to_owned());
                let id = catalog
                    .definition_id(name)
                    .unwrap_or_else(|| panic!("undefined allocated face {name}"));
                let entry = pool
                    .definitions
                    .iter()
                    .find(|row| row.card_def_id == id)
                    .unwrap();
                assert_eq!(
                    entry.eligibility == Eligibility::Eligible,
                    row["face_index"] == 0,
                    "{name}"
                );
            }
            "token_template" if row["in_tla_closure"] == true => {
                tokens += 1;
                let id = catalog
                    .definition_id(name)
                    .unwrap_or_else(|| panic!("undefined allocated token {name}"));
                assert!(catalog.definition(id).unwrap().is_token, "{name}");
            }
            "copy_marker" => {
                markers += 1;
                assert_eq!(name, "Copy");
                assert!(catalog.definition_id(name).is_none());
            }
            "token_template" => {
                excluded += 1;
                assert_eq!(name, "Treasure");
                assert!(row["planned_pr"].is_null());
                assert_eq!(row["in_tla_closure"], false);
            }
            kind => panic!("unreviewed allocation kind {kind}"),
        }
    }
    assert_eq!(
        (identities.len(), faces, tokens, markers, excluded),
        (286, 294, 9, 1, 1)
    );
}

/// Every subtype a definition, token or modification names, as JSON strings
/// under the fields that hold subtypes.
fn named_subtypes(value: &Value, key: Option<&str>, found: &mut Vec<String>) {
    const SUBTYPE_FIELDS: [&str; 6] = [
        "subtypes",
        "creature_types",
        "subtypes_any",
        "subtype",
        "AddSubtypes",
        "SetCreatureTypes",
    ];
    match value {
        Value::String(name) if key.is_some_and(|k| SUBTYPE_FIELDS.contains(&k)) => {
            found.push(name.clone())
        }
        Value::Array(items) => {
            for item in items {
                named_subtypes(item, key, found);
            }
        }
        Value::Object(fields) => {
            for (field, item) in fields {
                named_subtypes(item, Some(field), found);
            }
        }
        _ => {}
    }
}

/// CR 205.3: the shared subtype table classifies every subtype the catalog
/// names, so no rule infers a creature type from an unlisted name.
#[test]
fn every_authored_subtype_has_a_category() {
    let mut found = Vec::new();
    for definition in catalog() {
        let value = serde_json::to_value(definition).expect("definitions serialize");
        named_subtypes(&value, None, &mut found);
    }
    found.sort();
    found.dedup();
    assert!(found.len() > 50, "the scan reaches the catalog's subtypes");
    let missing: Vec<_> = found
        .iter()
        .filter(|name| subtype_category(name).is_none())
        .collect();
    assert!(missing.is_empty(), "unclassified subtypes: {missing:?}");
}
