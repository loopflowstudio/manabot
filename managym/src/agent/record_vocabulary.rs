//! Every name the fact and shared-effect encoders hash into an id column.
//!
//! Fact rows (`observation_encoder::facts`) and shared-effect records
//! (`shared_effects`) carry open-vocabulary names as `name_value` hashes:
//! counter kinds, colors, subtypes, keywords, modifications, set names and
//! combat restrictions. The hash keeps the transport exact, but a model needs
//! the finite set of names to embed each one as its own row. This module
//! lists that set, each name with the learning token card text uses for the
//! same concept when there is one (a granted `flying` and a printed
//! `Keyword::flying`), so the policy can share one embedding between them.
//!
//! Names fall in three namespaces, because the same string may mean different
//! things in different columns: `token` (rules vocabulary), `name` (a set card
//! name) and `restriction` (a combat restriction's predicate description).
//!
//! Entry point: [`record_names`].

use crate::{
    agent::{ability_description::predicate_description, observation::MODIFICATION_NAMES},
    cards::{
        content_pack::default_content_pack,
        tokens::{name_value, TokenKind},
    },
    state::{
        card::Keyword, mana::Color, permanent::CounterKind, predicate::CardPredicate,
        subtype::subtypes,
    },
};

/// Which id column family a name belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecordNamespace {
    Token,
    Name,
    Restriction,
}

impl RecordNamespace {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Token => "token",
            Self::Name => "name",
            Self::Restriction => "restriction",
        }
    }
}

/// One encodable name: `value` is the hash an encoder writes; `token` is the
/// `(kind, value)` learning token card text uses for the same concept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordName {
    pub namespace: RecordNamespace,
    pub name: String,
    pub value: i32,
    pub token: Option<(u16, i32)>,
}

fn record(namespace: RecordNamespace, name: &str, token: Option<(TokenKind, &str)>) -> RecordName {
    RecordName {
        namespace,
        name: name.to_string(),
        value: name_value(name),
        token: token.map(|(kind, token)| (kind as u16, name_value(token))),
    }
}

/// A name card text serializes as the same enum variant.
fn variant(name: &str) -> RecordName {
    record(
        RecordNamespace::Token,
        name,
        Some((TokenKind::Variant, name)),
    )
}

/// Every name the encoders can hash, sorted and without duplicates.
pub fn record_names() -> Vec<RecordName> {
    use RecordNamespace::{Name, Restriction, Token};
    let mut out = Vec::new();
    for keyword in Keyword::ALL {
        // Effects name a keyword bare; card text serializes the enum variant.
        let printed = format!("Keyword::{}", keyword.name());
        out.push(record(
            Token,
            keyword.name(),
            Some((TokenKind::Variant, &printed)),
        ));
    }
    for kind in CounterKind::ALL {
        out.push(variant(&format!("CounterKind::{}", kind.name())));
    }
    for color in [
        Color::White,
        Color::Blue,
        Color::Black,
        Color::Red,
        Color::Green,
        Color::Colorless,
        Color::Generic,
    ] {
        out.push(variant(&format!("Color::{color:?}")));
    }
    for name in MODIFICATION_NAMES {
        out.push(variant(&format!("Modification::{name}")));
    }
    for subtype in subtypes() {
        out.push(record(Token, subtype, Some((TokenKind::Text, subtype))));
    }
    out.push(record(Token, "CopyEffect", None));
    let pack = default_content_pack();
    for (_, definition) in pack.definition_entries() {
        out.push(record(Name, &definition.name, None));
        let json = serde_json::to_value(definition).expect("serializable definition");
        for name in set_names(&json) {
            out.push(record(Name, &name, None));
        }
        for predicate in combat_restrictions(&json) {
            let description = predicate_description(&predicate, "creatures");
            out.push(record(Restriction, &description, None));
        }
    }
    out.sort_by(|a, b| (a.namespace, &a.name).cmp(&(b.namespace, &b.name)));
    out.dedup();
    out
}

/// Every JSON object nested in a serialized definition.
fn objects(value: &serde_json::Value) -> Vec<&serde_json::Map<String, serde_json::Value>> {
    let mut out = Vec::new();
    let mut stack = vec![value];
    while let Some(value) = stack.pop() {
        match value {
            serde_json::Value::Object(map) => {
                out.push(map);
                stack.extend(map.values());
            }
            serde_json::Value::Array(items) => stack.extend(items),
            _ => {}
        }
    }
    out
}

/// The names a definition's `SetCharacteristics` give other permanents (a
/// copy takes a definition's own name).
fn set_names(value: &serde_json::Value) -> Vec<String> {
    objects(value)
        .into_iter()
        .filter_map(|map| map.get("named")?.as_str().map(str::to_owned))
        .collect()
}

/// The attacker predicates of every `AdditionalCombatPhase` in a definition:
/// the only source of a combat's attacker restriction.
fn combat_restrictions(value: &serde_json::Value) -> Vec<CardPredicate> {
    objects(value)
        .into_iter()
        .filter_map(|map| map.get("AdditionalCombatPhase")?.get("attackers"))
        .map(|attackers| {
            serde_json::from_value(attackers.clone()).expect("serialized attacker predicate")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_names_are_unambiguous_within_a_namespace() {
        let names = record_names();
        let mut values = std::collections::BTreeMap::new();
        for name in &names {
            if let Some(previous) = values.insert((name.namespace, name.value), &name.name) {
                panic!("{previous:?} and {:?} share a hash", name.name);
            }
        }
        assert!(names
            .iter()
            .any(|name| name.namespace == RecordNamespace::Restriction));
        assert!(names
            .iter()
            .any(|name| name.namespace == RecordNamespace::Name && name.name == "Humble Merchant"));
    }
}
