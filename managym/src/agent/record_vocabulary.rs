//! Every name the fact and shared-effect encoders hash into an id column.
//!
//! Fact rows (`observation_encoder::facts`) and shared-effect records
//! (`shared_effects`) and retained execution programs (`effect_execution`)
//! carry open-vocabulary names as `name_value` hashes:
//! counter kinds, colors, subtypes, keywords, modifications, set names and
//! combat restrictions, projection labels and runtime-only rules payloads. The hash keeps the transport exact, but a model needs
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

#[cfg(test)]
use crate::state::card::SpellProfile;
use crate::{
    agent::{
        ability_description::{condition_text, predicate_description},
        effect_execution::EXECUTION_TEXT,
        observation::MODIFICATION_NAMES,
        observation_encoder::{PLAY_GRANT, PLAY_GRANT_SOURCE, POOLED_MANA, SPENT_TYPE, WARD},
    },
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
    out.extend(
        crate::agent::decision_memory::RECORD_NAMES
            .iter()
            .map(|name| variant(&format!("DecisionMemory::{name}"))),
    );
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
    out.push(record(Token, POOLED_MANA, None));
    out.push(record(Token, WARD, None));
    out.push(record(Token, PLAY_GRANT, None));
    out.push(record(Token, PLAY_GRANT_SOURCE, None));
    out.push(record(Token, SPENT_TYPE, None));
    let pack = default_content_pack();
    for candidate in pack.name_candidates().iter() {
        out.push(record(Name, &candidate.name, None));
    }
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
    execution_names(&mut out);
    out.sort_by(|a, b| (a.namespace, &a.name).cmp(&(b.namespace, &b.name)));
    out.dedup();
    out
}

/// Runtime-only programs have no printed card span from which to discover
/// their vocabulary. Register their serde names and projection labels explicitly.
fn execution_names(out: &mut Vec<RecordName>) {
    let mut add = |kind: TokenKind, name: &str| {
        let token = (kind as u16, name_value(name));
        if !out.iter().any(|entry| entry.token == Some(token)) {
            out.push(record(
                RecordNamespace::Token,
                &format!("execution/{}/{name}", kind as u16),
                Some((kind, name)),
            ));
        }
    };
    for candidate in default_content_pack().name_candidates().iter() {
        add(TokenKind::Text, &candidate.name);
    }
    for name in EXECUTION_TEXT {
        add(TokenKind::Text, name);
    }
    add(TokenKind::Variant, "CostComponent::Discard");
    // A departed permanent's last known name (a token's, or one an effect
    // set) and a play grant's condition reach fact programs as rules text.
    for (_, definition) in default_content_pack().definition_entries() {
        add(TokenKind::Text, &definition.name);
        let json = serde_json::to_value(definition).expect("serializable definition");
        for name in set_names(&json) {
            add(TokenKind::Text, &name);
        }
        for condition in play_grant_conditions(&json) {
            add(TokenKind::Text, &condition_text("permanent", &condition));
        }
    }
    add(TokenKind::Structure, "PlayGrantExtension");
    add(TokenKind::Structure, "/PlayGrantExtension");
    for name in [
        "PlayerId",
        "CardDefId",
        "ZoneSelection",
        "CounterProvenance",
        "CastChoices",
        "CopyEffect",
        "EffectiveTrigger",
        "ManaSpent",
        "Help",
    ] {
        add(TokenKind::Structure, name);
        add(TokenKind::Structure, &format!("/{name}"));
    }
    for (owner, variants) in [
        (
            "GameEvent",
            &["Scried", "Surveilled", "LibrarySearched"][..],
        ),
        ("SpellProfile", &["Omen", "BackFace"][..]),
        ("LibraryEnd", &["Top", "Bottom"][..]),
        // Supported stack-ability targeting has no printed card program yet.
        ("TargetSpec", &["Ability"][..]),
        // Retained execution can refer to a spell already on the stack;
        // printed card programs do not supply this zone's token.
        ("ZoneType", &["Stack"][..]),
        ("DamageTransform", &["Prevent"][..]),
        // Offered only by a resolved shield record, never printed.
        (
            "ReplacementEffect",
            &[
                "PreventCombatDamage",
                "AdventureExile",
                "ExileInsteadOfGraveyard",
            ][..],
        ),
        ("EventCause", &["Effect", "Cost", "TurnAction"][..]),
        (
            "EventBinding",
            &[
                "None",
                "ThisWay",
                "Destroyed",
                "OwnSacrifices",
                "LifeLost",
                "Damage",
                "Deaths",
                "Combat",
                "PlaceInLibrary",
                "Shuffle",
                "Bending",
                "Created",
                "ExileLinked",
            ][..],
        ),
        ("Bending", &["Water", "Earth", "Fire", "Air"][..]),
        // What a payment made during resolution still owes.
        ("CostComponent", &["Mana", "Life"][..]),
        // What a payment's contributing objects did and paid for.
        (
            "How",
            &["Tapped", "Sacrificed", "Exiled", "Discarded", "Returned"][..],
        ),
        (
            "Toward",
            &[
                "Printed",
                "Waterbend",
                "Crew",
                "Alternative",
                "Convoke",
                "Improvise",
                "Delve",
                "Bargain",
                "Forage",
                "Saddle",
                "Station",
            ][..],
        ),
        ("EntryChoice", &["Color", "CreatureType", "CardName"][..]),
        (
            "SpendPredicate",
            &["CastCreatureSubtype", "ActivateCreatureSubtype"][..],
        ),
        (
            "Effect",
            &[
                "PlaceTargetInLibrary",
                "RestoreController",
                "SelectTargetRequirement",
                "HaveAbilitiesOf",
                "ExileNamed",
                "CounterAbility",
            ][..],
        ),
    ] {
        for variant in variants {
            let name = format!("{owner}::{variant}");
            add(TokenKind::Variant, &name);
            add(TokenKind::Structure, &format!("/{name}"));
        }
    }
    for field in [
        "player",
        "owner",
        "zone",
        "predicate",
        "cause",
        "replaced",
        "result",
        "second_from_top",
        "index",
        "kind",
        "attacking",
        "kicked",
        "flash_surcharge",
        "flashback",
        "from_prepared",
        "exile_instead_of_graveyard",
        "x",
        "free",
        "resolving",
        "ability",
        "program",
        "amounts",
        "entry_counters",
        "gains",
        "copies",
        "combat",
        "definition",
        "name",
        "cost",
        "help",
        "waterbend",
        "actor",
        "grantee",
        "source",
        "prepared",
        "face_down",
        "play_lands",
        "condition",
    ] {
        add(TokenKind::Field, field);
    }
}

/// Every condition a serialized definition puts on the cards it lets you
/// play (Flameshape's "you control a Wizard"), wherever the effect sits:
/// a spell, an inset, a trigger, an activation or an `If` branch.
fn play_grant_conditions(value: &serde_json::Value) -> Vec<crate::state::ability::Condition> {
    objects(value)
        .into_iter()
        .filter_map(|map| map.get("ExileTopCardsToPlay")?.get("condition"))
        .map(|condition| serde_json::from_value(condition.clone()).expect("serialized condition"))
        .collect()
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

/// The names a definition's `SetCharacteristics` or `Modification::SetName`
/// give other permanents (a copy takes a definition's own name).
fn set_names(value: &serde_json::Value) -> Vec<String> {
    objects(value)
        .into_iter()
        // `SetCharacteristics::named` (Honest Work) and `Modification::SetName`.
        .filter_map(|map| {
            map.get("named")
                .or_else(|| map.get("SetName"))?
                .as_str()
                .map(str::to_owned)
        })
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
    fn play_grant_conditions_are_found_at_any_depth() {
        use crate::state::ability::{Condition, Effect, PlayWindow};
        let wizard = Condition::ControlsAtLeast {
            distinct_names: false,
            count: 1,
            predicate: CardPredicate::default(),
        };
        let nested = Effect::If {
            condition: Condition::YourTurn,
            then: vec![Effect::ExileTopCardsToPlay {
                count: 2,
                until: PlayWindow::WhileExiled,
                face_down: true,
                condition: Some(wizard.clone()),
            }],
            otherwise: vec![Effect::exile_top_to_play(1, PlayWindow::ThisTurn)],
        };
        let json = serde_json::json!({ "abilities": [{ "effects": [nested] }] });
        assert_eq!(play_grant_conditions(&json), [wizard]);
    }

    #[test]
    fn runtime_only_programs_have_catalog_tokens() {
        use crate::{
            cards::tokens::{catalog_tokens, rules_tokens},
            flow::{
                event::GameEvent,
                game::CastChoices,
                payment::CostComponent,
                proposed_event::{CounterProvenance, EventBinding, EventCause},
                turn::Bending,
            },
            state::{ability::LibraryEnd, game_object::PlayerId},
        };
        let pack = default_content_pack();
        let catalog = catalog_tokens(pack).unwrap();
        let known: std::collections::HashSet<_> = catalog
            .token_kind
            .into_iter()
            .zip(catalog.token_value)
            .chain(record_names().into_iter().filter_map(|entry| entry.token))
            .collect();
        // Effect programs are covered by the cards crate's exhaustive witnesses.
        let mut tokens = rules_tokens(
            &[
                GameEvent::Scried {
                    player: PlayerId(0),
                },
                GameEvent::Surveilled {
                    player: PlayerId(0),
                },
                GameEvent::LibrarySearched {
                    player: PlayerId(0),
                },
            ],
            pack,
        )
        .unwrap();
        tokens.extend(
            rules_tokens(
                &[
                    EventBinding::None,
                    EventBinding::ThisWay,
                    EventBinding::Destroyed,
                    EventBinding::OwnSacrifices,
                    EventBinding::LifeLost,
                    EventBinding::Deaths,
                    EventBinding::Combat,
                    EventBinding::Damage { result: Some(1) },
                    EventBinding::PlaceInLibrary {
                        second_from_top: true,
                    },
                    EventBinding::Shuffle {
                        player: PlayerId(0),
                    },
                    EventBinding::Bending {
                        player: PlayerId(0),
                        kind: Bending::Earth,
                    },
                    EventBinding::Created { attacking: true },
                    EventBinding::ExileLinked {
                        actor: PlayerId(0),
                        grantee: PlayerId(1),
                        source: None,
                        grant: crate::state::ability::LinkedGrant::FlashAnyMana,
                    },
                ],
                pack,
            )
            .unwrap(),
        );
        tokens.extend(
            rules_tokens(
                &CastChoices {
                    profile: SpellProfile::Adventure,
                    exile_instead_of_graveyard: true,
                    route: None,
                    kicked: true,
                    flash_surcharge: true,
                    flashback: true,
                    from_prepared: true,
                    x: 1,
                    free: true,
                    resolving: true,
                },
                pack,
            )
            .unwrap(),
        );
        for profile in [SpellProfile::Omen, SpellProfile::BackFace] {
            tokens.extend(
                rules_tokens(
                    &CastChoices {
                        profile,
                        ..Default::default()
                    },
                    pack,
                )
                .unwrap(),
            );
        }
        for cause in [EventCause::Effect, EventCause::Cost, EventCause::TurnAction] {
            tokens.extend(rules_tokens(&CounterProvenance::new(cause), pack).unwrap());
        }
        tokens.extend(rules_tokens(&[LibraryEnd::Top, LibraryEnd::Bottom], pack).unwrap());
        tokens.extend(rules_tokens(&CostComponent::Discard, pack).unwrap());
        for (kind, value) in tokens {
            assert!(
                matches!(kind, 4 | 5) || known.contains(&(kind, value)),
                "unregistered runtime token: kind {kind}, value {value}"
            );
        }
    }

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
