//! Fact and shared-effect encoders hash names in three namespaces: rules
//! tokens, card names and combat restrictions. [`record_names`] lists each
//! name and the program token whose embedding it shares, if any (for example,
//! granted `flying` and printed `Keyword::flying`).
//!
//! [`runtime_tokens`] registers symbols supplied by runtime projections,
//! including those aliases. Together with printed [`catalog_tokens`] they
//! form the policy's program vocabulary.

use std::{
    collections::{BTreeSet, HashSet},
    sync::LazyLock,
};

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
        tokens::{catalog_tokens, name_value, TokenKind},
    },
    state::{
        card::{Keyword, SourceQuality},
        mana::Color,
        permanent::CounterKind,
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

/// A named program token emitted by a runtime projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeToken {
    pub kind: TokenKind,
    pub name: String,
    pub value: i32,
}

#[derive(Default)]
struct Vocabulary {
    records: Vec<RecordName>,
    runtime: Vec<RuntimeToken>,
    content_names: BTreeSet<String>,
}

impl Vocabulary {
    fn record(&mut self, namespace: RecordNamespace, name: &str, token: Option<(TokenKind, &str)>) {
        let token = token.map(|(kind, name)| {
            let token = runtime_token(kind, name);
            let key = (kind as u16, token.value);
            self.runtime.push(token);
            key
        });
        self.records.push(RecordName {
            namespace,
            name: name.to_owned(),
            value: name_value(name),
            token,
        });
    }

    // A copied or renamed permanent can emit its name both as a fact id
    // and as text in a last-known characteristics program.
    fn content_name(&mut self, name: &str) {
        if self.content_names.insert(name.to_owned()) {
            self.record(RecordNamespace::Name, name, None);
            self.runtime.push(runtime_token(TokenKind::Text, name));
        }
    }

    fn variant(&mut self, name: &str) {
        self.record(
            RecordNamespace::Token,
            name,
            Some((TokenKind::Variant, name)),
        );
    }
}

fn runtime_token(kind: TokenKind, name: &str) -> RuntimeToken {
    RuntimeToken {
        kind,
        name: name.to_owned(),
        value: name_value(name),
    }
}

/// Hashed names in fact and shared-effect rows, sorted by namespace and name.
pub fn record_names() -> Vec<RecordName> {
    VOCABULARY.records.clone()
}

/// Runtime program symbols, including record aliases, sorted by kind and name.
pub fn runtime_tokens() -> Vec<RuntimeToken> {
    VOCABULARY.runtime.clone()
}

/// Pack card and set names that also had synthetic record keys before 0.3.0.
pub(crate) fn content_names() -> &'static BTreeSet<String> {
    &VOCABULARY.content_names
}

static VOCABULARY: LazyLock<Vocabulary> = LazyLock::new(|| {
    use RecordNamespace::Token;
    let mut out = Vocabulary::default();
    for keyword in Keyword::ALL {
        // Effects name a keyword bare; card text serializes the enum variant.
        let printed = format!("Keyword::{}", keyword.name());
        out.record(Token, keyword.name(), Some((TokenKind::Variant, &printed)));
    }
    for kind in CounterKind::ALL {
        out.variant(&format!("CounterKind::{}", kind.name()));
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
        out.variant(&format!("Color::{color:?}"));
    }
    for name in MODIFICATION_NAMES {
        out.variant(&format!("Modification::{name}"));
    }
    for subtype in subtypes() {
        out.record(Token, subtype, Some((TokenKind::Text, subtype)));
    }
    out.record(Token, "CopyEffect", None);
    out.record(Token, POOLED_MANA, None);
    out.variant("StaticAbility::HexproofFrom");
    out.variant("StaticAbility::Protection");
    for quality in SourceQuality::ALL {
        out.variant(&quality.token_name());
    }
    out.record(Token, WARD, None);
    out.record(Token, PLAY_GRANT, None);
    out.record(Token, PLAY_GRANT_SOURCE, None);
    out.record(Token, SPENT_TYPE, None);
    let pack = default_content_pack();
    for candidate in pack.name_candidates().iter() {
        out.content_name(&candidate.name);
    }
    for (_, definition) in pack.definition_entries() {
        out.content_name(&definition.name);
        let json = serde_json::to_value(definition).expect("serializable definition");
        register_definition_rules(&json, &mut out);
    }
    register_runtime_tokens(&mut out.runtime);
    out.records
        .sort_by(|a, b| (a.namespace, &a.name).cmp(&(b.namespace, &b.name)));
    out.records.dedup();
    out.runtime
        .sort_by(|a, b| (a.kind as u16, &a.name).cmp(&(b.kind as u16, &b.name)));
    out.runtime.dedup();
    out
});

/// Printed and runtime program pairs admitted by a catalog policy.
pub(crate) static PROGRAM_VOCABULARY: LazyLock<HashSet<(u16, i32)>> = LazyLock::new(|| {
    let catalog = catalog_tokens(default_content_pack()).expect("default token catalog");
    catalog
        .token_kind
        .into_iter()
        .zip(catalog.token_value)
        .chain(
            VOCABULARY
                .runtime
                .iter()
                .map(|token| (token.kind as u16, token.value)),
        )
        .collect()
});

/// Distinct missing pairs in diagnostic order. Numeric literals carry their own value.
#[doc(hidden)]
pub fn missing_program_tokens(tokens: &[(u16, i32)]) -> BTreeSet<(u16, i32)> {
    let vocabulary = &*PROGRAM_VOCABULARY;
    tokens
        .iter()
        .copied()
        .filter(|token| !TokenKind::of(token.0).is_numeric() && !vocabulary.contains(token))
        .collect()
}

/// Runtime-only programs have no printed card span from which to discover
/// their vocabulary. Register their serde names and projection labels explicitly.
fn register_runtime_tokens(out: &mut Vec<RuntimeToken>) {
    let mut add = |kind, name: &str| out.push(runtime_token(kind, name));
    for name in EXECUTION_TEXT {
        add(TokenKind::Text, name);
    }
    add(TokenKind::Variant, "CostComponent::Discard");
    // Variable loyalty costs are supported before a printed ability uses one.
    add(TokenKind::Variant, "LoyaltyCost::MinusX");
    // These modifications are record names. Their bodies also emit closing tokens.
    add(TokenKind::Structure, "/Modification::AddTypes");
    add(TokenKind::Structure, "/Modification::AddColors");
    add(TokenKind::Structure, "/Modification::ModeChosen");
    add(TokenKind::Variant, "DamageKind::Combat");
    // A changeling's last known characteristics (CR 702.73a) derive every
    // creature type, which no printed definition serializes.
    add(TokenKind::Text, "all");
    add(TokenKind::Variant, "DamageRecipientFilter::Any");
    for name in [
        "PlayGrantExtension",
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
        ("DamageTransform", &["Prevent", "PreventAll"][..]),
        ("TriggerCondition", &["TappedForMana"][..]),
        // Offered only by a resolved shield record, never printed.
        (
            "ReplacementEffect",
            &["PreventCombatDamage", "AdventureExile"][..],
        ),
        ("EventCause", &["Effect", "Cost", "TurnAction"][..]),
        (
            "EventBinding",
            &[
                "None",
                "ThisWay",
                "Discard",
                "Countered",
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
                "SacrificedBy",
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
        ("ManaExpiry", &["Step", "EndOfCombat", "EndOfTurn"][..]),
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
                "GrantPlay",
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
        "remaining",
        "by",
        "index",
        "mode",
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
        "expiry",
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

/// Rules at any depth can set a name, condition a play grant or restrict
/// attackers in an additional combat phase.
fn register_definition_rules(definition: &serde_json::Value, out: &mut Vocabulary) {
    use serde_json::Value;

    let mut stack = vec![definition];
    while let Some(value) = stack.pop() {
        match value {
            Value::Object(map) => {
                if let Some(name) = map
                    .get("named")
                    .or_else(|| map.get("SetName"))
                    .and_then(Value::as_str)
                {
                    out.content_name(name);
                }
                if let Some(condition) = map
                    .get("GrantPlay")
                    .and_then(|grant| grant.get("condition"))
                {
                    let condition =
                        serde_json::from_value(condition.clone()).expect("serialized condition");
                    out.runtime.push(runtime_token(
                        TokenKind::Text,
                        &condition_text("permanent", &condition),
                    ));
                }
                if let Some(attackers) = map
                    .get("AdditionalCombatPhase")
                    .and_then(|phase| phase.get("attackers"))
                {
                    let predicate = serde_json::from_value(attackers.clone())
                        .expect("serialized attacker predicate");
                    out.record(
                        RecordNamespace::Restriction,
                        &predicate_description(&predicate, "creatures"),
                        None,
                    );
                }
                stack.extend(map.values());
            }
            Value::Array(items) => stack.extend(items),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn play_grant_conditions_are_found_at_any_depth() {
        use crate::state::{
            ability::{Condition, Effect, PlayWindow},
            predicate::CardPredicate,
        };
        let wizard = Condition::ControlsAtLeast {
            distinct_names: false,
            count: 1,
            predicate: CardPredicate::default(),
        };
        let nested = Effect::If {
            condition: Condition::YourTurn,
            then: vec![Effect::GrantPlay {
                what: crate::state::ability::ObjectSelector::Moved,
                until: PlayWindow::WhileExiled,
                play_lands: true,
                flashback: false,
                face_down: true,
                condition: Some(wizard.clone()),
            }],
            otherwise: Effect::exile_top_to_play(1, PlayWindow::ThisTurn),
        };
        let json = serde_json::json!({ "abilities": [{ "effects": [nested] }] });
        let mut out = Vocabulary::default();
        register_definition_rules(&json, &mut out);
        assert_eq!(
            out.runtime,
            [runtime_token(
                TokenKind::Text,
                &condition_text("permanent", &wizard)
            )]
        );
    }

    #[test]
    fn runtime_only_programs_have_catalog_tokens() {
        use crate::{
            cards::tokens::rules_tokens,
            flow::{
                event::GameEvent,
                game::CastChoices,
                payment::CostComponent,
                proposed_event::{CounterProvenance, EventBinding, EventCause},
                turn::Bending,
            },
            state::{ability::LibraryEnd, game_object::PlayerId},
        };
        fn check(value: &impl serde::Serialize) {
            let tokens = rules_tokens(value, default_content_pack()).unwrap();
            let missing = missing_program_tokens(&tokens);
            assert!(
                missing.is_empty(),
                "unregistered runtime tokens: {missing:?}"
            );
        }

        // Effect programs are covered by the cards crate's exhaustive witnesses.
        check(&[
            GameEvent::Scried {
                player: PlayerId(0),
            },
            GameEvent::Surveilled {
                player: PlayerId(0),
            },
            GameEvent::LibrarySearched {
                player: PlayerId(0),
            },
        ]);
        check(&[
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
        ]);
        check(&CastChoices {
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
        });
        for profile in [SpellProfile::Omen, SpellProfile::BackFace] {
            check(&CastChoices {
                profile,
                ..Default::default()
            });
        }
        for cause in [EventCause::Effect, EventCause::Cost, EventCause::TurnAction] {
            check(&CounterProvenance::new(cause));
        }
        check(&[LibraryEnd::Top, LibraryEnd::Bottom]);
        check(&CostComponent::Discard);
        check(&crate::state::subtype::CreatureTypes::All);
    }

    #[test]
    fn record_names_are_unambiguous_within_a_namespace() {
        let names = record_names();
        let memory_names = crate::agent::decision_memory::decision_memory_names();
        let mut values = std::collections::BTreeMap::new();
        for name in &names {
            if let Some(previous) = values.insert((name.namespace, name.value), &name.name) {
                panic!("{previous:?} and {:?} share a hash", name.name);
            }
        }
        for (name, value) in &memory_names {
            if let Some(previous) = values.insert((RecordNamespace::Token, *value), name) {
                panic!("{previous:?} and {name:?} share a hash");
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
