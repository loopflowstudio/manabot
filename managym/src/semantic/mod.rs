//! Generic interpreter over the checked-in two-deck typed IR.
//!
//! This module is the runtime consumer proof for W2-223. It parses the
//! versioned, reviewed IR emitted by `scripts/compile_semantic_content.py`,
//! binds every semantic definition **once** to a real [`ContentPack`]
//! [`CardDefId`], and then executes admitted card programs by dispatching purely
//! on the stable numeric [`Opcode`]. No step in this module inspects a card
//! name, registry name, or oracle string: the parser rejects any name-bearing
//! instruction field, and execution is a `match` over typed steps.
//!
//! The interpreter is deliberately additive. It does not replace the live
//! effect resolver in [`crate::flow`]; it consumes the same reviewed programs
//! and yields a deterministic [`TraceEvent`] sequence that tests assert against.
//! Differential parity between this trace and the legacy effect path remains a
//! documented future step (see `content/semantic/README.md`).

use std::collections::BTreeSet;

use serde_json::Value;

use crate::cardsets::alpha::ContentPack;
use crate::state::card::CardDefId;

/// Canonical, checked-in IR document. Kept in sync by the offline compiler's
/// `--check` mode; this crate only ever reads it.
const TWO_DECK_IR: &str = include_str!("../../../content/semantic/v1/generated/two_deck.ir.json");

/// Stable numeric opcodes. These mirror `manabot.semantic.compiler.Opcode` and
/// are the *only* dispatch key the interpreter uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u64)]
pub enum Opcode {
    AddMana = 1,
    DrawCards = 2,
    PutTopCardsInHand = 3,
    CreateToken = 4,
    PutCounters = 5,
    GainLife = 6,
    Scry = 7,
    Tap = 8,
    Untap = 9,
    ModifyPt = 10,
    GrantKeywords = 11,
    RestrictBlocking = 12,
    DealDamage = 13,
    ReturnToHand = 14,
    Learn = 15,
    CounterUnlessPays = 16,
    LookAndSelect = 17,
    Branch = 18,
    ForEachTarget = 19,
    Earthbend = 20,
    ExileUntilSourceLeaves = 21,
    DealPowerDamage = 22,
    SetPowerFromCount = 23,
}

impl Opcode {
    fn from_u64(value: u64) -> Result<Self, IrError> {
        let opcode = match value {
            1 => Opcode::AddMana,
            2 => Opcode::DrawCards,
            3 => Opcode::PutTopCardsInHand,
            4 => Opcode::CreateToken,
            5 => Opcode::PutCounters,
            6 => Opcode::GainLife,
            7 => Opcode::Scry,
            8 => Opcode::Tap,
            9 => Opcode::Untap,
            10 => Opcode::ModifyPt,
            11 => Opcode::GrantKeywords,
            12 => Opcode::RestrictBlocking,
            13 => Opcode::DealDamage,
            14 => Opcode::ReturnToHand,
            15 => Opcode::Learn,
            16 => Opcode::CounterUnlessPays,
            17 => Opcode::LookAndSelect,
            18 => Opcode::Branch,
            19 => Opcode::ForEachTarget,
            20 => Opcode::Earthbend,
            21 => Opcode::ExileUntilSourceLeaves,
            22 => Opcode::DealPowerDamage,
            23 => Opcode::SetPowerFromCount,
            other => return Err(IrError::UnknownOpcode(other)),
        };
        Ok(opcode)
    }
}

/// Error raised while parsing or binding the checked-in IR. Every variant is a
/// *fail-closed* signal: a missing admission or a name-bearing branch stops the
/// interpreter rather than silently degrading to card-specific behaviour.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IrError {
    /// A definition's `content_pack_binding` did not resolve to a real card.
    UnadmittedDefinition {
        semantic_key: String,
        registry_name: String,
    },
    /// An instruction carried a card name / registry name / oracle string.
    NameBasedDispatch { field: String },
    /// The opcode number is outside the known typed set.
    UnknownOpcode(u64),
    /// A structural expectation about the IR document was violated.
    Malformed(String),
}

impl std::fmt::Display for IrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IrError::UnadmittedDefinition {
                semantic_key,
                registry_name,
            } => write!(
                f,
                "definition {semantic_key} ({registry_name}) is not admitted by the content pack"
            ),
            IrError::NameBasedDispatch { field } => {
                write!(f, "instruction carries name-based dispatch field {field:?}")
            }
            IrError::UnknownOpcode(value) => write!(f, "unknown opcode {value}"),
            IrError::Malformed(message) => write!(f, "malformed IR: {message}"),
        }
    }
}

impl std::error::Error for IrError {}

/// Fields whose presence on an instruction would mean the interpreter is
/// dispatching on card identity rather than typed structure. Parsing rejects
/// them so a regression cannot re-introduce name-based branches.
const FORBIDDEN_FIELDS: [&str; 3] = ["card_name", "registry_name", "definition_ref"];

/// A reviewed selection predicate over game objects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Predicate {
    Any,
    All(Vec<Predicate>),
    CardType(String),
    Subtype(String),
    NotCardTypes(Vec<String>),
    PowerAtMost(i64),
}

/// A reviewed runtime condition gating a [`Step::Branch`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Condition {
    Kicked,
    NthResolution(u64),
    GraveyardAtLeast { count: u64, predicate: Predicate },
    TargetMatches { role: String, predicate: Predicate },
}

/// One typed, executable IR step. Nested arms (`Branch`, `ForEachTarget`) carry
/// their own step lists, matching the compiler's `then`/`otherwise`/`body`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    AddMana {
        mana: String,
        until: Option<String>,
    },
    DrawCards {
        count: i64,
    },
    PutTopCardsInHand {
        count: i64,
    },
    CreateToken {
        definition_index: usize,
        count: i64,
        tapped_and_attacking: bool,
    },
    PutCounters {
        target: String,
        count: i64,
    },
    GainLife {
        amount: i64,
    },
    Scry {
        count: i64,
    },
    Tap {
        target: String,
    },
    Untap {
        target: String,
    },
    ModifyPt {
        target: String,
        power: i64,
        toughness: i64,
        duration: String,
    },
    GrantKeywords {
        target: String,
        keywords: Vec<String>,
        duration: String,
    },
    RestrictBlocking {
        target: String,
        duration: String,
    },
    DealDamage {
        amount: i64,
        target: String,
    },
    ReturnToHand {
        target: String,
    },
    Learn,
    CounterUnlessPays {
        target: String,
        cost: String,
    },
    LookAndSelect {
        look: i64,
        min_select: i64,
        max_select: i64,
        destination: String,
    },
    Branch {
        condition: Condition,
        then: Vec<Step>,
        otherwise: Vec<Step>,
    },
    ForEachTarget {
        role: String,
        body: Vec<Step>,
    },
    Earthbend {
        target: String,
        count: i64,
    },
    ExileUntilSourceLeaves {
        target: String,
    },
    DealPowerDamage {
        sources: String,
        target: String,
    },
    SetPowerFromCount {
        zone: String,
        controller: String,
    },
}

/// One admitted card program: a labelled instruction list bound to a semantic
/// definition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub semantic_key: String,
    pub kind_name: String,
    pub definition_index: usize,
    pub steps: Vec<Step>,
}

/// One semantic definition and the reviewed registry name it binds through.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Definition {
    pub semantic_index: usize,
    pub semantic_key: String,
    pub registry_name: String,
}

/// The parsed IR document, independent of any content pack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticPack {
    pub schema_version: u64,
    pub pack_key: String,
    pub ir_hash: String,
    pub source_hash: String,
    pub definitions: Vec<Definition>,
    pub programs: Vec<Program>,
    /// Definition indexes referenced by the two admitted decks. Every one of
    /// these must bind for the slice to be considered fully admitted.
    pub deck_definition_indexes: BTreeSet<usize>,
}

/// A [`SemanticPack`] resolved once against a concrete [`ContentPack`]. After
/// binding, execution carries typed `CardDefId`s and opcodes only.
#[derive(Clone, Debug)]
pub struct BoundPack<'a> {
    pack: &'a SemanticPack,
    definition_ids: Vec<CardDefId>,
}

/// A single observable effect produced by executing a program. Trace tests
/// assert on these sequences.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceEvent {
    AddMana {
        mana: String,
        until: Option<String>,
    },
    DrawCards(i64),
    PutTopCardsInHand(i64),
    CreateToken {
        definition: CardDefId,
        count: i64,
        tapped_and_attacking: bool,
    },
    PutCounters {
        target: String,
        count: i64,
    },
    GainLife(i64),
    Scry(i64),
    Tap(String),
    Untap(String),
    ModifyPt {
        target: String,
        power: i64,
        toughness: i64,
        duration: String,
    },
    GrantKeywords {
        target: String,
        keywords: Vec<String>,
        duration: String,
    },
    RestrictBlocking {
        target: String,
        duration: String,
    },
    DealDamage {
        target: String,
        amount: i64,
    },
    ReturnToHand(String),
    Learn,
    CounterUnlessPays {
        target: String,
        cost: String,
    },
    LookAndSelect {
        look: i64,
        min_select: i64,
        max_select: i64,
        destination: String,
    },
    Earthbend {
        target: String,
        count: i64,
    },
    ExileUntilSourceLeaves(String),
    DealPowerDamage {
        sources: String,
        target: String,
    },
    SetPowerFromCount {
        zone: String,
        controller: String,
    },
}

/// Runtime facts an executing program may consult. Implementors supply the
/// match state a branch or per-target expansion depends on. The interpreter
/// never reaches into game state directly, which keeps it deterministic and
/// unit-testable.
pub trait InterpreterContext {
    fn kicked(&self) -> bool {
        false
    }

    fn nth_resolution(&self) -> u64 {
        1
    }

    fn graveyard_at_least(&self, _predicate: &Predicate, _count: u64) -> bool {
        false
    }

    fn target_matches(&self, _role: &str, _predicate: &Predicate) -> bool {
        false
    }

    /// Number of objects bound to a `for_each_target` role in this resolution.
    fn role_target_count(&self, _role: &str) -> usize {
        1
    }
}

/// A context in which nothing is true; useful for structural "does it run"
/// coverage where the branch/for-each shape is exercised through defaults.
pub struct DefaultContext;

impl InterpreterContext for DefaultContext {}

impl SemanticPack {
    /// Parse the process-wide checked-in two-deck IR.
    pub fn two_deck() -> Result<Self, IrError> {
        Self::from_json(TWO_DECK_IR)
    }

    /// Parse an IR document from JSON text.
    pub fn from_json(text: &str) -> Result<Self, IrError> {
        let value: Value =
            serde_json::from_str(text).map_err(|error| IrError::Malformed(error.to_string()))?;
        Self::from_value(&value)
    }

    fn from_value(value: &Value) -> Result<Self, IrError> {
        let schema_version = u64_field(value, "schema_version")?;
        let pack_key = str_field(value, "pack_key")?.to_owned();
        let ir_hash = str_field(value, "ir_hash")?.to_owned();
        let source_hash = str_field(value, "source_hash")?.to_owned();

        let definitions = array_field(value, "definitions")?
            .iter()
            .enumerate()
            .map(|(expected_index, definition)| parse_definition(definition, expected_index))
            .collect::<Result<Vec<_>, _>>()?;

        let programs = array_field(value, "programs")?
            .iter()
            .map(parse_program)
            .collect::<Result<Vec<_>, _>>()?;

        let mut deck_definition_indexes = BTreeSet::new();
        for deck in array_field(value, "decks")? {
            for card in array_field(deck, "cards")? {
                let index = usize_field(card, "definition_index")?;
                deck_definition_indexes.insert(index);
            }
        }

        Ok(SemanticPack {
            schema_version,
            pack_key,
            ir_hash,
            source_hash,
            definitions,
            programs,
            deck_definition_indexes,
        })
    }

    /// Bind every definition once to `pack`. Fails closed if any definition in
    /// the admitted decks does not resolve to a real card, so a dropped
    /// admission cannot pass silently.
    pub fn bind<'a>(&'a self, pack: &ContentPack) -> Result<BoundPack<'a>, IrError> {
        let mut definition_ids = Vec::with_capacity(self.definitions.len());
        let mut seen = BTreeSet::new();
        for definition in &self.definitions {
            let card_def_id = pack
                .definition_id(&definition.registry_name)
                .ok_or_else(|| IrError::UnadmittedDefinition {
                    semantic_key: definition.semantic_key.clone(),
                    registry_name: definition.registry_name.clone(),
                })?;
            if !seen.insert(card_def_id) {
                return Err(IrError::Malformed(format!(
                    "duplicate content-pack binding for {}",
                    definition.registry_name
                )));
            }
            definition_ids.push(card_def_id);
        }
        Ok(BoundPack {
            pack: self,
            definition_ids,
        })
    }
}

impl<'a> BoundPack<'a> {
    pub fn pack(&self) -> &SemanticPack {
        self.pack
    }

    /// The bound card id for a semantic definition index.
    pub fn definition_id(&self, semantic_index: usize) -> Option<CardDefId> {
        self.definition_ids.get(semantic_index).copied()
    }

    /// Execute a program against `context`, returning its resolved effect trace.
    /// Dispatch is a pure `match` over typed [`Step`]s keyed on the numeric
    /// opcode; no card identity is consulted.
    pub fn run(
        &self,
        program: &Program,
        context: &dyn InterpreterContext,
    ) -> Result<Vec<TraceEvent>, IrError> {
        let mut trace = Vec::new();
        self.run_steps(&program.steps, context, &mut trace)?;
        Ok(trace)
    }

    fn run_steps(
        &self,
        steps: &[Step],
        context: &dyn InterpreterContext,
        trace: &mut Vec<TraceEvent>,
    ) -> Result<(), IrError> {
        for step in steps {
            self.run_step(step, context, trace)?;
        }
        Ok(())
    }

    fn run_step(
        &self,
        step: &Step,
        context: &dyn InterpreterContext,
        trace: &mut Vec<TraceEvent>,
    ) -> Result<(), IrError> {
        match step {
            Step::AddMana { mana, until } => trace.push(TraceEvent::AddMana {
                mana: mana.clone(),
                until: until.clone(),
            }),
            Step::DrawCards { count } => trace.push(TraceEvent::DrawCards(*count)),
            Step::PutTopCardsInHand { count } => trace.push(TraceEvent::PutTopCardsInHand(*count)),
            Step::CreateToken {
                definition_index,
                count,
                tapped_and_attacking,
            } => {
                let definition = self.definition_id(*definition_index).ok_or_else(|| {
                    IrError::Malformed(format!(
                        "create_token references unbound definition index {definition_index}"
                    ))
                })?;
                trace.push(TraceEvent::CreateToken {
                    definition,
                    count: *count,
                    tapped_and_attacking: *tapped_and_attacking,
                });
            }
            Step::PutCounters { target, count } => trace.push(TraceEvent::PutCounters {
                target: target.clone(),
                count: *count,
            }),
            Step::GainLife { amount } => trace.push(TraceEvent::GainLife(*amount)),
            Step::Scry { count } => trace.push(TraceEvent::Scry(*count)),
            Step::Tap { target } => trace.push(TraceEvent::Tap(target.clone())),
            Step::Untap { target } => trace.push(TraceEvent::Untap(target.clone())),
            Step::ModifyPt {
                target,
                power,
                toughness,
                duration,
            } => trace.push(TraceEvent::ModifyPt {
                target: target.clone(),
                power: *power,
                toughness: *toughness,
                duration: duration.clone(),
            }),
            Step::GrantKeywords {
                target,
                keywords,
                duration,
            } => trace.push(TraceEvent::GrantKeywords {
                target: target.clone(),
                keywords: keywords.clone(),
                duration: duration.clone(),
            }),
            Step::RestrictBlocking { target, duration } => {
                trace.push(TraceEvent::RestrictBlocking {
                    target: target.clone(),
                    duration: duration.clone(),
                })
            }
            Step::DealDamage { amount, target } => trace.push(TraceEvent::DealDamage {
                target: target.clone(),
                amount: *amount,
            }),
            Step::ReturnToHand { target } => trace.push(TraceEvent::ReturnToHand(target.clone())),
            Step::Learn => trace.push(TraceEvent::Learn),
            Step::CounterUnlessPays { target, cost } => trace.push(TraceEvent::CounterUnlessPays {
                target: target.clone(),
                cost: cost.clone(),
            }),
            Step::LookAndSelect {
                look,
                min_select,
                max_select,
                destination,
            } => trace.push(TraceEvent::LookAndSelect {
                look: *look,
                min_select: *min_select,
                max_select: *max_select,
                destination: destination.clone(),
            }),
            Step::Branch {
                condition,
                then,
                otherwise,
            } => {
                let arm = if evaluate_condition(condition, context) {
                    then
                } else {
                    otherwise
                };
                self.run_steps(arm, context, trace)?;
            }
            Step::ForEachTarget { role, body } => {
                for _ in 0..context.role_target_count(role) {
                    self.run_steps(body, context, trace)?;
                }
            }
            Step::Earthbend { target, count } => trace.push(TraceEvent::Earthbend {
                target: target.clone(),
                count: *count,
            }),
            Step::ExileUntilSourceLeaves { target } => {
                trace.push(TraceEvent::ExileUntilSourceLeaves(target.clone()))
            }
            Step::DealPowerDamage { sources, target } => trace.push(TraceEvent::DealPowerDamage {
                sources: sources.clone(),
                target: target.clone(),
            }),
            Step::SetPowerFromCount { zone, controller } => {
                trace.push(TraceEvent::SetPowerFromCount {
                    zone: zone.clone(),
                    controller: controller.clone(),
                })
            }
        }
        Ok(())
    }
}

fn evaluate_condition(condition: &Condition, context: &dyn InterpreterContext) -> bool {
    match condition {
        Condition::Kicked => context.kicked(),
        Condition::NthResolution(n) => context.nth_resolution() == *n,
        Condition::GraveyardAtLeast { count, predicate } => {
            context.graveyard_at_least(predicate, *count)
        }
        Condition::TargetMatches { role, predicate } => context.target_matches(role, predicate),
    }
}

fn parse_definition(value: &Value, expected_index: usize) -> Result<Definition, IrError> {
    let semantic_index = usize_field(value, "semantic_index")?;
    if semantic_index != expected_index {
        return Err(IrError::Malformed(format!(
            "definition {semantic_index} is out of order (expected {expected_index})"
        )));
    }
    let semantic_key = str_field(value, "semantic_key")?.to_owned();
    let binding = object_field(value, "content_pack_binding")?;
    let kind = str_field(binding, "kind")?;
    if kind != "legacy_registry_name" {
        return Err(IrError::Malformed(format!(
            "definition {semantic_key} has unexpected binding kind {kind:?}"
        )));
    }
    let registry_name = str_field(binding, "value")?.to_owned();
    Ok(Definition {
        semantic_index,
        semantic_key,
        registry_name,
    })
}

fn parse_program(value: &Value) -> Result<Program, IrError> {
    let semantic_key = str_field(value, "semantic_key")?.to_owned();
    let kind_name = str_field(value, "kind_name")?.to_owned();
    let definition_index = usize_field(value, "definition_index")?;
    let steps = parse_steps(array_field(value, "instructions")?)?;
    Ok(Program {
        semantic_key,
        kind_name,
        definition_index,
        steps,
    })
}

fn parse_steps(values: &[Value]) -> Result<Vec<Step>, IrError> {
    values.iter().map(parse_step).collect()
}

fn parse_step(value: &Value) -> Result<Step, IrError> {
    let object = value
        .as_object()
        .ok_or_else(|| IrError::Malformed("instruction must be an object".to_owned()))?;
    for field in FORBIDDEN_FIELDS {
        if object.contains_key(field) {
            return Err(IrError::NameBasedDispatch {
                field: field.to_owned(),
            });
        }
    }
    let opcode = Opcode::from_u64(u64_field(value, "opcode")?)?;
    let step = match opcode {
        Opcode::AddMana => Step::AddMana {
            mana: str_field(value, "mana")?.to_owned(),
            until: opt_str_field(value, "until")?.map(str::to_owned),
        },
        Opcode::DrawCards => Step::DrawCards {
            count: i64_field(value, "count")?,
        },
        Opcode::PutTopCardsInHand => Step::PutTopCardsInHand {
            count: i64_field(value, "count")?,
        },
        Opcode::CreateToken => Step::CreateToken {
            definition_index: usize_field(value, "definition_index")?,
            count: i64_field(value, "count")?,
            tapped_and_attacking: opt_bool_field(value, "tapped_and_attacking")?.unwrap_or(false),
        },
        Opcode::PutCounters => Step::PutCounters {
            target: str_field(value, "target")?.to_owned(),
            count: i64_field(value, "count")?,
        },
        Opcode::GainLife => Step::GainLife {
            amount: i64_field(value, "amount")?,
        },
        Opcode::Scry => Step::Scry {
            count: i64_field(value, "count")?,
        },
        Opcode::Tap => Step::Tap {
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::Untap => Step::Untap {
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::ModifyPt => Step::ModifyPt {
            target: str_field(value, "target")?.to_owned(),
            power: i64_field(value, "power")?,
            toughness: i64_field(value, "toughness")?,
            duration: str_field(value, "duration")?.to_owned(),
        },
        Opcode::GrantKeywords => Step::GrantKeywords {
            target: str_field(value, "target")?.to_owned(),
            keywords: str_array_field(value, "keywords")?,
            duration: str_field(value, "duration")?.to_owned(),
        },
        Opcode::RestrictBlocking => Step::RestrictBlocking {
            target: str_field(value, "target")?.to_owned(),
            duration: str_field(value, "duration")?.to_owned(),
        },
        Opcode::DealDamage => Step::DealDamage {
            amount: i64_field(value, "amount")?,
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::ReturnToHand => Step::ReturnToHand {
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::Learn => Step::Learn,
        Opcode::CounterUnlessPays => Step::CounterUnlessPays {
            target: str_field(value, "target")?.to_owned(),
            cost: str_field(value, "cost")?.to_owned(),
        },
        Opcode::LookAndSelect => Step::LookAndSelect {
            look: i64_field(value, "look")?,
            min_select: i64_field(value, "min_select")?,
            max_select: i64_field(value, "max_select")?,
            destination: str_field(value, "destination")?.to_owned(),
        },
        Opcode::Branch => Step::Branch {
            condition: parse_condition(object_field(value, "condition")?)?,
            then: parse_steps(array_field(value, "then")?)?,
            otherwise: parse_steps(array_field(value, "otherwise")?)?,
        },
        Opcode::ForEachTarget => Step::ForEachTarget {
            role: str_field(value, "role")?.to_owned(),
            body: parse_steps(array_field(value, "body")?)?,
        },
        Opcode::Earthbend => Step::Earthbend {
            target: str_field(value, "target")?.to_owned(),
            count: i64_field(value, "count")?,
        },
        Opcode::ExileUntilSourceLeaves => Step::ExileUntilSourceLeaves {
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::DealPowerDamage => Step::DealPowerDamage {
            sources: str_field(value, "sources")?.to_owned(),
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::SetPowerFromCount => Step::SetPowerFromCount {
            zone: str_field(value, "zone")?.to_owned(),
            controller: str_field(value, "controller")?.to_owned(),
        },
    };
    Ok(step)
}

fn parse_condition(value: &Value) -> Result<Condition, IrError> {
    let kind = str_field(value, "kind")?;
    let condition = match kind {
        "kicked" => Condition::Kicked,
        "nth_resolution" => Condition::NthResolution(u64_field(value, "n")?),
        "graveyard_at_least" => Condition::GraveyardAtLeast {
            count: u64_field(value, "count")?,
            predicate: parse_predicate(object_field(value, "predicate")?)?,
        },
        "target_matches" => Condition::TargetMatches {
            role: str_field(value, "role")?.to_owned(),
            predicate: parse_predicate(object_field(value, "predicate")?)?,
        },
        other => {
            return Err(IrError::Malformed(format!(
                "unknown branch condition kind {other:?}"
            )))
        }
    };
    Ok(condition)
}

fn parse_predicate(value: &Value) -> Result<Predicate, IrError> {
    let kind = str_field(value, "kind")?;
    let predicate = match kind {
        "any" => Predicate::Any,
        "all" => Predicate::All(
            array_field(value, "predicates")?
                .iter()
                .map(parse_predicate)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        "card_type" => Predicate::CardType(str_field(value, "value")?.to_owned()),
        "subtype" => Predicate::Subtype(str_field(value, "value")?.to_owned()),
        "not_card_types" => Predicate::NotCardTypes(str_array_field(value, "values")?),
        "power_at_most" => Predicate::PowerAtMost(i64_field(value, "value")?),
        other => {
            return Err(IrError::Malformed(format!(
                "unknown predicate kind {other:?}"
            )))
        }
    };
    Ok(predicate)
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, IrError> {
    value
        .get(name)
        .ok_or_else(|| IrError::Malformed(format!("missing field {name:?}")))
}

fn str_field<'a>(value: &'a Value, name: &str) -> Result<&'a str, IrError> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| IrError::Malformed(format!("field {name:?} must be a string")))
}

fn opt_str_field<'a>(value: &'a Value, name: &str) -> Result<Option<&'a str>, IrError> {
    match value.get(name) {
        None => Ok(None),
        Some(inner) => inner
            .as_str()
            .map(Some)
            .ok_or_else(|| IrError::Malformed(format!("field {name:?} must be a string"))),
    }
}

fn i64_field(value: &Value, name: &str) -> Result<i64, IrError> {
    field(value, name)?
        .as_i64()
        .ok_or_else(|| IrError::Malformed(format!("field {name:?} must be an integer")))
}

fn u64_field(value: &Value, name: &str) -> Result<u64, IrError> {
    field(value, name)?
        .as_u64()
        .ok_or_else(|| IrError::Malformed(format!("field {name:?} must be an unsigned integer")))
}

fn usize_field(value: &Value, name: &str) -> Result<usize, IrError> {
    Ok(u64_field(value, name)? as usize)
}

fn opt_bool_field(value: &Value, name: &str) -> Result<Option<bool>, IrError> {
    match value.get(name) {
        None => Ok(None),
        Some(inner) => inner
            .as_bool()
            .map(Some)
            .ok_or_else(|| IrError::Malformed(format!("field {name:?} must be a boolean"))),
    }
}

fn array_field<'a>(value: &'a Value, name: &str) -> Result<&'a Vec<Value>, IrError> {
    field(value, name)?
        .as_array()
        .ok_or_else(|| IrError::Malformed(format!("field {name:?} must be an array")))
}

fn str_array_field(value: &Value, name: &str) -> Result<Vec<String>, IrError> {
    array_field(value, name)?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| IrError::Malformed(format!("field {name:?} must hold strings")))
        })
        .collect()
}

fn object_field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, IrError> {
    let inner = field(value, name)?;
    if inner.is_object() {
        Ok(inner)
    } else {
        Err(IrError::Malformed(format!(
            "field {name:?} must be an object"
        )))
    }
}
