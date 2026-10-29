//! Runtime lowering and diagnostic interpretation of typed card IR.
//!
//! This module parses the versioned IR that [`compile`] produces in process from
//! the typed set definitions in `cardsets/` (or a frozen historical document),
//! binds every semantic definition **once** to a real [`ContentPack`]
//! [`CardDefId`], and then executes admitted card programs by dispatching purely
//! on the stable numeric [`Opcode`]. No step in this module inspects a card
//! name, registry name, or oracle string: the parser rejects any name-bearing
//! instruction field, and execution is a `match` over typed steps.
//!
//! The trace interpreter remains an additive diagnostic surface. The same
//! parsed pack also lowers every admitted characteristic and program into the
//! live [`ContentPack`] used by the exact authored matchup, so ordinary game
//! commands, events, stabilization, search forks, and terminal state now run
//! on compiled semantics. Directed rules tests independently witness behavior;
//! shared pool descriptors expose identity and support to deck admission.

pub mod compile;
mod content;
mod pool;

pub use pool::{content_pool, pool_content, ContentPool, Eligibility, PoolDefinition};

pub(crate) use content::content_pack_for_authored_match;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use serde_json::Value;

use crate::cards::content_pack::ContentPack;
use crate::state::{
    ability::PaymentCost,
    card::CardDefId,
    mana::{ManaCost, ManaUse, SPEND_SUBTYPES},
};

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
    SearchBasicLand = 24,
    OptionalDiscardDraw = 25,
    DiscardTargetPlayer = 26,
    Destroy = 27,
    Exile = 28,
    BuffTargetValue = 29,
    ReturnToHandDrawIfControlled = 30,
    ModifyTriggeringPermanent = 31,
    DealDamageValue = 32,
    PutCountersValue = 33,
    CreateTokensValue = 34,
    ExileWithControllerToken = 35,
    DamageOpponentCreatures = 36,
    MillCards = 37,
    MillAndSelect = 38,
    DrainLifeValue = 39,
    CounterSpell = 40,
    CantBeBlockedTarget = 41,
    ReturnGraveyardTarget = 42,
    Modal = 43,
    TargetClause = 44,
    PreventLifeGain = 45,
    CombatRequiresOther = 46,
    BlockingRestriction = 47,
    PowerFromLandCounters = 48,
    SpellCostReduction = 49,
    StaticPt = 50,
    SearchBasicLands = 51,
    Sacrifice = 52,
    DealDamageToController = 53,
    DrainLife = 54,
    UnblockableIf = 55,
    AddLessonMana = 56,
    GrantFlashback = 57,
    GrantMana = 58,
    CreateNonlegendaryCopy = 59,
    ModifyControlled = 60,
    WaterbendOrDiscard = 61,
    Discard = 62,
    ExileIfDies = 63,
    AddSubtype = 64,
    OptionalPayment = 65,
    Airbend = 66,
    AirbendAll = 67,
    OpponentsCastOnlyFromHand = 68,
    ExileTopCardPlayThisTurn = 69,
    SacrificeSource = 70,
    PutIntoLibrarySecondOrBottom = 71,
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
            24 => Opcode::SearchBasicLand,
            25 => Opcode::OptionalDiscardDraw,
            26 => Opcode::DiscardTargetPlayer,
            27 => Opcode::Destroy,
            28 => Opcode::Exile,
            29 => Opcode::BuffTargetValue,
            30 => Opcode::ReturnToHandDrawIfControlled,
            31 => Opcode::ModifyTriggeringPermanent,
            32 => Opcode::DealDamageValue,
            33 => Opcode::PutCountersValue,
            34 => Opcode::CreateTokensValue,
            35 => Opcode::ExileWithControllerToken,
            36 => Opcode::DamageOpponentCreatures,
            37 => Opcode::MillCards,
            38 => Opcode::MillAndSelect,
            39 => Opcode::DrainLifeValue,
            40 => Opcode::CounterSpell,
            41 => Opcode::CantBeBlockedTarget,
            42 => Opcode::ReturnGraveyardTarget,
            43 => Opcode::Modal,
            44 => Opcode::TargetClause,
            45 => Opcode::PreventLifeGain,
            46 => Opcode::CombatRequiresOther,
            47 => Opcode::BlockingRestriction,
            48 => Opcode::PowerFromLandCounters,
            49 => Opcode::SpellCostReduction,
            50 => Opcode::StaticPt,
            51 => Opcode::SearchBasicLands,
            52 => Opcode::Sacrifice,
            53 => Opcode::DealDamageToController,
            54 => Opcode::DrainLife,
            55 => Opcode::UnblockableIf,
            56 => Opcode::AddLessonMana,
            57 => Opcode::GrantFlashback,
            58 => Opcode::GrantMana,
            59 => Opcode::CreateNonlegendaryCopy,
            60 => Opcode::ModifyControlled,
            61 => Opcode::WaterbendOrDiscard,
            62 => Opcode::Discard,
            63 => Opcode::ExileIfDies,
            64 => Opcode::AddSubtype,
            65 => Opcode::OptionalPayment,
            66 => Opcode::Airbend,
            67 => Opcode::AirbendAll,
            68 => Opcode::OpponentsCastOnlyFromHand,
            69 => Opcode::ExileTopCardPlayThisTurn,
            70 => Opcode::SacrificeSource,
            71 => Opcode::PutIntoLibrarySecondOrBottom,

            other => return Err(IrError::UnknownOpcode(other)),
        };
        Ok(opcode)
    }
}

/// Error raised while compiling, parsing or binding IR. Every variant is a
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
    AnyOf(Vec<Predicate>),
    CardType(String),
    Subtype(String),
    NotCardTypes(Vec<String>),
    PowerAtMost(i64),
    PowerAtLeast(i64),
    ManaValueAtLeast(u8),
    Flying,
    WithCounter,
    Attacking,
    TypesAny(Vec<String>),
    SubtypesAny(Vec<String>),
    NotSubtypes(Vec<String>),
}

/// A reviewed runtime condition gating a [`Step::Branch`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Condition {
    Kicked,
    Always,
    CardsDrawnAtLeast(u32),
    NthResolution(u64),
    GraveyardAtLeast { count: u64, predicate: Predicate },
    TargetMatches { role: String, predicate: Predicate },
}

/// One typed, executable IR step. Nested arms (`Branch`, `ForEachTarget`) carry
/// their own step lists, matching the compiler's `then`/`otherwise`/`body`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    Discard,
    ExileIfDies {
        target: String,
    },
    AddSubtype {
        target: String,
        subtype: String,
    },
    /// Restricted mana (CR 106.6). The opcode predates other restrictions;
    /// an omitted `spend_only` keeps its original meaning, Lesson spells.
    AddLessonMana {
        mana: String,
        spend_only: ManaUse,
    },
    GrantFlashback,
    GrantMana {
        predicates: Vec<Predicate>,
    },
    CreateNonlegendaryCopy,
    ModifyControlled {
        predicate: Predicate,
        power: i64,
        toughness: i64,
    },
    OptionalPayment {
        cost: PaymentCost,
        if_paid: Vec<Step>,
        if_declined: Vec<Step>,
    },

    SearchBasicLands {
        count: usize,
        tapped: bool,
        untap_if_lands: Option<usize>,
        bottom_source: bool,
    },
    Sacrifice {
        predicate: Predicate,
    },
    DealDamageToController {
        amount: i64,
    },
    DrainLife {
        amount: i64,
    },
    UnblockableIf {
        condition: Condition,
    },
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
        predicate: Predicate,
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
        reveal: bool,
        look: Amount,
        min_select: i64,
        max_select: i64,
        destination: String,
        predicate: Predicate,
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
        count: Amount,
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
        predicate: Predicate,
    },
    SearchBasicLand {
        subtype: Option<String>,
        to_hand: bool,
    },
    OptionalDiscardDraw {
        count: usize,
        up_to: usize,
    },
    DiscardTargetPlayer {
        count: usize,
        target: String,
    },
    Destroy {
        target: String,
    },
    Exile {
        target: String,
    },
    Airbend {
        target: String,
    },
    AirbendAll {
        except: String,
        predicate: Predicate,
    },
    OpponentsCastOnlyFromHand,
    ExileTopCardPlayThisTurn,
    SacrificeSource {
        then: Vec<Step>,
    },
    PutIntoLibrarySecondOrBottom {
        target: String,
    },
    BuffTargetValue {
        power: ValueExpression,
        toughness: Amount,
        target: String,
    },
    ReturnToHandDrawIfControlled {
        target: String,
    },
    ModifyTriggeringPermanent {
        counters: i32,
        keywords: Vec<String>,
    },
    DealDamageValue {
        amount: ValueExpression,
        target: String,
    },
    PutCountersValue {
        count: ValueExpression,
        target: String,
    },
    CreateTokensValue {
        definition_index: usize,
        count: ValueExpression,
    },
    ExileWithControllerToken {
        target: String,
        definition_index: usize,
    },
    DamageOpponentCreatures {
        amount: i32,
    },
    MillCards {
        count: usize,
    },
    MillAndSelect {
        count: usize,
        predicate: Predicate,
        if_declined: Vec<Step>,
        milled_any: Option<Predicate>,
        then: Vec<Step>,
    },
    DrainLifeValue {
        amount: ValueExpression,
    },
    CounterSpell {
        target: String,
    },
    CantBeBlockedTarget {
        target: String,
    },
    ReturnGraveyardTarget {
        target: String,
        tapped_onto_battlefield: bool,
    },
    Modal {
        modes: Vec<Vec<Step>>,
    },
    TargetClause {
        role: String,
        body: Vec<Step>,
    },
    PreventLifeGain,
    CombatRequiresOther {
        predicate: Predicate,
    },
    BlockingRestriction {
        predicate: Predicate,
        direction: String,
    },
    PowerFromLandCounters,
    SpellCostReduction {
        predicate: Predicate,
        condition: Condition,
        generic: usize,
        first_on_own_turn: bool,
    },
    StaticPt {
        power: i32,
        toughness: i32,
        condition: Condition,
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
    metadata: Value,
}

/// One semantic definition and the reviewed registry name it binds through.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Definition {
    pub semantic_index: usize,
    pub semantic_key: String,
    pub registry_name: String,
    pub text_box: Option<String>,
    characteristics: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticDeck {
    pub key: String,
    pub card_count: usize,
    pub cards: BTreeMap<usize, usize>,
    pub sideboard: BTreeMap<usize, usize>,
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
    pub decks: Vec<SemanticDeck>,
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
    RuleStep(Step),
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
    fn cards_drawn_this_turn(&self) -> u32 {
        0
    }
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
    /// Every authored setup pack; shared by runtime and Python setup discovery.
    pub fn catalog() -> Result<&'static [Self], IrError> {
        static CATALOG: OnceLock<Result<Vec<SemanticPack>, IrError>> = OnceLock::new();
        CATALOG
            .get_or_init(|| {
                compile::presets()?
                    .iter()
                    .filter(|preset| preset.as_str() != "pool")
                    .map(|preset| Self::current(preset))
                    .collect()
            })
            .as_deref()
            .map_err(Clone::clone)
    }

    /// One current preset, compiled in process from the typed set definitions:
    /// an authored setup named in `setups.json`, or `pool`.
    pub fn current(preset: &str) -> Result<Self, IrError> {
        Self::from_value(compile::document(preset)?)
    }

    /// Parse an IR document from JSON text.
    pub fn from_json(text: &str) -> Result<Self, IrError> {
        let value: Value =
            serde_json::from_str(text).map_err(|error| IrError::Malformed(error.to_string()))?;
        Self::from_value(&value)
    }

    fn from_value(value: &Value) -> Result<Self, IrError> {
        let schema_version = u64_field(value, "schema_version")?;
        if u64_field(value, "setup_schema_version")? != 1 {
            return Err(IrError::Malformed(
                "unsupported compiled setup schema".to_string(),
            ));
        }
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
        let mut decks = Vec::new();
        for (deck_index, deck) in array_field(value, "decks")?.iter().enumerate() {
            let deck_key = deck
                .get("key")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("deck_{deck_index}"));
            let mut cards = BTreeMap::new();
            for card in array_field(deck, "cards")? {
                let index = usize_field(card, "definition_index")?;
                deck_definition_indexes.insert(index);
                let count = usize_field(card, "count")?;
                if cards.insert(index, count).is_some() {
                    return Err(IrError::Malformed(format!(
                        "deck {:?} repeats definition index {index}",
                        deck_key
                    )));
                }
            }
            let card_count = deck
                .get("card_count")
                .and_then(Value::as_u64)
                .map(|count| count as usize)
                .unwrap_or_else(|| cards.values().sum());
            if cards.values().sum::<usize>() != card_count {
                return Err(IrError::Malformed(format!(
                    "deck {:?} card count does not match its entries",
                    deck_key
                )));
            }
            let mut sideboard = BTreeMap::new();
            for card in array_field(deck, "sideboard")? {
                let index = usize_field(card, "definition_index")?;
                let count = usize_field(card, "count")?;
                let definition = definitions.get(index).ok_or_else(|| {
                    IrError::Malformed(format!("sideboard references unknown definition {index}"))
                })?;
                if count == 0
                    || opt_bool_field(&definition.characteristics, "token")?.unwrap_or(false)
                {
                    return Err(IrError::Malformed(format!(
                        "invalid sideboard entry in {deck_key:?}"
                    )));
                }
                if sideboard.insert(index, count).is_some() {
                    return Err(IrError::Malformed(format!(
                        "duplicate sideboard entry in {deck_key:?}"
                    )));
                }
                deck_definition_indexes.insert(index);
            }
            decks.push(SemanticDeck {
                key: deck_key,
                card_count,
                cards,
                sideboard,
            });
        }

        Ok(SemanticPack {
            schema_version,
            pack_key,
            ir_hash,
            source_hash,
            definitions,
            programs,
            decks,
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
            Step::RestrictBlocking {
                target, duration, ..
            } => trace.push(TraceEvent::RestrictBlocking {
                target: target.clone(),
                duration: duration.clone(),
            }),
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
                reveal: _,
                look: Amount::Fixed(look),
                min_select,
                max_select,
                destination,
                ..
            } => trace.push(TraceEvent::LookAndSelect {
                look: *look,
                min_select: *min_select,
                max_select: *max_select,
                destination: destination.clone(),
            }),
            Step::LookAndSelect { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
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
            Step::Earthbend {
                target,
                count: Amount::Fixed(count),
            } => trace.push(TraceEvent::Earthbend {
                target: target.clone(),
                count: *count,
            }),
            Step::Earthbend { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::ExileUntilSourceLeaves { target } => {
                trace.push(TraceEvent::ExileUntilSourceLeaves(target.clone()))
            }
            Step::DealPowerDamage { sources, target } => trace.push(TraceEvent::DealPowerDamage {
                sources: sources.clone(),
                target: target.clone(),
            }),
            Step::Discard
            | Step::ExileIfDies { .. }
            | Step::AddSubtype { .. }
            | Step::AddLessonMana { .. }
            | Step::GrantFlashback
            | Step::GrantMana { .. }
            | Step::CreateNonlegendaryCopy
            | Step::ModifyControlled { .. }
            | Step::OptionalPayment { .. }
            | Step::SearchBasicLands { .. }
            | Step::Sacrifice { .. }
            | Step::DealDamageToController { .. }
            | Step::DrainLife { .. }
            | Step::UnblockableIf { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::SearchBasicLand { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::OptionalDiscardDraw { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::DiscardTargetPlayer { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::Destroy { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::Exile { .. }
            | Step::Airbend { .. }
            | Step::AirbendAll { .. }
            | Step::OpponentsCastOnlyFromHand
            | Step::ExileTopCardPlayThisTurn
            | Step::SacrificeSource { .. }
            | Step::PutIntoLibrarySecondOrBottom { .. } => {
                trace.push(TraceEvent::RuleStep(step.clone()))
            }
            Step::BuffTargetValue { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::ReturnToHandDrawIfControlled { .. } => {
                trace.push(TraceEvent::RuleStep(step.clone()))
            }
            Step::ModifyTriggeringPermanent { .. } => {
                trace.push(TraceEvent::RuleStep(step.clone()))
            }
            Step::DealDamageValue { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::PutCountersValue { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::CreateTokensValue { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::ExileWithControllerToken { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::DamageOpponentCreatures { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::MillCards { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::MillAndSelect { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::DrainLifeValue { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::CounterSpell { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::CantBeBlockedTarget { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::ReturnGraveyardTarget { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::Modal { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::TargetClause { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::PreventLifeGain => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::CombatRequiresOther { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::BlockingRestriction { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::PowerFromLandCounters => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::SpellCostReduction { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::StaticPt { .. } => trace.push(TraceEvent::RuleStep(step.clone())),
            Step::SetPowerFromCount {
                zone, controller, ..
            } => trace.push(TraceEvent::SetPowerFromCount {
                zone: zone.clone(),
                controller: controller.clone(),
            }),
        }
        Ok(())
    }
}

fn evaluate_condition(condition: &Condition, context: &dyn InterpreterContext) -> bool {
    match condition {
        Condition::Always => true,
        Condition::CardsDrawnAtLeast(count) => context.cards_drawn_this_turn() >= *count,
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
        text_box: opt_str_field(value, "text_box")?.map(str::to_owned),
        characteristics: value
            .get("characteristics")
            .cloned()
            .unwrap_or_else(|| Value::Object(Default::default())),
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
        metadata: value.clone(),
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
        Opcode::AddLessonMana => Step::AddLessonMana {
            mana: str_field(value, "mana")?.to_owned(),
            spend_only: match value.get("spend_only") {
                None => ManaUse::spell(["Lesson"]),
                Some(_) => mana_use(object_field(value, "spend_only")?)?,
            },
        },
        Opcode::Discard => Step::Discard,
        Opcode::ExileIfDies => Step::ExileIfDies {
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::AddSubtype => Step::AddSubtype {
            target: str_field(value, "target")?.to_owned(),
            subtype: str_field(value, "subtype")?.to_owned(),
        },
        Opcode::GrantFlashback => Step::GrantFlashback,
        Opcode::GrantMana => Step::GrantMana {
            predicates: array_field(value, "predicates")?
                .iter()
                .map(parse_predicate)
                .collect::<Result<_, _>>()?,
        },
        Opcode::CreateNonlegendaryCopy => Step::CreateNonlegendaryCopy,
        Opcode::ModifyControlled => Step::ModifyControlled {
            predicate: parse_predicate(object_field(value, "predicate")?)?,
            power: i64_field(value, "power")?,
            toughness: i64_field(value, "toughness")?,
        },
        Opcode::OptionalPayment => {
            let cost = object_field(value, "cost")?;
            // object_field has already checked the JSON type; exactly one key
            // selects the payment method, so extra or mixed fields are invalid.
            let fields = cost.as_object().unwrap();
            let cost = match (fields.len(), fields.get("mana"), fields.get("waterbend")) {
                (1, Some(_), None) => PaymentCost::Mana(ManaCost::parse(str_field(cost, "mana")?)),
                (1, None, Some(_)) => PaymentCost::Waterbend {
                    generic: u8::try_from(u64_field(cost, "waterbend")?)
                        .map_err(|_| IrError::Malformed("waterbend cost exceeds u8".into()))?,
                },
                _ => {
                    return Err(IrError::Malformed(
                        "payment cost requires exactly one of mana or waterbend".into(),
                    ))
                }
            };
            Step::OptionalPayment {
                cost,
                if_paid: parse_steps(array_field(value, "if_paid")?)?,
                if_declined: parse_steps(array_field(value, "if_declined")?)?,
            }
        }
        // Frozen opcode 61 enters the shared representation at the read boundary.
        Opcode::WaterbendOrDiscard => Step::OptionalPayment {
            cost: PaymentCost::Waterbend {
                generic: u8::try_from(u64_field(value, "generic")?)
                    .map_err(|_| IrError::Malformed("waterbend cost exceeds u8".into()))?,
            },
            if_paid: Vec::new(),
            if_declined: vec![Step::Discard],
        },
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
            predicate: parse_predicate(object_field(value, "predicate")?)?,
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
            reveal: opt_bool_field(value, "reveal")?.unwrap_or(false),
            look: Amount::parse(field(value, "look")?)?,
            min_select: i64_field(value, "min_select")?,
            max_select: i64_field(value, "max_select")?,
            destination: str_field(value, "destination")?.to_owned(),
            predicate: parse_predicate(object_field(value, "predicate")?)?,
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
            count: Amount::parse(field(value, "count")?)?,
        },
        Opcode::ExileUntilSourceLeaves => Step::ExileUntilSourceLeaves {
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::DealPowerDamage => Step::DealPowerDamage {
            sources: str_field(value, "sources")?.to_owned(),
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::SearchBasicLands => Step::SearchBasicLands {
            count: usize_field(value, "count")?,
            tapped: opt_bool_field(value, "tapped")?
                .ok_or_else(|| IrError::Malformed("missing tapped".into()))?,
            untap_if_lands: value
                .get("untap_if_lands")
                .filter(|v| !v.is_null())
                .map(|v| {
                    v.as_u64()
                        .and_then(|v| usize::try_from(v).ok())
                        .ok_or_else(|| IrError::Malformed("invalid untap land count".into()))
                })
                .transpose()?,
            bottom_source: opt_bool_field(value, "bottom_source")?
                .ok_or_else(|| IrError::Malformed("missing bottom_source".into()))?,
        },
        Opcode::Sacrifice => Step::Sacrifice {
            predicate: parse_predicate(object_field(value, "predicate")?)?,
        },
        Opcode::DealDamageToController => Step::DealDamageToController {
            amount: i64_field(value, "amount")?,
        },
        Opcode::DrainLife => Step::DrainLife {
            amount: i64_field(value, "amount")?,
        },
        Opcode::UnblockableIf => Step::UnblockableIf {
            condition: parse_condition(object_field(value, "condition")?)?,
        },
        Opcode::SearchBasicLand => Step::SearchBasicLand {
            subtype: match field(value, "subtype")? {
                Value::Null => None,
                Value::String(subtype) if !subtype.is_empty() => Some(subtype.clone()),
                _ => {
                    return Err(IrError::Malformed(
                        "search subtype must be a string or null".into(),
                    ))
                }
            },
            to_hand: match opt_str_field(value, "destination")? {
                None => false,
                Some("hand") => true,
                Some(other) => {
                    return Err(IrError::Malformed(format!(
                        "unsupported search destination {other:?}"
                    )))
                }
            },
        },
        Opcode::OptionalDiscardDraw => Step::OptionalDiscardDraw {
            count: usize_field(value, "count")?,
            up_to: match value.get("up_to") {
                None => 1,
                Some(_) => usize_field(value, "up_to")?,
            },
        },
        Opcode::DiscardTargetPlayer => Step::DiscardTargetPlayer {
            count: usize_field(value, "count")?,
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::Destroy => Step::Destroy {
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::Exile => Step::Exile {
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::Airbend => Step::Airbend {
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::AirbendAll => Step::AirbendAll {
            except: str_field(value, "except")?.to_owned(),
            predicate: parse_predicate(object_field(value, "predicate")?)?,
        },
        Opcode::OpponentsCastOnlyFromHand => Step::OpponentsCastOnlyFromHand,
        Opcode::ExileTopCardPlayThisTurn => Step::ExileTopCardPlayThisTurn,
        Opcode::SacrificeSource => Step::SacrificeSource {
            then: parse_steps(array_field(value, "then")?)?,
        },
        Opcode::PutIntoLibrarySecondOrBottom => Step::PutIntoLibrarySecondOrBottom {
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::BuffTargetValue => Step::BuffTargetValue {
            power: ValueExpression::parse(object_field(value, "power")?)?,
            toughness: Amount::parse(field(value, "toughness")?)?,
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::ReturnToHandDrawIfControlled => Step::ReturnToHandDrawIfControlled {
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::ModifyTriggeringPermanent => Step::ModifyTriggeringPermanent {
            counters: i32::try_from(i64_field(value, "counters")?)
                .map_err(|_| IrError::Malformed("integer exceeds i32".into()))?,
            keywords: str_array_field(value, "keywords")?,
        },
        Opcode::DealDamageValue => Step::DealDamageValue {
            amount: ValueExpression::parse(object_field(value, "amount")?)?,
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::PutCountersValue => Step::PutCountersValue {
            count: ValueExpression::parse(object_field(value, "count")?)?,
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::CreateTokensValue => Step::CreateTokensValue {
            definition_index: usize_field(value, "definition_index")?,
            count: ValueExpression::parse(object_field(value, "count")?)?,
        },
        Opcode::ExileWithControllerToken => Step::ExileWithControllerToken {
            target: str_field(value, "target")?.to_owned(),
            definition_index: usize_field(value, "definition_index")?,
        },
        Opcode::DamageOpponentCreatures => Step::DamageOpponentCreatures {
            amount: i32::try_from(i64_field(value, "amount")?)
                .map_err(|_| IrError::Malformed("integer exceeds i32".into()))?,
        },
        Opcode::MillCards => Step::MillCards {
            count: usize_field(value, "count")?,
        },
        Opcode::MillAndSelect => Step::MillAndSelect {
            count: usize_field(value, "count")?,
            predicate: parse_predicate(object_field(value, "predicate")?)?,
            if_declined: parse_steps(array_field(value, "if_declined")?)?,
            milled_any: value.get("milled_any").map(parse_predicate).transpose()?,
            then: match value.get("then") {
                None => Vec::new(),
                Some(_) => parse_steps(array_field(value, "then")?)?,
            },
        },
        Opcode::DrainLifeValue => Step::DrainLifeValue {
            amount: ValueExpression::parse(object_field(value, "amount")?)?,
        },
        Opcode::CounterSpell => Step::CounterSpell {
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::CantBeBlockedTarget => Step::CantBeBlockedTarget {
            target: str_field(value, "target")?.to_owned(),
        },
        Opcode::ReturnGraveyardTarget => Step::ReturnGraveyardTarget {
            target: str_field(value, "target")?.to_owned(),
            tapped_onto_battlefield: match opt_str_field(value, "destination")? {
                None | Some("hand") => false,
                Some("battlefield_tapped") => true,
                Some(other) => {
                    return Err(IrError::Malformed(format!(
                        "unknown graveyard return destination {other:?}"
                    )))
                }
            },
        },
        Opcode::Modal => Step::Modal {
            modes: array_field(value, "modes")?
                .iter()
                .map(|v| {
                    v.as_array()
                        .ok_or_else(|| IrError::Malformed("mode must be array".into()))
                        .and_then(|v| parse_steps(v))
                })
                .collect::<Result<_, _>>()?,
        },
        Opcode::TargetClause => Step::TargetClause {
            role: str_field(value, "role")?.to_owned(),
            body: parse_steps(array_field(value, "body")?)?,
        },
        Opcode::PreventLifeGain => Step::PreventLifeGain,
        Opcode::CombatRequiresOther => Step::CombatRequiresOther {
            predicate: parse_predicate(object_field(value, "predicate")?)?,
        },
        Opcode::BlockingRestriction => Step::BlockingRestriction {
            predicate: parse_predicate(object_field(value, "predicate")?)?,
            direction: str_field(value, "direction")?.to_owned(),
        },
        Opcode::PowerFromLandCounters => Step::PowerFromLandCounters,
        Opcode::SpellCostReduction => Step::SpellCostReduction {
            predicate: parse_predicate(object_field(value, "predicate")?)?,
            condition: parse_condition(object_field(value, "condition")?)?,
            generic: usize_field(value, "generic")?,
            first_on_own_turn: opt_bool_field(value, "first_on_own_turn")?
                .ok_or_else(|| IrError::Malformed("missing boolean".into()))?,
        },
        Opcode::StaticPt => Step::StaticPt {
            power: i32::try_from(i64_field(value, "power")?)
                .map_err(|_| IrError::Malformed("integer exceeds i32".into()))?,
            toughness: i32::try_from(i64_field(value, "toughness")?)
                .map_err(|_| IrError::Malformed("integer exceeds i32".into()))?,
            condition: parse_condition(object_field(value, "condition")?)?,
        },
        Opcode::SetPowerFromCount => Step::SetPowerFromCount {
            zone: str_field(value, "zone")?.to_owned(),
            controller: str_field(value, "controller")?.to_owned(),
            predicate: parse_predicate(object_field(value, "predicate")?)?,
        },
    };
    Ok(step)
}

fn parse_condition(value: &Value) -> Result<Condition, IrError> {
    let kind = str_field(value, "kind")?;
    let condition = match kind {
        "kicked" => Condition::Kicked,
        "always" => Condition::Always,
        "cards_drawn_at_least" => Condition::CardsDrawnAtLeast(
            u32::try_from(u64_field(value, "count")?)
                .map_err(|_| IrError::Malformed("count exceeds u32".into()))?,
        ),
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
        "any_of" => Predicate::AnyOf(
            array_field(value, "predicates")?
                .iter()
                .map(parse_predicate)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        "card_type" => Predicate::CardType(str_field(value, "value")?.to_owned()),
        "subtype" => Predicate::Subtype(str_field(value, "value")?.to_owned()),
        "not_card_types" => Predicate::NotCardTypes(str_array_field(value, "values")?),
        "power_at_most" => Predicate::PowerAtMost(i64_field(value, "value")?),
        "power_at_least" => Predicate::PowerAtLeast(i64_field(value, "value")?),
        "mana_value_at_least" => Predicate::ManaValueAtLeast(
            u8::try_from(u64_field(value, "value")?)
                .map_err(|_| IrError::Malformed("mana value exceeds u8".into()))?,
        ),
        "flying" => Predicate::Flying,
        "with_counter" => Predicate::WithCounter,
        "attacking" => Predicate::Attacking,
        "types_any" => Predicate::TypesAny(str_array_field(value, "values")?),
        "subtypes_any" => Predicate::SubtypesAny(str_array_field(value, "values")?),
        "not_subtypes" => Predicate::NotSubtypes(str_array_field(value, "values")?),
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

/// `{"spell_subtypes": [...], "ability_source_subtypes": [...]}`, each optional.
fn mana_use(value: &Value) -> Result<ManaUse, IrError> {
    let subtypes = |name: &str| -> Result<Vec<String>, IrError> {
        let subtypes = match value.get(name) {
            None => Vec::new(),
            Some(_) => str_array_field(value, name)?,
        };
        match subtypes
            .iter()
            .find(|s| !SPEND_SUBTYPES.contains(&s.as_str()))
        {
            Some(other) => Err(IrError::Malformed(format!(
                "unsupported spending restriction subtype {other:?}"
            ))),
            None => Ok(subtypes),
        }
    };
    let spells = subtypes("spell_subtypes")?;
    let abilities = subtypes("ability_source_subtypes")?;
    let use_ = ManaUse::new(
        spells.iter().map(String::as_str),
        abilities.iter().map(String::as_str),
    );
    if use_.is_ordinary() {
        return Err(IrError::Malformed("spend_only names no purpose".into()));
    }
    Ok(use_)
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

/// Quantities evaluated by the live engine relative to the resolving controller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueExpression {
    CreaturesWithCounters,
    BattlefieldMatching(Predicate),
    GraveyardMatching { base: i32, predicate: Predicate },
    CardsDrawnMinus(u32),
}
/// A printed integer or a resolution-time [`ValueExpression`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Amount {
    Fixed(i64),
    Value(ValueExpression),
}

impl Amount {
    fn parse(value: &Value) -> Result<Self, IrError> {
        match value.as_i64() {
            Some(fixed) => Ok(Self::Fixed(fixed)),
            None => ValueExpression::parse(value).map(Self::Value),
        }
    }
}

impl ValueExpression {
    fn parse(value: &Value) -> Result<Self, IrError> {
        Ok(match str_field(value, "kind")? {
            "creatures_with_counters" => Self::CreaturesWithCounters,
            "battlefield_matching" => {
                Self::BattlefieldMatching(parse_predicate(object_field(value, "predicate")?)?)
            }
            "graveyard_matching" => Self::GraveyardMatching {
                base: i32::try_from(i64_field(value, "base")?)
                    .map_err(|_| IrError::Malformed("base exceeds i32".into()))?,
                predicate: parse_predicate(object_field(value, "predicate")?)?,
            },
            "cards_drawn_minus" => Self::CardsDrawnMinus(
                u32::try_from(u64_field(value, "subtract")?)
                    .map_err(|_| IrError::Malformed("subtract exceeds u32".into()))?,
            ),
            other => return Err(IrError::Malformed(format!("unknown quantity {other}"))),
        })
    }
}
