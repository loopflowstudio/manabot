//! CR 613 evaluation over a single explicit battlefield context. Effects inspect
//! instructions, not final values, when determining dependencies. A context is
//! private derived data; it never calls `Game::characteristics`.

use crate::{
    flow::{
        game::Game,
        statics::{has_board_statics, CharacteristicDetails, Characteristics, LayerBoard},
    },
    state::{
        ability::{
            Ability, AbilityOrigin, Aggregate, Condition, Effect, EffectValue, EffectiveTrigger,
            Except, GrantClock, ObjectProperty, ObjectSelector, Scope,
        },
        card::{
            AbilityGrant, ActivatedAbilityDefinition, CardType, CardTypes, Keyword, Keywords,
            LayerCount, LayerValue, StaticAbility, StaticScope,
        },
        effect::Modification,
        game_object::{
            BattlefieldFacts, Object, ObjectId, ObjectKind, ObjectLki, PermanentId, PlayerId,
            PublicId,
        },
        mana::{Colors, Mana, SpendingContext, SpendingObject},
        permanent::{CounterKind, Counters, Permanent},
        predicate::CardPredicate,
        subtype::{is_creature_type, subtype_category, subtype_corresponds_to, CreatureTypes},
        zone::ZoneType,
    },
};
use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    sync::Arc,
};

const BASIC_LANDS: [(&str, &str); 5] = [
    ("Plains", "W"),
    ("Island", "U"),
    ("Swamp", "B"),
    ("Mountain", "R"),
    ("Forest", "G"),
];

/// Evaluation order, including the four distinct power/toughness sublayers.
/// Copiable values are committed at entry, before this context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Layer {
    Name,
    Control,
    Type,
    Color,
    Ability,
    CharacteristicPower, // 7a
    BasePower,           // 7b
    ModifyPower,         // 7c
    SwitchPower,         // 7d
}

impl Layer {
    const ALL: [Self; 9] = [
        Self::Name,
        Self::Control,
        Self::Type,
        Self::Color,
        Self::Ability,
        Self::CharacteristicPower,
        Self::BasePower,
        Self::ModifyPower,
        Self::SwitchPower,
    ];

    #[inline]
    const fn bit(self) -> u16 {
        1u16 << (self as u8)
    }

    /// Bitmask of all layers strictly after `self` in evaluation order.
    #[inline]
    const fn after_mask(self) -> u16 {
        !0u16 << (self as u8 + 1)
    }
}

/// Compact provenance key for static abilities, counters, and effect records.
/// Seed statics (len 2) and resolved records (len 3) stay inline with no heap
/// allocation; granted statics spill to a `Vec`.
#[derive(Clone, Debug)]
enum NodeKey {
    Inline { len: u8, data: [usize; 3] },
    Heap(Vec<usize>),
}

impl NodeKey {
    #[inline]
    fn from_2(a: usize, b: usize) -> Self {
        Self::Inline {
            len: 2,
            data: [a, b, 0],
        }
    }

    #[inline]
    fn from_3(a: usize, b: usize, c: usize) -> Self {
        Self::Inline {
            len: 3,
            data: [a, b, c],
        }
    }

    #[inline]
    fn as_slice(&self) -> &[usize] {
        match self {
            Self::Inline { len, data } => &data[..usize::from(*len)],
            Self::Heap(vec) => vec.as_slice(),
        }
    }

    fn extended(&self, a: usize, b: usize) -> Self {
        let slice = self.as_slice();
        let mut vec = Vec::with_capacity(slice.len() + 2);
        vec.extend_from_slice(slice);
        vec.push(a);
        vec.push(b);
        Self::Heap(vec)
    }
}

impl PartialEq for NodeKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for NodeKey {}

impl PartialOrd for NodeKey {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NodeKey {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

#[derive(Clone, Debug, PartialEq)]
struct StaticEntry {
    origin: AbilityOrigin,
    key: NodeKey,
    timestamp: PublicId,
    ability: StaticAbility,
}

const DIRTY_TYPES: u8 = 1 << 0;
const DIRTY_ACTIVATIONS: u8 = 1 << 1;
const DIRTY_STATICS: u8 = 1 << 2;
const DIRTY_ALL: u8 = DIRTY_TYPES | DIRTY_ACTIVATIONS | DIRTY_STATICS;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct LayerObject {
    pub view: Arc<Characteristics>,
    pub controller: PlayerId,
    /// `None` borrows `seeds[slot].statics`; `Some(v)` holds owned post-mutation
    /// statics (after ability loss, hexproof removal, or `GrantStatic`).
    statics_override: Option<Vec<StaticEntry>>,
    dirty: u8,
}

/// Active evaluator state during CR 613 pass. Borrowing `live` and `seeds`
/// keeps dependency-probe clones to a single `Vec<Option<LayerObject>>` copy
/// (reusing a scratch buffer) with zero per-permanent `statics` refcount churn.
struct Eval<'a, 'b, 's> {
    game: &'a Game,
    seeds: &'s [Option<Arc<SourceSeed>>],
    objects: Vec<Option<LayerObject>>,
    live: &'b [PermanentId],
    reads_mana: bool,
    reads_tapped: bool,
    reads_life: bool,
}

/// Definition-derived static node owned inline by a [`SourceSeed`].
#[derive(Clone)]
struct SeedNode {
    origin: AbilityOrigin,
    source: PermanentId,
    key: NodeKey,
    timestamp: PublicId,
    plan: SourcePlan,
}

/// Either a dynamically granted static ability (layer 6) or a fixed
/// single-target counter/effect record.
enum DynamicNodeKind {
    Granted {
        plan: Box<SourcePlan>,
        /// Populated on first application for multi-layer static effects (CR 613.6).
        targets: Option<Vec<PermanentId>>,
    },
    Resolved {
        layer: Layer,
        operation: Operation<i32>,
    },
}

struct DynamicNode {
    origin: AbilityOrigin,
    source: PermanentId,
    key: NodeKey,
    timestamp: PublicId,
    kind: DynamicNodeKind,
}

/// An effect keeps one identity across layers. Seed statics borrow their
/// immutable provenance and [`SourcePlan`] from `seeds` without atomic cloning.
enum EvalNode {
    Seed {
        slot: u32,
        node_index: u32,
        /// Populated on first application for multi-layer static effects (CR 613.6).
        targets: Option<Vec<PermanentId>>,
    },
    Dynamic(DynamicNode),
}

/// Definition-derived instructions, shared across revisions and search forks.
#[derive(Clone)]
struct SourcePlan {
    cda: bool,
    layers: u16,
    scope: StaticScope,
    condition: Option<Condition>,
    parts: Vec<(Layer, Operation)>,
    reads: ReadSet,
}

/// One live incarnation's copiable object, bound static plans, and precomputed
/// board summaries. Holding the card keeps definition identity valid even when
/// a fixture/copy detaches it.
#[derive(Clone)]
pub(super) struct SourceSeed {
    source: Object,
    incarnation: PublicId,
    timestamp: PublicId,
    controller: PlayerId,
    view: Arc<Characteristics>,
    statics: Vec<StaticEntry>,
    nodes: Vec<SeedNode>,
    nodes_layers: u16,
    nodes_reads_mana: bool,
    nodes_reads_tapped: bool,
    nodes_reads_life: bool,
    pub(super) has_dynamic_activations: bool,
    pub(super) has_board_statics: bool,
    pub(super) has_triggered_mana_abilities: bool,
}

impl SourceSeed {
    #[inline]
    pub(super) fn view_details_ptr_eq(&self, view: &Characteristics) -> bool {
        Arc::ptr_eq(&self.view.details, &view.details)
    }
}

/// Conservative reads used for both dependency pruning and cache invalidation.
/// Walk each predicate/condition once, including nested alternatives.
#[derive(Clone, Copy, Default)]
struct ReadSet {
    fields: u16,
    mana: bool,
    tapped: bool,
    life: bool,
}

impl ReadSet {
    fn predicate(&mut self, p: &CardPredicate) {
        if p.namesake || p.named.is_some() {
            self.fields |= NAME;
        }
        if p.card_type.is_some() || !p.card_types_any.is_empty() || !p.not_card_types.is_empty() {
            self.fields |= CARD_TYPES;
        }
        if p.chosen_type_of_source
            || p.subtype.is_some()
            || !p.subtypes_any.is_empty()
            || !p.not_subtypes.is_empty()
        {
            self.fields |= SUBTYPES;
        }
        if p.supertype.is_some() {
            self.fields |= SUPERTYPES;
        }
        if !p.colors_any.is_empty() {
            self.fields |= COLORS;
        }
        if p.flying || p.not_flying || p.deathtouch {
            self.fields |= KEYWORDS;
        }
        if p.min_power.is_some() || p.max_power.is_some() || p.min_toughness.is_some() {
            self.fields |= POWER;
        }
        self.tapped |= p.tapped;
        for alternative in &p.any_of {
            self.predicate(alternative);
        }
    }

    fn selector(&mut self, of: &ObjectSelector) {
        if let ObjectSelector::Each {
            predicate,
            exact_mana_value,
            max_mana_value,
            ..
        } = of
        {
            self.predicate(predicate);
            for bound in [exact_mana_value, max_mana_value].into_iter().flatten() {
                self.value(bound);
            }
        }
    }

    fn value(&mut self, value: &EffectValue) {
        match value {
            EffectValue::Half(inner) | EffectValue::Scaled { value: inner, .. } => {
                self.value(inner)
            }
            EffectValue::GraveyardMatching { predicate, .. } if predicate.reads_power() => {
                self.fields |= CARD_TYPES
            }
            EffectValue::SourcePower => self.fields |= POWER,
            EffectValue::GreatestSharedCreatureType => self.fields |= CARD_TYPES | SUBTYPES,
            EffectValue::CreaturesWithCounters => self.fields |= CARD_TYPES,
            EffectValue::GreatestPowerYouControl { predicate } => {
                self.fields |= CARD_TYPES | POWER;
                self.predicate(predicate);
            }
            EffectValue::BattlefieldMatching { predicate }
            | EffectValue::AttackingMatching { predicate } => self.predicate(predicate),
            EffectValue::Property { of, property, .. } => {
                if matches!(property, ObjectProperty::Power | ObjectProperty::Toughness) {
                    self.fields |= POWER;
                }
                self.selector(of);
            }
            EffectValue::Objects { of, predicate, .. } => {
                if let Some(p) = predicate {
                    self.predicate(p);
                }
                self.selector(of);
            }
            EffectValue::DistinctManaValues { of } => self.selector(of),
            EffectValue::LifeTotal(_)
            | EffectValue::LifeAboveStarting(_)
            | EffectValue::LifeGainedThisTurn(_)
            | EffectValue::LifeLostThisTurn(_) => self.life = true,
            _ => {}
        }
    }

    fn layer_value(&mut self, value: &LayerValue) {
        if let LayerValue::Count(count) = value {
            match count.as_ref() {
                LayerCount::CreatureTypes { .. } => self.fields |= SUBTYPES,
                LayerCount::ColorsAmong(p) => {
                    self.fields |= COLORS;
                    self.predicate(p);
                }
                LayerCount::Graveyard { predicate, .. }
                | LayerCount::YouControl(predicate)
                | LayerCount::CountersOn { predicate, .. } => self.predicate(predicate),
                LayerCount::SourceCounters(_) => {}
            }
        }
    }

    fn condition(&mut self, condition: &Condition) {
        match condition {
            Condition::Compare { left, right, .. } => {
                self.value(left);
                self.value(right);
            }
            Condition::ControlsAtLeast {
                predicate,
                distinct_names,
                ..
            } => {
                self.predicate(predicate);
                if *distinct_names {
                    self.fields |= NAME;
                }
            }
            Condition::OpponentControlsMoreCreatures => self.fields |= CARD_TYPES,
            Condition::BattlefieldAtLeast { predicate, .. }
            | Condition::SourceMatches(predicate) => {
                self.predicate(predicate);
            }
            Condition::AllOf(conditions) | Condition::AnyOf(conditions) => {
                for condition in conditions {
                    self.condition(condition);
                }
            }
            Condition::Not(condition) => self.condition(condition),
            Condition::SourceTapped => self.tapped = true,
            Condition::UnspentManaAtLeast { .. } => self.mana = true,
            Condition::LifeAtLeast { .. } => self.life = true,
            _ => {}
        }
    }
}

impl SourcePlan {
    fn new() -> Self {
        Self {
            cda: false,
            layers: 0,
            scope: StaticScope::This,
            condition: None,
            parts: Vec::new(),
            reads: ReadSet::default(),
        }
    }

    fn summarize(mut self) -> Self {
        let mut reads = ReadSet::default();
        match &self.scope {
            StaticScope::All(p)
            | StaticScope::Other(p)
            | StaticScope::OtherYouControl(p)
            | StaticScope::YouControl(p)
            | StaticScope::SpellsYouControl(p) => reads.predicate(p),
            StaticScope::This | StaticScope::Attached => {}
        }
        if let Some(condition) = &self.condition {
            reads.condition(condition);
        }
        let mut layers = 0u16;
        for (part_layer, operation) in &self.parts {
            layers |= part_layer.bit();
            if let Operation::Modify(modification) = operation {
                match modification {
                    Modification::Control => reads.fields |= CONTROL,
                    Modification::SetBasePt { power, toughness } => {
                        for value in power.iter().chain(toughness.iter()) {
                            reads.layer_value(value);
                        }
                    }
                    Modification::ModifyPt { power, toughness } => {
                        reads.layer_value(power);
                        reads.layer_value(toughness);
                    }
                    _ => {}
                }
            }
        }
        self.layers = layers;
        self.reads = reads;
        self
    }
}

impl DynamicNode {
    /// Counters and resolved records already bind one affected object; their
    /// existence and membership no longer depend on a generating static ability.
    fn resolved(
        origin: AbilityOrigin,
        source: PermanentId,
        key: NodeKey,
        timestamp: PublicId,
        layer: Layer,
        operation: Operation<i32>,
    ) -> Self {
        Self {
            origin,
            source,
            key,
            timestamp,
            kind: DynamicNodeKind::Resolved { layer, operation },
        }
    }

    #[inline]
    fn cda(&self) -> bool {
        match &self.kind {
            DynamicNodeKind::Granted { plan, .. } => plan.cda,
            DynamicNodeKind::Resolved { .. } => false,
        }
    }

    #[inline]
    fn layers(&self) -> u16 {
        match &self.kind {
            DynamicNodeKind::Granted { plan, .. } => plan.layers,
            DynamicNodeKind::Resolved { layer, .. } => layer.bit(),
        }
    }

    #[inline]
    fn reads(&self) -> ReadSet {
        match &self.kind {
            DynamicNodeKind::Granted { plan, .. } => plan.reads,
            DynamicNodeKind::Resolved { .. } => ReadSet::default(),
        }
    }

    #[inline]
    fn is_bound(&self) -> bool {
        match &self.kind {
            DynamicNodeKind::Granted { targets, .. } => targets.is_some(),
            DynamicNodeKind::Resolved { .. } => true,
        }
    }
}

impl EvalNode {
    #[inline]
    fn seed_node<'s>(&self, seeds: &'s [Option<Arc<SourceSeed>>]) -> Option<&'s SeedNode> {
        match self {
            Self::Seed {
                slot, node_index, ..
            } => Some(&seeds[*slot as usize].as_ref().unwrap().nodes[*node_index as usize]),
            Self::Dynamic(_) => None,
        }
    }

    #[inline]
    fn source(&self) -> PermanentId {
        match self {
            Self::Seed { slot, .. } => PermanentId(*slot as usize),
            Self::Dynamic(d) => d.source,
        }
    }

    #[inline]
    fn cda(&self, seeds: &[Option<Arc<SourceSeed>>]) -> bool {
        match self {
            Self::Seed { .. } => self.seed_node(seeds).unwrap().plan.cda,
            Self::Dynamic(d) => d.cda(),
        }
    }

    #[inline]
    fn layers(&self, seeds: &[Option<Arc<SourceSeed>>]) -> u16 {
        match self {
            Self::Seed { .. } => self.seed_node(seeds).unwrap().plan.layers,
            Self::Dynamic(d) => d.layers(),
        }
    }

    #[inline]
    fn has_layer(&self, seeds: &[Option<Arc<SourceSeed>>], layer: Layer) -> bool {
        self.layers(seeds) & layer.bit() != 0
    }

    #[inline]
    fn reads(&self, seeds: &[Option<Arc<SourceSeed>>]) -> ReadSet {
        match self {
            Self::Seed { .. } => self.seed_node(seeds).unwrap().plan.reads,
            Self::Dynamic(d) => d.reads(),
        }
    }

    #[inline]
    fn is_bound(&self) -> bool {
        match self {
            Self::Seed { targets, .. } => targets.is_some(),
            Self::Dynamic(d) => d.is_bound(),
        }
    }

    #[inline]
    fn timestamp_key<'a>(
        &'a self,
        seeds: &'a [Option<Arc<SourceSeed>>],
    ) -> (bool, PublicId, &'a [usize]) {
        match self {
            Self::Seed {
                slot, node_index, ..
            } => {
                let s = &seeds[*slot as usize].as_ref().unwrap().nodes[*node_index as usize];
                (!s.plan.cda, s.timestamp, s.key.as_slice())
            }
            Self::Dynamic(d) => (!d.cda(), d.timestamp, d.key.as_slice()),
        }
    }

    fn is_unconditional_attached_or_self(&self, seeds: &[Option<Arc<SourceSeed>>]) -> bool {
        let plan = match self {
            Self::Seed { .. } => &self.seed_node(seeds).unwrap().plan,
            Self::Dynamic(d) => match &d.kind {
                DynamicNodeKind::Granted { plan, .. } => plan,
                DynamicNodeKind::Resolved { .. } => return false,
            },
        };
        plan.condition.is_none() && matches!(plan.scope, StaticScope::This | StaticScope::Attached)
    }

    fn layer_writes_and_grant_static(
        &self,
        seeds: &[Option<Arc<SourceSeed>>],
        layer: Layer,
    ) -> (u16, bool) {
        let parts = match self {
            Self::Seed { .. } => &self.seed_node(seeds).unwrap().plan.parts,
            Self::Dynamic(d) => match &d.kind {
                DynamicNodeKind::Granted { plan, .. } => &plan.parts,
                DynamicNodeKind::Resolved { operation, .. } => {
                    return (
                        operation.writes(),
                        matches!(operation, Operation::Modify(Modification::GrantStatic(_))),
                    );
                }
            },
        };
        let mut writes = 0;
        let mut grants_static = false;
        for (part_layer, op) in parts {
            if *part_layer == layer {
                writes |= op.writes();
                grants_static |= matches!(op, Operation::Modify(Modification::GrantStatic(_)));
            }
        }
        (writes, grants_static)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Operation<N = LayerValue> {
    Control(PlayerId),
    Modify(Modification<N>),
    LandType(String),
    Inherit,
    CatalogTrigger(Box<crate::state::ability::Ability>),
}
// Intermediate fields that instructions can read or change. State-only reads
// (zones, counters, mana, printed mana value) cannot change during
// a layer pass. ABILITIES includes a generating static's continued existence.
const CONTROL: u16 = 256;
const NAME: u16 = 1;
const CARD_TYPES: u16 = 2;
const SUBTYPES: u16 = 4;
const SUPERTYPES: u16 = 8;
const COLORS: u16 = 16;
const KEYWORDS: u16 = 32;
const POWER: u16 = 64;
const ABILITIES: u16 = 128;

impl<N> Operation<N> {
    fn writes(&self) -> u16 {
        match self {
            Self::Control(_) => CONTROL,
            Self::LandType(_) => SUBTYPES | ABILITIES | KEYWORDS,
            Self::Inherit | Self::CatalogTrigger(_) => 0,
            Self::Modify(m) => match m {
                Modification::Control => CONTROL,
                Modification::SetName(_) => NAME,
                Modification::SetCreatureTypes(_)
                | Modification::AddSubtypes(_)
                | Modification::AllCreatureTypes => SUBTYPES,
                Modification::AddTypes(_) | Modification::BecomeCreature => CARD_TYPES,
                Modification::SetTypes { .. } => CARD_TYPES | SUBTYPES,
                Modification::Nonlegendary => SUPERTYPES,
                Modification::SetColors(_) | Modification::AddColors(_) => COLORS,
                Modification::LoseKeywords(keywords) if keywords.has(Keyword::Hexproof) => {
                    ABILITIES | KEYWORDS
                }
                Modification::AddKeywords(_) | Modification::LoseKeywords(_) => KEYWORDS,
                Modification::LoseAllAbilities | Modification::LoseNonManaAbilities => {
                    ABILITIES | KEYWORDS
                }
                Modification::GrantStatic(_) => ABILITIES,
                Modification::SetBasePt { .. }
                | Modification::ModifyPt { .. }
                | Modification::SwitchPt => POWER,
                Modification::Firebending(_)
                | Modification::GrantActivated(_)
                | Modification::GrantTriggered(_)
                | Modification::HasAbilitiesOfSource => 0,
                // Future rules payloads remain conservative until classified.
                _ => u16::MAX,
            },
        }
    }
}

/// Only the state-dependent fields of a resolved instruction; fixed payloads
/// stay borrowed from the generating `Node`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolvedValue {
    None,
    Control(PlayerId),
    SetBasePt {
        power: Option<i32>,
        toughness: Option<i32>,
    },
    ModifyPt {
        power: i32,
        toughness: i32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Instruction {
    target: PermanentId,
    /// Index into `SourcePlan::parts` for `NodeKind::Static`; `0` for `NodeKind::Resolved`.
    part: usize,
    value: ResolvedValue,
}

/// Reused scratch buffers across all layer passes in one `evaluate_board`.
#[derive(Default)]
struct PassScratch {
    pending: Vec<usize>,
    applied: Vec<bool>,
    granted: Vec<DynamicNode>,
    instructions: Vec<Instruction>,
    ranges: Vec<Range<usize>>,
    selected: Vec<usize>,
    writes: Vec<u16>,
    probe: Vec<Instruction>,
    probe_objects: Vec<Option<LayerObject>>,
    dependencies: Vec<bool>,
    reach: Vec<bool>,
}

/// Thread-local scratch reused across board rebuilds (adds 0 bytes to `DerivedCache`).
#[derive(Default)]
struct LayerScratch {
    objects: Vec<Option<LayerObject>>,
    live: Vec<PermanentId>,
    nodes: Vec<EvalNode>,
    pass: PassScratch,
}

thread_local! {
    static LAYER_SCRATCH: Cell<LayerScratch> = Cell::new(LayerScratch::default());
}

/// Takes the thread's scratch buffer on entry and restores its empty capacities
/// on drop (reentrant- and unwind-safe; drops every `Arc` before storing).
struct ScratchGuard(LayerScratch);

impl ScratchGuard {
    fn take() -> Self {
        Self(LAYER_SCRATCH.with(Cell::take))
    }
}

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        self.0.objects.clear();
        self.0.live.clear();
        self.0.nodes.clear();
        self.0.pass.granted.clear();
        self.0.pass.probe_objects.clear();
        let scratch = std::mem::take(&mut self.0);
        let _ = LAYER_SCRATCH.try_with(|cell| {
            let prev = cell.take();
            if prev.objects.capacity() > scratch.objects.capacity() {
                cell.set(prev);
            } else {
                cell.set(scratch);
            }
        });
    }
}

fn layer<N>(modification: &Modification<N>) -> Option<Layer> {
    Some(match modification {
        Modification::Control => Layer::Control,
        Modification::SetName(_) => Layer::Name,
        Modification::SetCreatureTypes(_)
        | Modification::AddSubtypes(_)
        | Modification::AllCreatureTypes
        | Modification::AddTypes(_)
        | Modification::SetTypes { .. }
        | Modification::BecomeCreature
        | Modification::Nonlegendary => Layer::Type,
        Modification::SetColors(_) | Modification::AddColors(_) => Layer::Color,
        Modification::AddKeywords(_)
        | Modification::LoseNonManaAbilities
        | Modification::LoseAllAbilities
        | Modification::LoseKeywords(_)
        | Modification::Firebending(_)
        | Modification::GrantActivated(_)
        | Modification::GrantTriggered(_)
        | Modification::GrantStatic(_)
        | Modification::HasAbilitiesOfSource => Layer::Ability,
        Modification::SetBasePt { .. } => Layer::BasePower,
        Modification::ModifyPt { .. } => Layer::ModifyPower,
        Modification::SwitchPt => Layer::SwitchPower,
        _ => return None,
    })
}

pub(super) fn evaluate_board(
    game: &Game,
    seeds: Option<&mut Vec<Option<Arc<SourceSeed>>>>,
    build: impl FnOnce(
        &mut [Option<LayerObject>],
        &[PermanentId],
        &[Option<Arc<SourceSeed>>],
        bool,
        bool,
        bool,
    ) -> LayerBoard,
) -> LayerBoard {
    #[cfg(feature = "benchmark")]
    let _timer = crate::benchmark_layers::Timer::new(|m, ns| {
        m.boards += 1;
        m.board_ns += ns;
    });
    // The debug oracle rebuilds source plans from definitions and forces full
    // `DIRTY_ALL` finalization so incremental dirty tracking is cross-checked.
    let retain_sources = seeds.is_some();
    let mut fresh = Vec::new();
    let seeds = seeds.unwrap_or(&mut fresh);
    let slot_count = game.state.permanents.len();
    let mut guard = ScratchGuard::take();
    let LayerScratch {
        objects,
        live,
        nodes,
        pass,
    } = &mut guard.0;
    objects.clear();
    objects.resize(slot_count, None);
    live.clear();
    nodes.clear();
    let mut reads_mana = false;
    let mut reads_tapped = false;
    let mut reads_life = false;
    let mut active_layers = 0u16;
    let mut has_pt_counters = false;
    let has_effects = !game.state.effects.is_empty();
    seeds.resize(slot_count, None);
    for (slot, permanent) in game.state.permanents.iter().enumerate() {
        let Some(p) = permanent else {
            seeds[slot] = None;
            continue;
        };
        let id = PermanentId(slot);
        let source = &game.state[p.object];
        let reusable = seeds[slot].as_ref().is_some_and(|seed| {
            seed.source.shares_definition_with(source)
                && seed.incarnation == p.id
                && seed.timestamp == p.timestamp
                && seed.controller == p.entry_controller()
        });
        if !reusable {
            seeds[slot] = Some(Arc::new(build_seed(game, id, p, source)));
        }
        let seed = seeds[slot].as_ref().unwrap();
        objects[slot] = Some(LayerObject {
            view: Arc::clone(&seed.view),
            controller: seed.controller,
            statics_override: None,
            dirty: 0,
        });
        live.push(id);
        // A phased-out permanent's static abilities apply to nothing
        // (CR 702.26b); its own view still takes the records on it.
        if !seed.nodes.is_empty() && !game.is_phased_out(id) {
            reads_mana |= seed.nodes_reads_mana;
            reads_tapped |= seed.nodes_reads_tapped;
            reads_life |= seed.nodes_reads_life;
            active_layers |= seed.nodes_layers;
            for node_index in 0..seed.nodes.len() {
                nodes.push(EvalNode::Seed {
                    slot: slot as u32,
                    node_index: node_index as u32,
                    targets: None,
                });
            }
        }
        if !p.counters.is_empty() {
            for (kind, count) in p.counters.iter() {
                match kind {
                    CounterKind::PlusOnePlusOne | CounterKind::MinusOneMinusOne => {
                        has_pt_counters |= count != 0;
                    }
                    _ => {
                        if let Some(keyword) = kind.keyword().filter(|_| count > 0) {
                            let node = DynamicNode::resolved(
                                AbilityOrigin::Counter,
                                id,
                                NodeKey::from_3(slot, usize::MAX - 1, kind as usize),
                                p.keyword_counter_timestamps
                                    .get(&kind)
                                    .copied()
                                    .unwrap_or(p.timestamp),
                                Layer::Ability,
                                Operation::Modify(Modification::AddKeywords(Keywords::of(&[
                                    keyword,
                                ]))),
                            );
                            active_layers |= node.layers();
                            nodes.push(EvalNode::Dynamic(node));
                        }
                    }
                }
            }
        }
        if has_effects {
            for (i, record) in game.effects_on(id).enumerate() {
                let Some(layer) = layer(&record.modification) else {
                    continue;
                };
                let operation = if record.modification == Modification::HasAbilitiesOfSource {
                    Operation::Inherit
                } else if record.modification == Modification::Control {
                    Operation::Control(record.controller)
                } else {
                    Operation::Modify(record.modification.clone())
                };
                let node = DynamicNode::resolved(
                    AbilityOrigin::Effect(record.timestamp),
                    id,
                    NodeKey::from_3(slot, usize::MAX, i),
                    record.timestamp,
                    layer,
                    operation,
                );
                active_layers |= node.layers();
                nodes.push(EvalNode::Dynamic(node));
            }
        }
    }
    let seeds_slice: &[Option<Arc<SourceSeed>>] = seeds.as_slice();
    let mut eval = Eval {
        game,
        seeds: seeds_slice,
        objects: std::mem::take(objects),
        live: live.as_slice(),
        reads_mana,
        reads_tapped,
        reads_life,
    };
    if active_layers != 0 || has_pt_counters {
        for layer in Layer::ALL {
            let layer_bit = layer.bit();
            if active_layers & layer_bit != 0 {
                let after_mask = layer.after_mask();
                pass.applied.clear();
                loop {
                    pass.pending.clear();
                    for (i, node) in nodes.iter().enumerate() {
                        if !pass.applied.get(i).copied().unwrap_or(false)
                            && node.has_layer(seeds_slice, layer)
                            && eval.exists(node)
                        {
                            pass.pending.push(i);
                        }
                    }
                    if pass.pending.is_empty() {
                        break;
                    }
                    pass.instructions.clear();
                    pass.ranges.clear();
                    for &i in &pass.pending {
                        let start = pass.instructions.len();
                        eval.push_instructions(&nodes[i], layer, &mut pass.instructions);
                        pass.ranges.push(start..pass.instructions.len());
                    }
                    eval.next_effects(nodes, layer, pass);
                    let all_applied = pass.selected.len() == pass.pending.len();
                    let mut discovered_this_layer = false;
                    for &slot in &pass.selected {
                        let index = pass.pending[slot];
                        let slice = &pass.instructions[pass.ranges[slot].clone()];
                        match &mut nodes[index] {
                            EvalNode::Seed {
                                slot: s,
                                node_index,
                                targets,
                            } => {
                                let plan_layers = seeds_slice[*s as usize].as_ref().unwrap().nodes
                                    [*node_index as usize]
                                    .plan
                                    .layers;
                                if targets.is_none()
                                    && plan_layers & after_mask != 0
                                    && !slice.is_empty()
                                {
                                    // Instructions already resolved the affected set. Retain
                                    // it only for effects that continue into a later layer.
                                    let mut bound: Vec<_> = slice
                                        .iter()
                                        .map(|instruction| instruction.target)
                                        .collect();
                                    bound.dedup();
                                    *targets = Some(bound);
                                }
                            }
                            EvalNode::Dynamic(DynamicNode {
                                kind: DynamicNodeKind::Granted { plan, targets },
                                ..
                            }) => {
                                if targets.is_none()
                                    && plan.layers & after_mask != 0
                                    && !slice.is_empty()
                                {
                                    let mut bound: Vec<_> = slice
                                        .iter()
                                        .map(|instruction| instruction.target)
                                        .collect();
                                    bound.dedup();
                                    *targets = Some(bound);
                                }
                            }
                            EvalNode::Dynamic(DynamicNode {
                                kind: DynamicNodeKind::Resolved { .. },
                                ..
                            }) => {}
                        }
                        eval.apply(&nodes[index], slice, Some(&mut pass.granted));
                        for granted_node in pass.granted.drain(..) {
                            let added = eval.push_node(nodes, &mut active_layers, granted_node);
                            discovered_this_layer |= added & layer_bit != 0;
                        }
                    }
                    if all_applied && !discovered_this_layer {
                        debug_assert!(nodes.iter().enumerate().all(|(i, node)| {
                            pass.applied.get(i).copied().unwrap_or(false)
                                || pass.pending.contains(&i)
                                || !node.has_layer(seeds_slice, layer)
                                || !eval.exists(node)
                        }));
                        break;
                    }
                    if pass.applied.len() < nodes.len() {
                        pass.applied.resize(nodes.len(), false);
                    }
                    for &slot in &pass.selected {
                        pass.applied[pass.pending[slot]] = true;
                    }
                }
            } else {
                debug_assert!(!nodes.iter().any(|node| node.has_layer(seeds_slice, layer)));
            }
            if layer == Layer::ModifyPower && has_pt_counters {
                for &id in eval.live {
                    // CR 613.4c: +1/+1 and -1/-1 counters apply together.
                    let permanent = game.live_permanent(id);
                    if permanent.counters.is_empty() {
                        continue;
                    }
                    let count = permanent.counter_count(CounterKind::PlusOnePlusOne)
                        - permanent.counter_count(CounterKind::MinusOneMinusOne);
                    if count != 0 {
                        let view = Arc::make_mut(&mut eval.object_mut(id).view);
                        view.power += count;
                        view.toughness += count;
                    }
                }
            }
        }
    }
    for &id in eval.live {
        let object = eval.objects[id.0].as_mut().unwrap();
        let dirty = if retain_sources {
            object.dirty
        } else {
            DIRTY_ALL
        };
        let statics = object
            .statics_override
            .as_deref()
            .unwrap_or(&seeds_slice[id.0].as_ref().unwrap().statics);
        if dirty != 0 {
            let record = &game.state[game.live_permanent(id).object];
            finish_object(&mut object.view, statics, record, dirty);
            object.dirty = 0;
        }
        #[cfg(debug_assertions)]
        {
            let record = &game.state[game.live_permanent(id).object];
            assert_finished(&object.view, statics, record);
        }
    }
    let Eval {
        objects: eval_objects,
        reads_mana,
        reads_tapped,
        reads_life,
        ..
    } = eval;
    *objects = eval_objects;
    build(
        objects.as_mut_slice(),
        live.as_slice(),
        seeds_slice,
        reads_mana,
        reads_tapped,
        reads_life,
    )
}

/// The copiable object, bound static plans, and board summaries of one live incarnation.
fn build_seed(game: &Game, id: PermanentId, permanent: &Permanent, source: &Object) -> SourceSeed {
    #[cfg(feature = "benchmark")]
    crate::benchmark_layers::record(|m| m.source_plans += 1);
    let statics: Vec<StaticEntry> = source
        .statics
        .iter()
        .enumerate()
        .map(|(i, ability)| StaticEntry {
            origin: AbilityOrigin::Own,
            key: NodeKey::from_2(id.0, i),
            timestamp: permanent.timestamp,
            ability: ability.clone(),
        })
        .collect();
    let controller = permanent.entry_controller();
    let mut object = LayerObject {
        controller,
        view: Arc::new(Characteristics {
            is_legendary: source.is_legendary(),
            keywords: source.keywords,
            power: source.power.unwrap_or(0),
            toughness: source.toughness.unwrap_or(0),
            details: Arc::new(CharacteristicDetails {
                name: source.name.clone(),
                subtypes: source.subtypes.clone(),
                creature_types: source.creature_types(),
                types: source.types.clone(),
                colors: source.colors(),
                mana_abilities: source.mana_abilities.clone(),
                triggered_mana_abilities: source.triggered_mana_abilities.clone(),
                ward: source.ward().cloned(),
                activated_abilities: source.activated_abilities.clone(),
                triggered_abilities: source
                    .abilities
                    .iter()
                    .cloned()
                    .map(|ability| EffectiveTrigger {
                        origin: AbilityOrigin::Own,
                        clock: None,
                        ability,
                        program: false,
                    })
                    .collect(),
                ..Default::default()
            }),
            ..Default::default()
        }),
        statics_override: None,
        dirty: 0,
    };
    finish_object(
        &mut object.view,
        &statics,
        source,
        DIRTY_ACTIVATIONS | DIRTY_STATICS,
    );
    let nodes: Vec<SeedNode> = statics
        .iter()
        .filter_map(|entry| build_seed_node(game, id, entry))
        .collect();
    let mut nodes_layers = 0u16;
    let mut nodes_reads_mana = false;
    let mut nodes_reads_tapped = false;
    let mut nodes_reads_life = false;
    for node in &nodes {
        nodes_layers |= node.plan.layers;
        nodes_reads_mana |= node.plan.reads.mana;
        nodes_reads_tapped |= node.plan.reads.tapped;
        nodes_reads_life |= node.plan.reads.life;
    }
    let has_dynamic_activations = object.view.has_dynamic_activations();
    let has_board_statics = has_board_statics(&object.view.static_abilities);
    let has_triggered_mana_abilities = !object.view.triggered_mana_abilities.is_empty();
    SourceSeed {
        source: source.clone(),
        incarnation: permanent.id,
        timestamp: permanent.timestamp,
        controller,
        view: object.view,
        statics,
        nodes,
        nodes_layers,
        nodes_reads_mana,
        nodes_reads_tapped,
        nodes_reads_life,
        has_dynamic_activations,
        has_board_statics,
        has_triggered_mana_abilities,
    }
}

fn finish_object(
    view: &mut Arc<Characteristics>,
    statics: &[StaticEntry],
    record: &Object,
    dirty: u8,
) {
    if dirty & DIRTY_TYPES != 0 {
        let is_animated = view.is_creature() && !record.types.is_creature();
        if view.is_animated != is_animated {
            Arc::make_mut(view).is_animated = is_animated;
        }
        // During the pass `CreatureTypes::All` marks a record granting all
        // creature types; a listed set is derived from the final subtypes.
        let creature_types = if !view.types.has_creature_subtypes() {
            CreatureTypes::default()
        } else if view.creature_types.is_all() {
            CreatureTypes::All
        } else {
            CreatureTypes::among(&view.subtypes)
        };
        if view.creature_types != creature_types {
            Arc::make_mut(view).details_mut().creature_types = creature_types;
        }
    }
    // Activations are paid as abilities of this source (CR 106.6), read
    // from its final subtypes; an unchanged source keeps sharing its
    // earlier snapshot.
    if dirty & (DIRTY_TYPES | DIRTY_ACTIVATIONS) != 0 && !view.activated_abilities.is_empty() {
        let shared = view
            .activated_abilities
            .iter()
            .map(|ability| &ability.mana_cost.purpose)
            .find(|purpose| {
                matches!(purpose, SpendingContext::Activate {
                    zone: ZoneType::Battlefield,
                    source,
                } if source.describes(&view.subtypes, view.creature_types.is_all()))
            })
            .cloned();
        let purpose = shared.unwrap_or_else(|| SpendingContext::Activate {
            zone: ZoneType::Battlefield,
            source: SpendingObject::new(
                view.subtypes.iter().map(String::as_str),
                view.creature_types.is_all(),
            ),
        });
        if view
            .activated_abilities
            .iter()
            .any(|ability| ability.mana_cost.purpose != purpose)
        {
            for ability in &mut Arc::make_mut(view).details_mut().activated_abilities {
                if ability.mana_cost.purpose != purpose {
                    ability.mana_cost.purpose = purpose.clone();
                }
            }
        }
    }
    if dirty & DIRTY_STATICS != 0 {
        let statics_differ = view.static_abilities.len() != statics.len()
            || view
                .static_abilities
                .iter()
                .zip(statics.iter())
                .any(|(ability, entry)| ability != &entry.ability);
        if statics_differ {
            Arc::make_mut(view).details_mut().static_abilities =
                statics.iter().map(|entry| entry.ability.clone()).collect();
        }
    }
}

#[cfg(debug_assertions)]
fn assert_finished(view: &Characteristics, statics: &[StaticEntry], record: &Object) {
    debug_assert_eq!(
        view.is_animated,
        view.is_creature() && !record.types.is_creature(),
        "incremental finish_object diverged on is_animated"
    );
    if !view.types.has_creature_subtypes() {
        debug_assert!(
            view.creature_types.is_empty(),
            "incremental finish_object diverged on creature_types"
        );
    } else if !view.creature_types.is_all() {
        let listed = view.creature_types.listed();
        debug_assert!(
            listed.windows(2).all(|w| w[0] < w[1]),
            "creature_types must be sorted and distinct"
        );
        debug_assert!(
            listed
                .iter()
                .all(|t| is_creature_type(t) && view.subtypes.iter().any(|s| s == t))
                && view
                    .subtypes
                    .iter()
                    .filter(|s| is_creature_type(s))
                    .all(|s| listed.binary_search(s).is_ok()),
            "incremental finish_object diverged on creature_types"
        );
    }
    if let Some(first) = view.activated_abilities.first() {
        let expected_purpose = &first.mana_cost.purpose;
        debug_assert!(
            matches!(
                expected_purpose,
                SpendingContext::Activate {
                    zone: ZoneType::Battlefield,
                    source,
                } if source.describes(&view.subtypes, view.creature_types.is_all())
            ),
            "incremental finish_object diverged on activation spending context"
        );
        debug_assert!(
            view.activated_abilities
                .iter()
                .all(|ability| ability.mana_cost.purpose == *expected_purpose),
            "incremental finish_object diverged on shared activation spending context"
        );
    }
    debug_assert!(
        view.static_abilities
            .iter()
            .eq(statics.iter().map(|entry| &entry.ability)),
        "incremental finish_object diverged on static_abilities"
    );
}

impl Eval<'_, '_, '_> {
    #[inline]
    fn get(&self, id: PermanentId) -> Option<&LayerObject> {
        self.objects.get(id.0).and_then(Option::as_ref)
    }

    #[inline]
    fn object(&self, id: PermanentId) -> &LayerObject {
        self.objects[id.0].as_ref().unwrap()
    }

    #[inline]
    fn object_mut(&mut self, id: PermanentId) -> &mut LayerObject {
        self.objects[id.0].as_mut().unwrap()
    }

    #[inline]
    fn statics(&self, id: PermanentId) -> &[StaticEntry] {
        match &self.object(id).statics_override {
            Some(v) => v.as_slice(),
            None => &self.seeds[id.0].as_ref().unwrap().statics,
        }
    }

    fn statics_mut(&mut self, id: PermanentId) -> &mut Vec<StaticEntry> {
        let seed_statics = &self.seeds[id.0].as_ref().unwrap().statics;
        self.objects[id.0]
            .as_mut()
            .unwrap()
            .statics_override
            .get_or_insert_with(|| seed_statics.clone())
    }

    fn push_node(
        &mut self,
        nodes: &mut Vec<EvalNode>,
        active_layers: &mut u16,
        node: DynamicNode,
    ) -> u16 {
        let reads = node.reads();
        self.reads_mana |= reads.mana;
        self.reads_tapped |= reads.tapped;
        self.reads_life |= reads.life;
        let layers = node.layers();
        *active_layers |= layers;
        nodes.push(EvalNode::Dynamic(node));
        layers
    }

    /// Select a pending effect under CR 613.8: dependencies first, then
    /// characteristic-defining status and timestamps. Independent instructions
    /// can all apply in one pass; dependent sets return one pending index so
    /// discovery, applicability and dependency ordering are reevaluated.
    fn next_effects(&self, nodes: &[EvalNode], layer: Layer, scratch: &mut PassScratch) {
        let PassScratch {
            pending,
            instructions,
            ranges,
            selected,
            writes,
            probe,
            probe_objects,
            dependencies,
            reach,
            ..
        } = scratch;
        selected.clear();
        let n = pending.len();
        if n == 1 {
            selected.push(0);
            return;
        }
        let seeds = self.seeds;
        let timestamp_order = |&a: &usize| nodes[pending[a]].timestamp_key(seeds);
        // A disjoint read/write pair cannot change the other's instructions.
        // Summaries are conservative; true dependencies still use CR 613.8's
        // before/after semantic comparison. Granted statics force rediscovery.
        writes.clear();
        let mut grants_static = false;
        for (slot, &index) in pending.iter().enumerate() {
            if ranges[slot].is_empty() {
                writes.push(0);
            } else {
                let (w, g) = nodes[index].layer_writes_and_grant_static(seeds, layer);
                writes.push(w);
                grants_static |= g;
            }
        }
        let plausible = |a: usize, b: usize| {
            let node = &nodes[pending[a]];
            let reads = node.reads(seeds).fields | if node.is_bound() { 0 } else { ABILITIES };
            // An unconditional attachment reads only its own controller
            // in layer 2. Independent Auras can apply in one timestamp pass.
            if layer == Layer::Control
                && node.is_unconditional_attached_or_self(seeds)
                && !instructions[ranges[b].clone()]
                    .iter()
                    .any(|i| i.target == node.source())
            {
                return false;
            }
            a != b && node.cda(seeds) == nodes[pending[b]].cda(seeds) && reads & writes[b] != 0
        };
        let independent = !grants_static && !(0..n).any(|a| (0..n).any(|b| plausible(a, b)));
        if independent && !cfg!(debug_assertions) {
            selected.extend(0..n);
            selected.sort_by_key(timestamp_order);
            return;
        }
        dependencies.clear();
        dependencies.resize(n * n, false);
        #[cfg(feature = "benchmark")]
        let _timer = crate::benchmark_layers::Timer::new(|m, ns| {
            m.dependency_passes += 1;
            m.dependency_ns += ns;
        });
        for (b, &index) in pending.iter().enumerate() {
            if !cfg!(debug_assertions) && !(0..n).any(|a| plausible(a, b)) {
                continue;
            }
            {
                #[cfg(feature = "benchmark")]
                let _timer = crate::benchmark_layers::Timer::new(|m, ns| {
                    m.contexts_cloned += 1;
                    m.clone_ns += ns;
                });
                probe_objects.clone_from(&self.objects);
            }
            let mut after = Eval {
                game: self.game,
                seeds: self.seeds,
                objects: std::mem::take(probe_objects),
                live: self.live,
                reads_mana: self.reads_mana,
                reads_tapped: self.reads_tapped,
                reads_life: self.reads_life,
            };
            after.apply(&nodes[index], &instructions[ranges[b].clone()], None);
            let b_cda = nodes[index].cda(seeds);
            for (a, &other) in pending.iter().enumerate() {
                if a != b && nodes[other].cda(seeds) == b_cda {
                    if !cfg!(debug_assertions) && !plausible(a, b) {
                        continue;
                    }
                    debug_assert!(self.exists(&nodes[other]));
                    let changed = if !after.exists(&nodes[other]) {
                        true
                    } else {
                        probe.clear();
                        after.push_instructions(&nodes[other], layer, probe);
                        instructions[ranges[a].clone()] != *probe.as_slice()
                    };
                    dependencies[a * n + b] = changed;
                    debug_assert!(
                        plausible(a, b) || !changed,
                        "layer read/write summary missed a dependency"
                    );
                    #[cfg(feature = "benchmark")]
                    crate::benchmark_layers::record(|m| {
                        m.comparisons += 1;
                        m.edges += u64::from(changed);
                    });
                }
            }
            *probe_objects = after.objects;
        }
        if independent {
            // Keep the complete pairwise oracle in debug, once per independent
            // batch rather than cloning the same board for every suffix.
            debug_assert!(dependencies.iter().all(|&edge| !edge));
            selected.extend(0..n);
            selected.sort_by_key(timestamp_order);
            return;
        }
        // Transitive reachability identifies only edges within dependency
        // loops; external prerequisites still precede the loop (CR 613.8b).
        reach.clear();
        reach.extend_from_slice(dependencies);
        for k in 0..n {
            for a in 0..n {
                if reach[a * n + k] {
                    for b in 0..n {
                        reach[a * n + b] |= reach[k * n + b];
                    }
                }
            }
        }
        selected.push(
            (0..n)
                .filter(|&a| !(0..n).any(|b| dependencies[a * n + b] && !reach[b * n + a]))
                .min_by_key(timestamp_order)
                .expect("dependency graph has a source component"),
        );
    }

    #[inline]
    fn exists(&self, node: &EvalNode) -> bool {
        match node {
            EvalNode::Seed {
                slot,
                node_index,
                targets,
            } => {
                if targets.is_some() {
                    return true;
                }
                let Some(object) = self.objects.get(*slot as usize).and_then(Option::as_ref) else {
                    return false;
                };
                match &object.statics_override {
                    None => {
                        debug_assert!({
                            let seed = self.seeds[*slot as usize].as_ref().unwrap();
                            let key = &seed.nodes[*node_index as usize].key;
                            seed.statics.iter().any(|s| s.key == *key)
                        });
                        true
                    }
                    Some(statics) => {
                        let key = &self.seeds[*slot as usize].as_ref().unwrap().nodes
                            [*node_index as usize]
                            .key;
                        statics.iter().any(|s| s.key == *key)
                    }
                }
            }
            EvalNode::Dynamic(d) => {
                d.is_bound()
                    || self
                        .get(d.source)
                        .is_some_and(|_| self.statics(d.source).iter().any(|s| s.key == d.key))
            }
        }
    }

    fn push_target_instructions(
        &self,
        source: PermanentId,
        controller: PlayerId,
        plan: &SourcePlan,
        layer: Layer,
        target: PermanentId,
        out: &mut Vec<Instruction>,
    ) {
        for (part, (part_layer, op)) in plan.parts.iter().enumerate() {
            if *part_layer != layer {
                continue;
            }
            let value = match op {
                Operation::Modify(Modification::Control) => ResolvedValue::Control(controller),
                Operation::Control(player) => ResolvedValue::Control(*player),
                Operation::Modify(Modification::SetBasePt { power, toughness }) => {
                    ResolvedValue::SetBasePt {
                        power: power.as_ref().map(|n| {
                            self.layer_value(
                                n,
                                || self.game.live_permanent(target).object,
                                controller,
                                Some(source),
                                Some(target),
                            )
                        }),
                        toughness: toughness.as_ref().map(|n| {
                            self.layer_value(
                                n,
                                || self.game.live_permanent(target).object,
                                controller,
                                Some(source),
                                Some(target),
                            )
                        }),
                    }
                }
                Operation::Modify(Modification::ModifyPt { power, toughness }) => {
                    ResolvedValue::ModifyPt {
                        power: self.layer_value(
                            power,
                            || self.game.live_permanent(target).object,
                            controller,
                            Some(source),
                            Some(target),
                        ),
                        toughness: self.layer_value(
                            toughness,
                            || self.game.live_permanent(target).object,
                            controller,
                            Some(source),
                            Some(target),
                        ),
                    }
                }
                _ => ResolvedValue::None,
            };
            out.push(Instruction {
                target,
                part,
                value,
            });
        }
    }

    /// Caller guarantees `self.exists(node)` (verified in `pending` filter and
    /// `next_effects` probe before calling).
    fn push_instructions(&self, node: &EvalNode, layer: Layer, out: &mut Vec<Instruction>) {
        debug_assert!(self.exists(node));
        let (source, plan, bound_targets) = match node {
            EvalNode::Seed {
                slot,
                node_index,
                targets,
            } => {
                let s = &self.seeds[*slot as usize].as_ref().unwrap().nodes[*node_index as usize];
                (s.source, &s.plan, targets.as_deref())
            }
            EvalNode::Dynamic(d) => match &d.kind {
                DynamicNodeKind::Resolved {
                    layer: part_layer,
                    operation,
                } => {
                    if *part_layer == layer {
                        let value = match operation {
                            Operation::Control(player) => ResolvedValue::Control(*player),
                            Operation::Modify(Modification::SetBasePt { power, toughness }) => {
                                ResolvedValue::SetBasePt {
                                    power: *power,
                                    toughness: *toughness,
                                }
                            }
                            Operation::Modify(Modification::ModifyPt { power, toughness }) => {
                                ResolvedValue::ModifyPt {
                                    power: *power,
                                    toughness: *toughness,
                                }
                            }
                            _ => ResolvedValue::None,
                        };
                        out.push(Instruction {
                            target: d.source,
                            part: 0,
                            value,
                        });
                    }
                    return;
                }
                DynamicNodeKind::Granted { plan, targets } => {
                    (d.source, plan.as_ref(), targets.as_deref())
                }
            },
        };
        if let Some(targets) = bound_targets {
            if targets.is_empty() {
                return;
            }
            let controller = self.controller(source);
            for &id in targets {
                self.push_target_instructions(source, controller, plan, layer, id, out);
            }
            return;
        }
        if plan
            .condition
            .as_ref()
            .is_some_and(|c| !self.condition(c, source))
        {
            return;
        }
        match &plan.scope {
            StaticScope::This => {
                if self.get(source).is_some() {
                    let controller = self.controller(source);
                    self.push_target_instructions(source, controller, plan, layer, source, out);
                }
            }
            StaticScope::Attached => {
                if let Some(id) = self
                    .game
                    .attached_permanent(source)
                    .filter(|&id| self.get(id).is_some())
                {
                    let controller = self.controller(source);
                    self.push_target_instructions(source, controller, plan, layer, id, out);
                }
            }
            StaticScope::SpellsYouControl(_) => {}
            scope => {
                let controller = self.controller(source);
                for &id in self.live {
                    if matches_static_scope(
                        scope,
                        source,
                        controller,
                        None,
                        id,
                        self.controller(id),
                        |predicate| self.matches(id, predicate, Some(source)),
                    ) {
                        self.push_target_instructions(source, controller, plan, layer, id, out);
                    }
                }
            }
        }
    }

    fn controlled(
        &self,
        controller: PlayerId,
    ) -> impl Iterator<Item = (PermanentId, &LayerObject)> {
        self.live.iter().copied().filter_map(move |id| {
            let object = self.object(id);
            (object.controller == controller && !self.game.is_phased_out(id))
                .then_some((id, object))
        })
    }

    fn controller(&self, id: PermanentId) -> PlayerId {
        self.object(id).controller
    }

    fn layer_value(
        &self,
        value: &LayerValue,
        object: impl FnOnce() -> ObjectId,
        controller: PlayerId,
        source: Option<PermanentId>,
        affected: Option<PermanentId>,
    ) -> i32 {
        match value {
            LayerValue::Fixed(n) => *n,
            LayerValue::ManaValue => i32::from(self.game.object_mana_value(object())),
            LayerValue::GraveyardCardTypes { offset } => self.game.graveyard_card_types() + offset,
            LayerValue::Count(count) => self.count(count, controller, source, affected),
        }
    }

    fn count(
        &self,
        count: &LayerCount,
        controller: PlayerId,
        source: Option<PermanentId>,
        affected: Option<PermanentId>,
    ) -> i32 {
        let controlled = |p| {
            self.controlled(controller)
                .filter(move |(id, _)| self.matches(*id, p, source))
        };
        match count {
            LayerCount::CreatureTypes { max } => {
                let Some(id) = affected else {
                    return 0;
                };
                let view = &self.object(id).view;
                if view.has_all_creature_types() {
                    *max
                } else {
                    (view.subtypes.iter().filter(|s| is_creature_type(s)).count() as i32).min(*max)
                }
            }
            LayerCount::ColorsAmong(p) => controlled(p)
                .flat_map(|(_, o)| o.view.colors.iter().copied())
                .collect::<BTreeSet<_>>()
                .len() as i32,
            LayerCount::Graveyard { who, predicate } => (0..self.game.state.players.len())
                .map(PlayerId)
                .filter(|player| who.includes(controller, *player))
                .map(|player| self.graveyard_count(player, predicate))
                .sum(),
            LayerCount::YouControl(p) => controlled(p).count() as i32,
            LayerCount::SourceCounters(kind) => source.map_or(0, |source| {
                self.game.live_permanent(source).counter_count(*kind)
            }),
            LayerCount::CountersOn { kind, predicate } => controlled(predicate)
                .map(|(id, _)| self.game.live_permanent(id).counter_count(*kind))
                .sum(),
        }
    }

    fn creature_count(&self, controller: PlayerId) -> usize {
        self.live
            .iter()
            .copied()
            .filter(|&id| self.controller(id) == controller && self.object(id).view.is_creature())
            .count()
    }
    fn graveyard_count(&self, owner: PlayerId, predicate: &CardPredicate) -> i32 {
        self.game
            .state
            .zone_cards(ZoneType::Graveyard, owner)
            .filter(|card| {
                let object = self.game.state.object_of(*card);
                self.game.off_battlefield_matches(object, predicate, || {
                    let card = &self.game.state[object];
                    card.pt_cda().and_then(|(power, _)| power).map_or_else(
                        || card.power.unwrap_or(0),
                        |value| self.layer_value(value, || object, card.owner, None, None),
                    )
                })
            })
            .count() as i32
    }

    fn property(&self, id: PermanentId, property: ObjectProperty) -> i32 {
        match property {
            ObjectProperty::Power => self.object(id).view.power,
            ObjectProperty::Toughness => self.object(id).view.toughness,
            ObjectProperty::ManaValue => self.game.permanent_mana_value(id),
            ObjectProperty::Colors => self.object(id).view.colors.len() as i32,
            ObjectProperty::Counters(kind) => self.game.state.permanents[id]
                .as_ref()
                .map_or(0, |permanent| permanent.counter_count(kind)),
        }
    }

    fn for_each_selected_permanent(
        &self,
        selector: &ObjectSelector,
        source: PermanentId,
        mut visit: impl FnMut(PermanentId),
    ) {
        let controller = self.controller(source);
        match selector {
            ObjectSelector::This => visit(source),
            ObjectSelector::Each {
                of,
                predicate,
                except,
                exact_mana_value,
                max_mana_value,
            } => {
                let exact = exact_mana_value
                    .as_ref()
                    .map(|n| self.comparison_value(n, source));
                let max = max_mana_value
                    .as_ref()
                    .map(|n| self.comparison_value(n, source));
                for &id in self.live {
                    if !self.game.is_phased_out(id)
                        && match of {
                            Scope::YouControl => self.controller(id) == controller,
                            Scope::OpponentsControl => self.controller(id) != controller,
                            Scope::Battlefield => true,
                            _ => false,
                        }
                        && !(id == source && *except == Some(Except::This))
                        && self.matches(id, predicate, Some(source))
                        && exact.is_none_or(|n| self.game.permanent_mana_value(id) == n)
                        && max.is_none_or(|n| self.game.permanent_mana_value(id) <= n)
                    {
                        visit(id);
                    }
                }
            }
            _ => {}
        }
    }

    /// Static comparisons see the characteristics at this point in the layer system.
    fn comparison_value(&self, value: &EffectValue, source: PermanentId) -> i32 {
        let controller = self.controller(source);
        let controlled = || self.controlled(controller).map(|(id, _)| id);
        match value {
            EffectValue::Half(inner) => self.comparison_value(inner, source).div_euclid(2),
            EffectValue::Scaled { factor, value } => {
                factor.saturating_mul(self.comparison_value(value, source))
            }
            EffectValue::SourcePower => self.object(source).view.power,
            EffectValue::Property {
                of,
                property,
                aggregate,
            } => match aggregate {
                Aggregate::Sum => {
                    let mut sum = 0;
                    self.for_each_selected_permanent(of, source, |id| {
                        sum += self.property(id, *property);
                    });
                    sum
                }
                Aggregate::Max => {
                    let mut max: Option<i32> = None;
                    self.for_each_selected_permanent(of, source, |id| {
                        let value = self.property(id, *property);
                        max = Some(max.map_or(value, |current| current.max(value)));
                    });
                    max.unwrap_or(0)
                }
            },
            EffectValue::Objects { of, predicate, .. } => {
                let mut count = 0;
                self.for_each_selected_permanent(of, source, |id| {
                    if predicate
                        .as_ref()
                        .is_none_or(|p| self.matches(id, p, Some(source)))
                    {
                        count += 1;
                    }
                });
                count
            }
            EffectValue::DistinctManaValues { of } => {
                let mut seen = BTreeSet::new();
                self.for_each_selected_permanent(of, source, |id| {
                    seen.insert(self.game.permanent_mana_value(id));
                });
                seen.len() as i32
            }
            EffectValue::GreatestPowerYouControl { predicate } => controlled()
                .filter(|&id| {
                    self.object(id).view.is_creature() && self.matches(id, predicate, None)
                })
                .map(|id| self.object(id).view.power)
                .max()
                .unwrap_or(0)
                .max(0),
            EffectValue::GreatestSharedCreatureType => {
                let mut counts = BTreeMap::new();
                for id in controlled().filter(|&id| self.object(id).view.is_creature()) {
                    for subtype in self
                        .object(id)
                        .view
                        .subtypes
                        .iter()
                        .filter(|s| is_creature_type(s))
                    {
                        *counts.entry(subtype).or_insert(0) += 1;
                    }
                }
                counts.into_values().max().unwrap_or(0)
            }
            EffectValue::CreaturesWithCounters => controlled()
                .filter(|&id| {
                    self.object(id).view.is_creature()
                        && self
                            .game
                            .live_permanent(id)
                            .counter_count(CounterKind::PlusOnePlusOne)
                            > 0
                })
                .count() as i32,
            EffectValue::AttackingMatching { predicate }
            | EffectValue::BattlefieldMatching { predicate } => controlled()
                .filter(|&id| {
                    self.matches(id, predicate, None)
                        && (!matches!(value, EffectValue::AttackingMatching { .. })
                            || self.game.live_permanent(id).attacking)
                })
                .count() as i32,
            EffectValue::GraveyardMatching { base, predicate } => {
                base.saturating_add(self.graveyard_count(controller, predicate))
            }
            _ => self.game.controller_value(value, controller),
        }
    }

    fn condition(&self, condition: &Condition, source: PermanentId) -> bool {
        match condition {
            Condition::Compare { left, op, right } => op.test(
                self.comparison_value(left, source),
                self.comparison_value(right, source),
            ),
            Condition::OpponentControlsMoreCreatures => {
                let controller = self.controller(source);
                self.creature_count(PlayerId(1 - controller.0)) > self.creature_count(controller)
            }
            Condition::BattlefieldAtLeast { count, predicate } => {
                self.live
                    .iter()
                    .copied()
                    .filter(|&id| self.matches(id, predicate, Some(source)))
                    .count()
                    >= *count
            }
            Condition::ControlsAtLeast {
                count,
                predicate,
                distinct_names,
            } => {
                let controller = self.controller(source);
                let matching = self.live.iter().copied().filter(|&id| {
                    self.controller(id) == controller && self.matches(id, predicate, Some(source))
                });
                super::statics::controlled_threshold(matching, *count, *distinct_names, |id| {
                    self.object(id).view.name.as_str()
                })
            }
            Condition::SourceMatches(predicate) => self.matches(source, predicate, Some(source)),
            Condition::AllOf(conditions) => conditions.iter().all(|c| self.condition(c, source)),
            Condition::AnyOf(conditions) => conditions.iter().any(|c| self.condition(c, source)),
            Condition::Not(condition) => !self.condition(condition, source),
            // All other static conditions read state, not characteristics.
            _ => self
                .game
                .layer_state_condition(condition, self.controller(source), source),
        }
    }

    fn matches(
        &self,
        id: PermanentId,
        predicate: &CardPredicate,
        source: Option<PermanentId>,
    ) -> bool {
        matches_view(
            self.game,
            id,
            predicate,
            &self.object(id).view,
            source.map(|s| self.object(s).view.name.as_str()),
            self.game.chosen_type_for(predicate, source),
        )
    }

    fn apply(
        &mut self,
        node: &EvalNode,
        instructions: &[Instruction],
        mut granted: Option<&mut Vec<DynamicNode>>,
    ) {
        let seeds = self.seeds;
        let (node_origin, node_source, node_key, node_timestamp) = match node {
            EvalNode::Seed {
                slot, node_index, ..
            } => {
                let s = &seeds[*slot as usize].as_ref().unwrap().nodes[*node_index as usize];
                (s.origin, s.source, &s.key, s.timestamp)
            }
            EvalNode::Dynamic(d) => (d.origin, d.source, &d.key, d.timestamp),
        };
        for (ordinal, instruction) in instructions.iter().enumerate() {
            let id = instruction.target;
            let origin = if id == node_source {
                node_origin
            } else {
                AbilityOrigin::Permanent(node_source)
            };
            if let ResolvedValue::Control(controller) = instruction.value {
                self.objects[id.0].as_mut().unwrap().controller = controller;
                continue;
            }
            match node {
                EvalNode::Seed {
                    slot, node_index, ..
                } => {
                    let op = &seeds[*slot as usize].as_ref().unwrap().nodes[*node_index as usize]
                        .plan
                        .parts[instruction.part]
                        .1;
                    self.apply_operation(
                        node_source,
                        node_key,
                        node_timestamp,
                        id,
                        origin,
                        ordinal,
                        op,
                        instruction.value,
                        granted.as_deref_mut(),
                    );
                }
                EvalNode::Dynamic(d) => match &d.kind {
                    DynamicNodeKind::Granted { plan, .. } => {
                        self.apply_operation(
                            node_source,
                            node_key,
                            node_timestamp,
                            id,
                            origin,
                            ordinal,
                            &plan.parts[instruction.part].1,
                            instruction.value,
                            granted.as_deref_mut(),
                        );
                    }
                    DynamicNodeKind::Resolved { operation, .. } => {
                        self.apply_operation(
                            node_source,
                            node_key,
                            node_timestamp,
                            id,
                            origin,
                            ordinal,
                            operation,
                            instruction.value,
                            granted.as_deref_mut(),
                        );
                    }
                },
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_operation<N>(
        &mut self,
        node_source: PermanentId,
        node_key: &NodeKey,
        node_timestamp: PublicId,
        id: PermanentId,
        origin: AbilityOrigin,
        ordinal: usize,
        operation: &Operation<N>,
        value: ResolvedValue,
        granted: Option<&mut Vec<DynamicNode>>,
    ) {
        let clock = |ability: &Ability, index| {
            let Ability::Triggered { condition, .. } = ability;
            let static_slot = node_key.as_slice()[1];
            condition.once_each_turn().then(|| GrantClock {
                timestamp: node_timestamp,
                grantor: node_source,
                static_slot: (static_slot != usize::MAX).then_some(static_slot),
                ability_index: index,
            })
        };
        match operation {
            Operation::LandType(subtype) => {
                if !self.object(id).view.is_land() {
                    return;
                }
                {
                    let details = Arc::make_mut(&mut self.object_mut(id).view).details_mut();
                    details
                        .subtypes
                        .retain(|s| !BASIC_LANDS.iter().any(|(name, _)| s == name));
                    details.subtypes.push(subtype.clone());
                    details.added_subtypes.push(subtype.clone());
                }
                self.lose_all_abilities(id);
                let object = self.object_mut(id);
                Arc::make_mut(&mut object.view)
                    .details_mut()
                    .activated_abilities
                    .extend(intrinsic_mana_ability(|name| name == subtype));
                object.dirty |= DIRTY_TYPES | DIRTY_ACTIVATIONS;
            }
            Operation::Inherit => {
                if let Some(definition) = self.game.inherited_definition(id) {
                    let object = self.object_mut(id);
                    let details = Arc::make_mut(&mut object.view).details_mut();
                    if !definition.activated_abilities.is_empty() {
                        details
                            .activated_abilities
                            .extend(definition.activated_abilities.clone());
                        object.dirty |= DIRTY_ACTIVATIONS;
                    }
                    details.triggered_abilities.extend(
                        definition
                            .abilities
                            .iter()
                            .cloned()
                            .enumerate()
                            .map(|(index, ability)| EffectiveTrigger {
                                origin,
                                clock: clock(&ability, index),
                                ability,
                                program: false,
                            }),
                    );
                }
            }
            Operation::CatalogTrigger(ability) => {
                Arc::make_mut(&mut self.object_mut(id).view)
                    .details_mut()
                    .triggered_abilities
                    .push(EffectiveTrigger {
                        origin,
                        clock: clock(ability, 0),
                        ability: (**ability).clone(),
                        program: false,
                    });
            }
            Operation::Modify(modification) => match modification {
                Modification::SetName(name) => {
                    let view = Arc::make_mut(&mut self.object_mut(id).view);
                    view.name_set = true;
                    if view.name != *name {
                        view.details_mut().name = name.clone();
                    }
                }
                Modification::AllCreatureTypes => {
                    // A later grant of every type overrides an earlier set list.
                    let object = self.object_mut(id);
                    if object.view.creature_types_set || !object.view.creature_types.is_all() {
                        let view = Arc::make_mut(&mut object.view);
                        view.creature_types_set = false;
                        if !view.creature_types.is_all() {
                            view.details_mut().creature_types = CreatureTypes::All;
                            object.dirty |= DIRTY_TYPES;
                        }
                    }
                }
                Modification::SetCreatureTypes(types) => {
                    let object = self.object_mut(id);
                    let view = Arc::make_mut(&mut object.view);
                    view.creature_types_set = true;
                    let details = view.details_mut();
                    details.creature_types = CreatureTypes::default();
                    details.subtypes.retain(|s| !is_creature_type(s));
                    details.subtypes.extend(types.iter().cloned());
                    details.added_subtypes.retain(|s| !is_creature_type(s));
                    object.dirty |= DIRTY_TYPES;
                }
                Modification::AddSubtypes(types) => {
                    let object = self.object_mut(id);
                    let gained = if object.view.is_land() {
                        intrinsic_mana_ability(|name| {
                            types.iter().any(|subtype| subtype == name)
                                && !object.view.subtypes.iter().any(|subtype| subtype == name)
                        })
                    } else {
                        None
                    };
                    let any_new = types
                        .iter()
                        .any(|subtype| !object.view.subtypes.contains(subtype));
                    if gained.is_some() || any_new {
                        let details = Arc::make_mut(&mut object.view).details_mut();
                        details.activated_abilities.extend(gained);
                        for subtype in types {
                            if !details.subtypes.contains(subtype) {
                                details.subtypes.push(subtype.clone());
                                details.added_subtypes.push(subtype.clone());
                            }
                        }
                        object.dirty |= DIRTY_TYPES;
                    }
                }
                Modification::AddTypes(types) => {
                    let object = self.object_mut(id);
                    if types.iter().any(|t| !object.view.types.types.contains(t)) {
                        Arc::make_mut(&mut object.view)
                            .details_mut()
                            .types
                            .types
                            .extend(types.iter().copied());
                        object.dirty |= DIRTY_TYPES;
                    }
                }
                Modification::SetTypes { types, subtypes } => {
                    // CR 205.1a, 205.3d: a subtype survives while it
                    // corresponds to a remaining type and the effect sets
                    // no subtype of its kind (a basic land imprisoned as a
                    // land stays a Forest; a Citizen loses other creature
                    // types).
                    let object = self.object_mut(id);
                    let view = Arc::make_mut(&mut object.view);
                    let keeps = |subtype: &String| {
                        types.iter().any(|t| subtype_corresponds_to(subtype, *t))
                            && !subtypes
                                .iter()
                                .any(|set| subtype_category(set) == subtype_category(subtype))
                    };
                    let sets_creature_types = subtypes.iter().any(|s| is_creature_type(s));
                    if !(types.contains(&CardType::Creature) || types.contains(&CardType::Kindred))
                        || sets_creature_types
                    {
                        view.creature_types_set |= sets_creature_types
                            || view.creature_types.is_all()
                            || view.subtypes.iter().any(|s| is_creature_type(s));
                        view.details_mut().creature_types = CreatureTypes::default();
                    }
                    let details = view.details_mut();
                    details.types.types = types.iter().copied().collect();
                    details.subtypes.retain(keeps);
                    details.added_subtypes.retain(keeps);
                    for subtype in subtypes {
                        if !details.subtypes.contains(subtype) {
                            details.subtypes.push(subtype.clone());
                            if !is_creature_type(subtype) {
                                details.added_subtypes.push(subtype.clone());
                            }
                        }
                    }
                    object.dirty |= DIRTY_TYPES;
                }
                Modification::BecomeCreature => {
                    let object = self.object_mut(id);
                    if !object.view.types.types.contains(&CardType::Creature) {
                        Arc::make_mut(&mut object.view)
                            .details_mut()
                            .types
                            .types
                            .insert(CardType::Creature);
                        object.dirty |= DIRTY_TYPES;
                    }
                }
                Modification::Nonlegendary => {
                    let object = self.object_mut(id);
                    if object.view.is_legendary {
                        Arc::make_mut(&mut object.view).is_legendary = false;
                    }
                }
                Modification::SetColors(colors) => {
                    let object = self.object_mut(id);
                    if object.view.colors != *colors {
                        Arc::make_mut(&mut object.view).details_mut().colors = colors.clone();
                    }
                }
                Modification::AddColors(colors) => {
                    let object = self.object_mut(id);
                    if colors.iter().any(|c| !object.view.colors.contains(c)) {
                        Arc::make_mut(&mut object.view)
                            .details_mut()
                            .colors
                            .extend(colors);
                    }
                }
                Modification::AddKeywords(keywords) => {
                    let object = self.object_mut(id);
                    let new_keywords = keywords.without(object.view.keywords);
                    let clears_grant = origin == AbilityOrigin::Own
                        && object
                            .view
                            .keyword_grants
                            .iter()
                            .any(|(keyword, _)| keywords.has(*keyword));
                    if new_keywords != Keywords::default() || clears_grant {
                        let view = Arc::make_mut(&mut object.view);
                        if origin != AbilityOrigin::Own {
                            view.keyword_grants
                                .extend(new_keywords.iter().map(|keyword| (keyword, origin)));
                        } else {
                            view.keyword_grants
                                .retain(|(keyword, _)| !keywords.has(*keyword));
                        }
                        view.keywords = view.keywords.union(*keywords);
                    }
                }
                Modification::LoseKeywords(keywords) => {
                    {
                        let object = self.object_mut(id);
                        let remaining = object.view.keywords.without(*keywords);
                        let clears_grant = object
                            .view
                            .keyword_grants
                            .iter()
                            .any(|(keyword, _)| keywords.has(*keyword));
                        if remaining != object.view.keywords || clears_grant {
                            let view = Arc::make_mut(&mut object.view);
                            view.keyword_grants
                                .retain(|(keyword, _)| !keywords.has(*keyword));
                            view.keywords = remaining;
                        }
                    }
                    if keywords.has(Keyword::Hexproof)
                        && self
                            .statics(id)
                            .iter()
                            .any(|entry| matches!(entry.ability, StaticAbility::HexproofFrom(_)))
                    {
                        self.statics_mut(id).retain(|entry| {
                            !matches!(entry.ability, StaticAbility::HexproofFrom(_))
                        });
                        self.object_mut(id).dirty |= DIRTY_STATICS;
                    }
                }
                Modification::LoseNonManaAbilities => {
                    self.lose_nonmana_abilities(id);
                }
                Modification::LoseAllAbilities => {
                    self.lose_all_abilities(id);
                }
                Modification::Firebending(n) => {
                    let view = Arc::make_mut(&mut self.object_mut(id).view);
                    view.firebending = view.firebending.saturating_add(*n);
                    view.details_mut()
                        .triggered_abilities
                        .push(EffectiveTrigger {
                            origin,
                            clock: None,
                            ability: Ability::firebending(*n),
                            program: true,
                        });
                }
                Modification::GrantActivated(ability) => {
                    let object = self.object_mut(id);
                    Arc::make_mut(&mut object.view)
                        .details_mut()
                        .activated_abilities
                        .push((**ability).clone());
                    object.dirty |= DIRTY_ACTIVATIONS;
                }
                Modification::GrantTriggered(ability) => {
                    Arc::make_mut(&mut self.object_mut(id).view)
                        .details_mut()
                        .triggered_abilities
                        .push(EffectiveTrigger {
                            origin,
                            clock: clock(ability, 0),
                            ability: (**ability).clone(),
                            program: true,
                        });
                }
                Modification::GrantStatic(ability) => {
                    self.object_mut(id).dirty |= DIRTY_STATICS;
                    if let StaticAbility::Ward(cost) = ability.as_ref() {
                        let details = Arc::make_mut(&mut self.object_mut(id).view).details_mut();
                        details.ward = Some(cost.clone());
                        details.triggered_abilities.push(EffectiveTrigger {
                            origin,
                            clock: None,
                            ability: Ability::ward(cost.clone()),
                            program: true,
                        });
                    }
                    let key = node_key.extended(id.0, ordinal);
                    let entry = StaticEntry {
                        origin,
                        key,
                        timestamp: node_timestamp.max(self.game.live_permanent(id).timestamp),
                        ability: (**ability).clone(),
                    };
                    if let Some(granted) = granted {
                        if !self.game.is_phased_out(id) {
                            if let Some(granted_node) = build_dynamic_node(self.game, id, &entry) {
                                granted.push(granted_node);
                            }
                        }
                    }
                    self.statics_mut(id).push(entry);
                }
                Modification::SetBasePt { .. } => {
                    let ResolvedValue::SetBasePt { power, toughness } = value else {
                        unreachable!("SetBasePt instruction carries resolved P/T");
                    };
                    let object = self.object_mut(id);
                    let new_power = power.unwrap_or(object.view.power);
                    let new_toughness = toughness.unwrap_or(object.view.toughness);
                    if new_power != object.view.power || new_toughness != object.view.toughness {
                        let view = Arc::make_mut(&mut object.view);
                        view.power = new_power;
                        view.toughness = new_toughness;
                    }
                }
                Modification::ModifyPt { .. } => {
                    let ResolvedValue::ModifyPt { power, toughness } = value else {
                        unreachable!("ModifyPt instruction carries resolved P/T");
                    };
                    if power != 0 || toughness != 0 {
                        let view = Arc::make_mut(&mut self.object_mut(id).view);
                        view.power += power;
                        view.toughness += toughness;
                    }
                }
                Modification::SwitchPt => {
                    let object = self.object_mut(id);
                    if object.view.power != object.view.toughness {
                        let view = Arc::make_mut(&mut object.view);
                        std::mem::swap(&mut view.power, &mut view.toughness);
                    }
                }
                Modification::Control => unreachable!("control handled above"),
                _ => unreachable!("non-layer modification in layer instruction"),
            },
            Operation::Control(_) => unreachable!("control handled above"),
        }
    }

    fn lose_nonmana_abilities(&mut self, id: PermanentId) {
        let seed_has_statics = !self.seeds[id.0].as_ref().unwrap().statics.is_empty();
        let object = self.object_mut(id);
        let view = Arc::make_mut(&mut object.view);
        view.abilities_lost = true;
        view.keywords = Keywords::default();
        view.keyword_grants.clear();
        view.firebending = 0;
        if view.ward.is_some()
            || !view.triggered_abilities.is_empty()
            || view
                .activated_abilities
                .iter()
                .any(|ability| !ability.is_mana_ability())
        {
            let details = view.details_mut();
            details.ward = None;
            details
                .activated_abilities
                .retain(ActivatedAbilityDefinition::is_mana_ability);
            details.triggered_abilities.clear();
        }
        let had_statics = match &object.statics_override {
            Some(v) => !v.is_empty(),
            None => seed_has_statics,
        };
        if had_statics {
            match &mut object.statics_override {
                Some(v) => v.clear(),
                None => object.statics_override = Some(Vec::new()),
            }
            object.dirty |= DIRTY_STATICS;
        }
    }

    fn lose_all_abilities(&mut self, id: PermanentId) {
        let object = self.object_mut(id);
        if !object.view.mana_abilities.is_empty()
            || !object.view.triggered_mana_abilities.is_empty()
            || !object.view.activated_abilities.is_empty()
        {
            let details = Arc::make_mut(&mut object.view).details_mut();
            details.mana_abilities.clear();
            details.triggered_mana_abilities.clear();
            details.activated_abilities.clear();
        }
        self.lose_nonmana_abilities(id);
    }
}

fn build_source_plan(game: &Game, source: PermanentId, entry: &StaticEntry) -> Option<SourcePlan> {
    let mut plan = SourcePlan::new();
    let mut add = |l, m| plan.parts.push((l, Operation::Modify(m)));
    match &entry.ability {
        StaticAbility::Continuous(effect) => {
            plan.scope = effect.scope.clone();
            plan.condition = effect.condition.clone();
            plan.cda = effect.characteristic_defining;
            for m in &effect.modifications {
                // A modification outside the layer system (a prohibition
                // or prevention) has no reader here; a printed static
                // spelling it would silently do nothing.
                debug_assert!(
                    layer(m).is_some(),
                    "continuous static modification {m:?} has no layer"
                );
                if let Some(l) = layer(m) {
                    let layer = if plan.cda && l == Layer::BasePower {
                        Layer::CharacteristicPower
                    } else {
                        l
                    };
                    add(layer, m.clone());
                }
            }
        }
        StaticAbility::TypeGrant(g) => {
            plan.scope = g.scope.clone();
            plan.condition = g.condition.clone();
            add(Layer::Type, Modification::AddTypes(g.types.clone()));
            add(Layer::Type, Modification::AddSubtypes(g.subtypes.clone()));
        }
        StaticAbility::KeywordGrant(g) => {
            plan.scope = g.scope.clone();
            plan.condition = g.condition.clone();
            add(Layer::Ability, Modification::AddKeywords(g.keywords));
        }
        StaticAbility::SetsLandType(set) => {
            plan.scope = StaticScope::All(set.predicate.clone());
            plan.condition = Some(set.condition.clone());
            plan.parts
                .push((Layer::Type, Operation::LandType(set.subtype.clone())));
        }
        StaticAbility::AttachedSetsCharacteristics(set) => {
            plan.scope = StaticScope::Attached;
            plan.parts = vec![
                (
                    Layer::Name,
                    Operation::Modify(Modification::SetName(set.named.clone())),
                ),
                (
                    Layer::Type,
                    Operation::Modify(Modification::SetCreatureTypes(set.creature_types.clone())),
                ),
                (
                    Layer::Ability,
                    Operation::Modify(Modification::LoseAllAbilities),
                ),
                (
                    Layer::BasePower,
                    Operation::Modify(Modification::SetBasePt {
                        power: Some(LayerValue::Fixed(set.power)),
                        toughness: Some(LayerValue::Fixed(set.toughness)),
                    }),
                ),
            ];
        }
        StaticAbility::AttachedHas(ability) => {
            plan.scope = StaticScope::Attached;
            let object = &game.state[game.live_permanent(source).object];
            let index = object
                .attached_grants()
                .position(|a| a == ability)
                .unwrap_or(0);
            let mut granted = ability.clone();
            granted.granted_by = game
                .permanent_object(source)
                .map(|grantor| AbilityGrant { grantor, index });
            add(
                Layer::Ability,
                Modification::GrantActivated(Box::new(granted)),
            );
        }
        StaticAbility::TriggeredGrant(grant) => {
            plan.scope = grant.scope.clone();
            plan.parts.push((
                Layer::Ability,
                Operation::CatalogTrigger(grant.ability.clone()),
            ));
        }
        StaticAbility::GrantsAnyColorManaAbility(predicate) => {
            plan.scope = StaticScope::YouControl(predicate.clone());
            add(
                Layer::Ability,
                Modification::GrantActivated(Box::new(tap_for_mana(Effect::any_color_mana()))),
            );
        }
        _ => return None,
    }
    Some(plan.summarize())
}

fn build_seed_node(game: &Game, source: PermanentId, entry: &StaticEntry) -> Option<SeedNode> {
    let plan = build_source_plan(game, source, entry)?;
    Some(SeedNode {
        origin: entry.origin,
        source,
        key: entry.key.clone(),
        timestamp: entry.timestamp,
        plan,
    })
}

fn build_dynamic_node(
    game: &Game,
    source: PermanentId,
    entry: &StaticEntry,
) -> Option<DynamicNode> {
    let plan = build_source_plan(game, source, entry)?;
    Some(DynamicNode {
        origin: entry.origin,
        source,
        key: entry.key.clone(),
        timestamp: entry.timestamp,
        kind: DynamicNodeKind::Granted {
            plan: Box::new(plan),
            targets: None,
        },
    })
}

impl Game {
    /// Layer evaluation only visits battlefield slots, which are always live.
    fn live_permanent(&self, id: PermanentId) -> &Permanent {
        self.state.permanents[id].as_ref().unwrap()
    }
}

fn tap_for_mana(effect: Effect) -> ActivatedAbilityDefinition {
    ActivatedAbilityDefinition {
        tap_source: true,
        effects: vec![effect],
        ..Default::default()
    }
}

/// CR 305.6 gives a land each basic type's intrinsic mana ability. Types gained
/// by one effect share one modal ability, the engine's form for a tap that
/// produces one chosen color; modes follow WUBRG order.
fn intrinsic_mana_ability(gained: impl Fn(&str) -> bool) -> Option<ActivatedAbilityDefinition> {
    let mut outputs: Vec<_> = BASIC_LANDS
        .iter()
        .filter(|(name, _)| gained(name))
        .map(|(_, mana)| Mana::parse(mana))
        .collect();
    match outputs.len() {
        0 => None,
        1 => Some(tap_for_mana(Effect::add_mana(outputs.remove(0)))),
        _ => Some(tap_for_mana(Effect::mana_choice(outputs))),
    }
}

pub(super) fn matches_static_scope(
    scope: &StaticScope,
    source: PermanentId,
    source_controller: PlayerId,
    attached: Option<PermanentId>,
    target: PermanentId,
    target_controller: PlayerId,
    matches: impl FnOnce(&CardPredicate) -> bool,
) -> bool {
    let predicate = match scope {
        StaticScope::This => return source == target,
        StaticScope::Attached => return attached == Some(target),
        StaticScope::SpellsYouControl(_) => return false,
        StaticScope::OtherYouControl(p)
            if source != target && source_controller == target_controller =>
        {
            p
        }
        StaticScope::YouControl(p) if source_controller == target_controller => p,
        StaticScope::Other(p) if source != target => p,
        StaticScope::All(p) => p,
        _ => return false,
    };
    matches(predicate)
}

/// A live permanent under `view` (an intermediate layer context or its
/// final view).
pub(super) fn matches_view(
    game: &Game,
    id: PermanentId,
    p: &CardPredicate,
    view: &Characteristics,
    source_name: Option<&str>,
    chosen_type: Option<&str>,
) -> bool {
    let Some(permanent) = game.state.permanents[id].as_ref() else {
        return false;
    };
    let object = &game.state[permanent.object];
    CharacteristicView {
        types: &view.types,
        subtypes: &view.subtypes,
        all_creature_types: view.has_all_creature_types(),
        supertypes: &object.supertypes,
        colors: &view.colors,
        name: &view.name,
        keywords: view.keywords,
        power: view.power,
        toughness: view.toughness,
        is_legendary: view.is_legendary,
        token: object.kind() == ObjectKind::Token,
        attacking: permanent.attacking,
        tapped: permanent.tapped,
        derived: DerivedStatus::Live {
            game,
            id,
            object: permanent.object,
        },
        counters: &permanent.counters,
    }
    .matches(p, source_name, chosen_type)
}

/// The characteristics and permanent status a battlefield predicate reads,
/// borrowed from a live permanent or a departed object's last known facts.
struct CharacteristicView<'a> {
    types: &'a CardTypes,
    subtypes: &'a [String],
    all_creature_types: bool,
    supertypes: &'a [String],
    colors: &'a Colors,
    name: &'a str,
    keywords: Keywords,
    power: i32,
    toughness: i32,
    is_legendary: bool,
    token: bool,
    attacking: bool,
    tapped: bool,
    derived: DerivedStatus<'a>,
    counters: &'a Counters,
}

/// Mana value and combat status, which few predicates read: a live
/// permanent computes them only when asked, a departure recorded them.
enum DerivedStatus<'a> {
    Live {
        game: &'a Game,
        id: PermanentId,
        object: ObjectId,
    },
    Departed(&'a BattlefieldFacts),
}

impl DerivedStatus<'_> {
    fn mana_value(&self) -> u8 {
        match self {
            Self::Live { game, object, .. } => game.object_mana_value(*object),
            Self::Departed(facts) => facts.mana_value,
        }
    }

    fn in_combat(&self) -> bool {
        match self {
            Self::Live { game, id, .. } => game.permanent_in_combat(*id),
            Self::Departed(facts) => facts.in_combat,
        }
    }
}

/// A departed object's predicate match against its last battlefield
/// characteristics (CR 603.10a), never the card it became. Namesake
/// predicates name a live source, so a departure has none.
pub(super) fn matches_departure(lki: &ObjectLki, p: &CardPredicate) -> bool {
    let facts = &lki.battlefield;
    CharacteristicView {
        types: &facts.types,
        subtypes: &facts.subtypes,
        all_creature_types: lki.creature_types.is_all(),
        supertypes: &facts.supertypes,
        colors: &facts.colors,
        name: &facts.name,
        keywords: lki.keywords,
        power: lki.power,
        toughness: lki.toughness,
        is_legendary: facts.is_legendary,
        token: facts.token,
        attacking: facts.attacking,
        tapped: facts.tapped,
        derived: DerivedStatus::Departed(facts),
        counters: &lki.counters,
    }
    .matches(p, None, None)
}

impl CharacteristicView<'_> {
    fn has_subtype(&self, subtype: &str) -> bool {
        self.subtypes.iter().any(|s| s == subtype)
            || (self.all_creature_types && is_creature_type(subtype))
    }

    fn matches(
        &self,
        p: &CardPredicate,
        source_name: Option<&str>,
        chosen_type: Option<&str>,
    ) -> bool {
        // A permanent is on the battlefield, never a graveyard card.
        !p.from_battlefield_this_turn
            && p.accepts_name(self.name)
            && (!p.namesake || source_name == Some(self.name))
            && (!p.chosen_type_of_source || chosen_type.is_some_and(|name| self.has_subtype(name)))
            && (!p.deathtouch || self.keywords.has(Keyword::Deathtouch))
            && (!p.flying || self.keywords.has(Keyword::Flying))
            && (!p.not_flying || !self.keywords.has(Keyword::Flying))
            && (!p.no_counters || self.counters.is_empty())
            && (!p.in_combat || self.derived.in_combat())
            && (!p.with_counter || self.counters.get(CounterKind::PlusOnePlusOne) > 0)
            && (!p.attacking || self.attacking)
            && (!p.tapped || self.tapped)
            && p.token.is_none_or(|token| token == self.token)
            && (p.any_of.is_empty()
                || p.any_of
                    .iter()
                    .any(|alternative| self.matches(alternative, source_name, chosen_type)))
            && p.supertype.as_ref().is_none_or(|s| {
                if s.eq_ignore_ascii_case("legendary") {
                    self.is_legendary
                } else {
                    self.supertypes.iter().any(|t| t.eq_ignore_ascii_case(s))
                }
            })
            && (!p.nonlegendary || !self.is_legendary)
            && (!p.nonbasic
                || !self
                    .supertypes
                    .iter()
                    .any(|s| s.eq_ignore_ascii_case("basic")))
            && p.card_type.is_none_or(|t| self.types.types.contains(&t))
            && (p.card_types_any.is_empty()
                || p.card_types_any
                    .iter()
                    .any(|t| self.types.types.contains(t)))
            && p.not_card_types
                .iter()
                .all(|t| !self.types.types.contains(t))
            && p.subtype.as_ref().is_none_or(|s| self.has_subtype(s))
            && (p.subtypes_any.is_empty() || p.subtypes_any.iter().any(|s| self.has_subtype(s)))
            && (p.colors_any.is_empty() || p.colors_any.iter().any(|c| self.colors.contains(c)))
            && p.not_subtypes.iter().all(|s| !self.has_subtype(s))
            && p.min_power.is_none_or(|min| self.power >= min)
            && p.min_toughness
                .is_none_or(|min| self.toughness >= i32::from(min))
            && p.max_power.is_none_or(|max| self.power <= max)
            && (p.min_mana_value.is_none() && p.max_mana_value.is_none() || {
                let mana_value = self.derived.mana_value();
                p.min_mana_value.is_none_or(|min| mana_value >= min)
                    && p.max_mana_value.is_none_or(|max| mana_value <= max)
            })
    }
}
