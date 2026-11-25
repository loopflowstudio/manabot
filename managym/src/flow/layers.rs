//! CR 613 evaluation over a single explicit battlefield context. Effects inspect
//! instructions, not final values, when determining dependencies. A context is
//! private derived data; it never calls `Game::characteristics`.

use crate::{
    flow::{
        game::Game,
        statics::{CharacteristicDetails, Characteristics},
    },
    state::{
        ability::{
            Ability, AbilityOrigin, Condition, Effect, EffectValue, EffectiveTrigger, Except,
            GrantClock, ObjectProperty, ObjectSelector, Scope,
        },
        card::{
            AbilityGrant, ActivatedAbilityDefinition, CardType, CardTypes, Keyword, Keywords,
            LayerValue, PowerCda, StaticAbility, StaticPtBuff, StaticScope,
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
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, OnceLock},
};

const BASIC_LANDS: [(&str, &str); 5] = [
    ("Plains", "W"),
    ("Island", "U"),
    ("Swamp", "B"),
    ("Mountain", "R"),
    ("Forest", "G"),
];

/// Evaluation order, including the four distinct power/toughness sublayers.
/// Copy and control are committed by their event owners before this context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Layer {
    Name,
    Type,
    Color,
    Ability,
    CharacteristicPower, // 7a
    BasePower,           // 7b
    ModifyPower,         // 7c
    SwitchPower,         // 7d
}

impl Layer {
    const ALL: [Self; 8] = [
        Self::Name,
        Self::Type,
        Self::Color,
        Self::Ability,
        Self::CharacteristicPower,
        Self::BasePower,
        Self::ModifyPower,
        Self::SwitchPower,
    ];

    #[inline]
    const fn bit(self) -> u8 {
        1 << (self as u8)
    }

    /// Bitmask of all layers strictly after `self` in evaluation order.
    #[inline]
    const fn later_mask(self) -> u8 {
        let bit = self.bit();
        !(bit | (bit - 1))
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

fn empty_statics() -> Arc<Vec<StaticEntry>> {
    static EMPTY: OnceLock<Arc<Vec<StaticEntry>>> = OnceLock::new();
    Arc::clone(EMPTY.get_or_init(|| Arc::new(Vec::new())))
}

const DIRTY_TYPES_SUBTYPES: u8 = 1;
const DIRTY_ACTIVATIONS: u8 = 2;
const DIRTY_STATICS: u8 = 4;
const DIRTY_ALL: u8 = DIRTY_TYPES_SUBTYPES | DIRTY_ACTIVATIONS | DIRTY_STATICS;

#[derive(Clone, Debug)]
pub(super) struct LayerObject {
    pub view: Arc<Characteristics>,
    statics: Arc<Vec<StaticEntry>>,
    dirty: u8,
}

pub(super) struct Context {
    pub objects: Vec<Option<LayerObject>>,
    /// Live battlefield permanent slots in ascending `PermanentId` order.
    pub live: Vec<PermanentId>,
    /// Includes conditional nodes discovered in any layer, even when an earlier
    /// condition short-circuits or a dependency later removes their source.
    pub reads_mana: bool,
    pub reads_tapped: bool,
    pub reads_life: bool,
}

/// Active evaluator state during CR 613 pass. Borrowing `live` keeps
/// dependency-probe clones to a single `Vec<Option<LayerObject>>` copy.
#[derive(Clone)]
struct Eval<'a, 'b> {
    game: &'a Game,
    objects: Vec<Option<LayerObject>>,
    live: &'b [PermanentId],
    reads_mana: bool,
    reads_tapped: bool,
    reads_life: bool,
}

/// Either a definition-derived static ability plan or a fixed single-target
/// counter/effect record (which needs no `SourcePlan`, `Arc`, or target vec).
#[derive(Clone)]
enum NodePlan {
    Static {
        plan: Arc<SourcePlan>,
        /// Populated on first application to objects (CR 613.6) if the effect
        /// continues into a later layer.
        targets: Option<Vec<PermanentId>>,
    },
    Resolved(Layer, Operation),
}

/// An effect keeps one identity across layers.
#[derive(Clone)]
struct Node {
    origin: AbilityOrigin,
    source: PermanentId,
    key: NodeKey,
    timestamp: PublicId,
    plan: NodePlan,
}

/// Definition-derived instructions, shared across revisions and search forks.
#[derive(Clone)]
struct SourcePlan {
    cda: bool,
    layers: u8,
    scope: StaticScope,
    condition: Option<Condition>,
    parts: Vec<(Layer, Operation)>,
    reads: ReadSet,
}

/// One live incarnation's copiable object and bound static plans. Holding the
/// card keeps definition identity valid even when a fixture/copy detaches it.
#[derive(Clone)]
pub(super) struct SourceSeed {
    source: Object,
    incarnation: PublicId,
    timestamp: PublicId,
    object: LayerObject,
    nodes: Vec<Node>,
}

/// Conservative reads used for both dependency pruning and cache invalidation.
/// Walk each predicate/condition once, including nested alternatives.
#[derive(Clone, Copy, Default)]
struct ReadSet {
    fields: u8,
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
        if p.subtype.is_some() || !p.subtypes_any.is_empty() || !p.not_subtypes.is_empty() {
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
    fn on_source(parts: Vec<(Layer, Operation)>) -> Self {
        Self {
            cda: false,
            layers: 0,
            scope: StaticScope::This,
            condition: None,
            parts,
            reads: ReadSet::default(),
        }
    }

    fn summarize(mut self) -> Self {
        let mut reads = ReadSet::default();
        let mut layers = 0u8;
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
        for (part_layer, operation) in &self.parts {
            layers |= part_layer.bit();
            match operation {
                Operation::Buff(buff) => {
                    if buff.per_creature_type_max.is_some() {
                        reads.fields |= SUBTYPES;
                    }
                    if let Some(predicate) = &buff.per_color_among {
                        reads.fields |= COLORS;
                        reads.predicate(predicate);
                    }
                    if let Some(predicate) = &buff.per_graveyard_matching {
                        reads.predicate(predicate);
                    }
                }
                Operation::Cda { cda, .. } => match cda.as_ref() {
                    PowerCda::CountersOnLandsYouControl | PowerCda::CreaturesYouControl => {
                        reads.fields |= CARD_TYPES;
                    }
                    PowerCda::GraveyardMatching(predicate)
                    | PowerCda::GraveyardMatchingBy { predicate, .. } => reads.predicate(predicate),
                },
                _ => {}
            }
        }
        self.layers = layers;
        self.reads = reads;
        self
    }
}

impl Node {
    /// Counters and resolved records already bind one affected object; their
    /// existence and membership no longer depend on a generating static ability.
    fn resolved(
        origin: AbilityOrigin,
        source: PermanentId,
        key: NodeKey,
        timestamp: PublicId,
        part: (Layer, Operation),
    ) -> Self {
        debug_assert!(
            !matches!(part.1, Operation::Buff(_) | Operation::Cda { .. }),
            "resolved node cannot carry dynamic static expressions"
        );
        Self {
            origin,
            source,
            key,
            timestamp,
            plan: NodePlan::Resolved(part.0, part.1),
        }
    }

    #[inline]
    fn cda(&self) -> bool {
        match &self.plan {
            NodePlan::Static { plan, .. } => plan.cda,
            NodePlan::Resolved(..) => false,
        }
    }

    #[inline]
    fn layers(&self) -> u8 {
        match &self.plan {
            NodePlan::Static { plan, .. } => plan.layers,
            NodePlan::Resolved(layer, _) => layer.bit(),
        }
    }

    #[inline]
    fn reads(&self) -> ReadSet {
        match &self.plan {
            NodePlan::Static { plan, .. } => plan.reads,
            NodePlan::Resolved(..) => ReadSet::default(),
        }
    }

    #[inline]
    fn is_bound(&self) -> bool {
        match &self.plan {
            NodePlan::Static { targets, .. } => targets.is_some(),
            NodePlan::Resolved(..) => true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Operation {
    Modify(Modification<LayerValue>),
    LandType(String),
    // Expressions resolve before becoming per-target instructions. Indirection
    // keeps their large predicates out of every small resolved modification.
    /// A characteristic-defining count; `sets_toughness` sets toughness to
    /// the same value ("power and toughness are each equal to").
    Cda {
        cda: Box<PowerCda>,
        sets_toughness: bool,
    },
    Buff(Box<StaticPtBuff>),
    Inherit,
    CatalogTrigger(Box<crate::state::ability::Ability>),
}
// Intermediate fields that instructions can read or change. State-only reads
// (zones, counters, mana, controller, printed mana value) cannot change during
// a layer pass. ABILITIES includes a generating static's continued existence.
const NAME: u8 = 1;
const CARD_TYPES: u8 = 2;
const SUBTYPES: u8 = 4;
const SUPERTYPES: u8 = 8;
const COLORS: u8 = 16;
const KEYWORDS: u8 = 32;
const POWER: u8 = 64;
const ABILITIES: u8 = 128;

impl Operation {
    fn writes(&self) -> u8 {
        match self {
            Self::LandType(_) => SUBTYPES | ABILITIES | KEYWORDS,
            Self::Cda { .. } | Self::Buff(_) => POWER,
            Self::Inherit | Self::CatalogTrigger(_) => 0,
            Self::Modify(m) => match m {
                Modification::SetName(_) => NAME,
                Modification::SetCreatureTypes(_)
                | Modification::AddSubtypes(_)
                | Modification::AllCreatureTypes => SUBTYPES,
                Modification::AddTypes(_) | Modification::BecomeCreature => CARD_TYPES,
                Modification::SetTypes { .. } => CARD_TYPES | SUBTYPES,
                Modification::Nonlegendary => SUPERTYPES,
                Modification::SetColors(_) | Modification::AddColors(_) => COLORS,
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
                _ => u8::MAX,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Instruction {
    target: PermanentId,
    operation: Operation,
}

fn layer<N>(modification: &Modification<N>) -> Option<Layer> {
    Some(match modification {
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

impl Context {
    pub fn evaluate(game: &Game, seeds: Option<&mut Vec<Option<Arc<SourceSeed>>>>) -> Self {
        #[cfg(feature = "benchmark")]
        let _timer = crate::benchmark_layers::Timer::new(|m, ns| {
            m.boards += 1;
            m.board_ns += ns;
        });
        // The debug oracle rebuilds source plans from definitions. Its fresh
        // objects need no retained seed owner while evaluating modifications.
        let retain_sources = seeds.is_some();
        let mut fresh = Vec::new();
        let seeds = seeds.unwrap_or(&mut fresh);
        let slot_count = game.state.permanents.len();
        let mut objects = vec![None; slot_count];
        let mut live = Vec::with_capacity(slot_count);
        let mut reads_mana = false;
        let mut reads_tapped = false;
        let mut reads_life = false;
        let mut nodes = Vec::new();
        let mut active_layers = 0u8;
        let mut has_pt_counters = false;
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
            });
            if !reusable {
                seeds[slot] = Some(Arc::new(build_seed(game, id, p, source)));
            }
            let seed = seeds[slot].as_ref().unwrap();
            objects[slot] = Some(seed.object.clone());
            live.push(id);
            // A phased-out permanent's static abilities apply to nothing
            // (CR 702.26b); its own view still takes the records on it.
            if !game.is_phased_out(id) {
                for node in &seed.nodes {
                    let reads = node.reads();
                    reads_mana |= reads.mana;
                    reads_tapped |= reads.tapped;
                    reads_life |= reads.life;
                    active_layers |= node.layers();
                    nodes.push(node.clone());
                }
            }
            for (kind, count) in p.counters.iter() {
                match kind {
                    CounterKind::PlusOnePlusOne | CounterKind::MinusOneMinusOne => {
                        has_pt_counters |= count != 0;
                    }
                    _ => {
                        if let Some(keyword) = kind.keyword().filter(|_| count > 0) {
                            let node = Node::resolved(
                                AbilityOrigin::Counter,
                                id,
                                NodeKey::from_3(slot, usize::MAX - 1, kind as usize),
                                p.keyword_counter_timestamps
                                    .get(&kind)
                                    .copied()
                                    .unwrap_or(p.timestamp),
                                (
                                    Layer::Ability,
                                    Operation::Modify(Modification::AddKeywords(Keywords::of(&[
                                        keyword,
                                    ]))),
                                ),
                            );
                            active_layers |= node.layers();
                            nodes.push(node);
                        }
                    }
                }
            }
            for (i, record) in game.effects_on(id).enumerate() {
                let Some(layer) = layer(&record.modification) else {
                    continue;
                };
                let operation = if record.modification == Modification::HasAbilitiesOfSource {
                    Operation::Inherit
                } else {
                    Operation::Modify(record.modification.map_amount(|n| LayerValue::Fixed(*n)))
                };
                let node = Node::resolved(
                    AbilityOrigin::Effect(record.timestamp),
                    id,
                    NodeKey::from_3(slot, usize::MAX, i),
                    record.timestamp,
                    (layer, operation),
                );
                active_layers |= node.layers();
                nodes.push(node);
            }
        }
        if !retain_sources {
            seeds.clear();
        }
        let mut eval = Eval {
            game,
            objects,
            live: &live,
            reads_mana,
            reads_tapped,
            reads_life,
        };
        if active_layers != 0 || has_pt_counters {
            let mut known: Option<BTreeSet<NodeKey>> = None;
            let mut pending: Vec<usize> = Vec::new();
            let mut applied: Vec<u8> = if active_layers != 0 {
                vec![0u8; nodes.len()]
            } else {
                Vec::new()
            };
            for layer in Layer::ALL {
                let layer_bit = layer.bit();
                if active_layers & layer_bit != 0 {
                    let later_mask = layer.later_mask();
                    loop {
                        pending.clear();
                        for (i, node) in nodes.iter().enumerate() {
                            if applied[i] & layer_bit == 0
                                && node.layers() & layer_bit != 0
                                && eval.exists(node)
                            {
                                pending.push(i);
                            }
                        }
                        if pending.is_empty() {
                            break;
                        }
                        let instructions: Vec<_> = pending
                            .iter()
                            .map(|&i| eval.instructions(&nodes[i], layer))
                            .collect();
                        let selected_order =
                            eval.next_effects(&nodes, &pending, &instructions, layer);
                        let all_pending_selected = selected_order.len() == pending.len();
                        let mut granted_static = false;
                        for selected in selected_order {
                            let index = pending[selected];
                            if let NodePlan::Static { plan, targets } = &mut nodes[index].plan {
                                if targets.is_none()
                                    && plan.layers & later_mask != 0
                                    && !instructions[selected].is_empty()
                                {
                                    // Instructions already resolved the affected set. Retain
                                    // it only for effects that continue into a later layer.
                                    let mut affected: Vec<_> = instructions[selected]
                                        .iter()
                                        .map(|instruction| instruction.target)
                                        .collect();
                                    affected.dedup();
                                    *targets = Some(affected);
                                }
                            }
                            // Only a static grant introduces layer nodes. Removing abilities
                            // changes existence, which the pending filter checks directly.
                            let has_grant_static = instructions[selected].iter().any(|i| {
                                matches!(i.operation, Operation::Modify(Modification::GrantStatic(_)))
                            });
                            if has_grant_static && known.is_none() {
                                let mut set = BTreeSet::new();
                                for &id in eval.live {
                                    for entry in eval.object(id).statics.iter() {
                                        set.insert(entry.key.clone());
                                    }
                                }
                                known = Some(set);
                            }
                            eval.apply(&nodes[index], &instructions[selected]);
                            if has_grant_static {
                                granted_static = true;
                                eval.discover(
                                    &mut nodes,
                                    known.as_mut().unwrap(),
                                    &mut active_layers,
                                );
                                applied.resize(nodes.len(), 0);
                            }
                            applied[index] |= layer_bit;
                        }
                        if all_pending_selected && !granted_static {
                            debug_assert!(
                                !(0..nodes.len()).any(|i| {
                                    applied[i] & layer_bit == 0
                                        && nodes[i].layers() & layer_bit != 0
                                        && eval.exists(&nodes[i])
                                }),
                                "pending layer nodes remained after full independent pass"
                            );
                            break;
                        }
                    }
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
            if dirty != 0 {
                let record = &game.state[game.live_permanent(id).object];
                finish_object(object, record, dirty);
                object.dirty = 0;
            }
        }
        let Eval {
            objects,
            reads_mana,
            reads_tapped,
            reads_life,
            ..
        } = eval;
        Self {
            objects,
            live,
            reads_mana,
            reads_tapped,
            reads_life,
        }
    }
}

/// The copiable object and bound static plans of one live incarnation.
fn build_seed(game: &Game, id: PermanentId, permanent: &Permanent, source: &Object) -> SourceSeed {
    #[cfg(feature = "benchmark")]
    crate::benchmark_layers::record(|m| m.source_plans += 1);
    let statics = if source.statics.is_empty() {
        empty_statics()
    } else {
        Arc::new(
            source
                .statics
                .iter()
                .enumerate()
                .map(|(i, ability)| StaticEntry {
                    origin: AbilityOrigin::Own,
                    key: NodeKey::from_2(id.0, i),
                    timestamp: permanent.timestamp,
                    ability: ability.clone(),
                })
                .collect(),
        )
    };
    let mut object = LayerObject {
        view: Arc::new(Characteristics {
            is_legendary: source.is_legendary(),
            keywords: source.keywords,
            power: source.power.unwrap_or(0),
            toughness: source.toughness.unwrap_or(0),
            details: Arc::new(CharacteristicDetails {
                name: source.name.clone(),
                subtypes: source.subtypes.clone(),
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
        statics,
        dirty: 0,
    };
    finish_object(&mut object, source, DIRTY_ALL);
    let nodes = object
        .statics
        .iter()
        .filter_map(|entry| build_node(game, id, entry))
        .collect();
    SourceSeed {
        source: source.clone(),
        incarnation: permanent.id,
        timestamp: permanent.timestamp,
        object,
        nodes,
    }
}

fn finish_object(object: &mut LayerObject, record: &Object, dirty: u8) {
    if dirty & DIRTY_TYPES_SUBTYPES != 0 {
        let is_animated = object.view.is_creature() && !record.types.is_creature();
        if object.view.is_animated != is_animated {
            Arc::make_mut(&mut object.view).is_animated = is_animated;
        }
        // During the pass `CreatureTypes::All` marks a record granting all
        // creature types; a listed set is derived from the final subtypes.
        let creature_types = if !object.view.is_creature() {
            CreatureTypes::default()
        } else if object.view.creature_types.is_all() {
            CreatureTypes::All
        } else {
            CreatureTypes::among(&object.view.subtypes)
        };
        if object.view.creature_types != creature_types {
            Arc::make_mut(&mut object.view).details_mut().creature_types = creature_types;
        }
    }
    // Activations are paid as abilities of this source (CR 106.6), read
    // from its final subtypes; an unchanged source keeps sharing its
    // earlier snapshot.
    if dirty & (DIRTY_TYPES_SUBTYPES | DIRTY_ACTIVATIONS) != 0
        && !object.view.activated_abilities.is_empty()
    {
        let shared = object
            .view
            .activated_abilities
            .iter()
            .map(|ability| &ability.mana_cost.purpose)
            .find(|purpose| {
                matches!(purpose, SpendingContext::Activate {
                    zone: ZoneType::Battlefield,
                    source,
                } if source.describes(&object.view.subtypes, object.view.creature_types.is_all()))
            })
            .cloned();
        let purpose = shared.unwrap_or_else(|| SpendingContext::Activate {
            zone: ZoneType::Battlefield,
            source: SpendingObject::new(
                object.view.subtypes.iter().map(String::as_str),
                object.view.creature_types.is_all(),
            ),
        });
        if object
            .view
            .activated_abilities
            .iter()
            .any(|ability| ability.mana_cost.purpose != purpose)
        {
            for ability in &mut Arc::make_mut(&mut object.view)
                .details_mut()
                .activated_abilities
            {
                if ability.mana_cost.purpose != purpose {
                    ability.mana_cost.purpose = purpose.clone();
                }
            }
        }
    }
    if dirty & DIRTY_STATICS != 0 {
        let statics_differ = object.view.static_abilities.len() != object.statics.len()
            || object
                .view
                .static_abilities
                .iter()
                .zip(object.statics.iter())
                .any(|(ability, entry)| ability != &entry.ability);
        if statics_differ {
            Arc::make_mut(&mut object.view).details_mut().static_abilities = object
                .statics
                .iter()
                .map(|entry| entry.ability.clone())
                .collect();
        }
    }
}

impl Eval<'_, '_> {
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

    fn discover(
        &mut self,
        nodes: &mut Vec<Node>,
        known: &mut BTreeSet<NodeKey>,
        active_layers: &mut u8,
    ) {
        // Occurrence provenance preserves duplicate grants, including statics
        // discovered during layer 6 that start applying in a later layer.
        for &id in self.live {
            if self.game.is_phased_out(id) {
                continue;
            }
            for entry in self.objects[id.0].as_ref().unwrap().statics.iter() {
                if known.insert(entry.key.clone()) {
                    if let Some(node) = build_node(self.game, id, entry) {
                        let reads = node.reads();
                        self.reads_mana |= reads.mana;
                        self.reads_tapped |= reads.tapped;
                        self.reads_life |= reads.life;
                        *active_layers |= node.layers();
                        nodes.push(node);
                    }
                }
            }
        }
    }

    /// Select a pending effect under CR 613.8: dependencies first, then
    /// characteristic-defining status and timestamps. Independent instructions
    /// can all apply in one pass; dependent sets return one pending index so
    /// discovery, applicability and dependency ordering are reevaluated.
    fn next_effects(
        &self,
        nodes: &[Node],
        pending: &[usize],
        instructions: &[Vec<Instruction>],
        layer: Layer,
    ) -> Vec<usize> {
        if pending.len() == 1 {
            return vec![0];
        }
        let timestamp_order = |&a: &usize| {
            let node = &nodes[pending[a]];
            (!node.cda(), node.timestamp, &node.key)
        };
        // A disjoint read/write pair cannot change the other's instructions.
        // Summaries are conservative; true dependencies still use CR 613.8's
        // before/after semantic comparison. Granted statics force rediscovery.
        let writes: Vec<_> = instructions
            .iter()
            .map(|is| is.iter().fold(0, |fields, i| fields | i.operation.writes()))
            .collect();
        let plausible = |a: usize, b: usize| {
            let node = &nodes[pending[a]];
            let reads = node.reads().fields | if node.is_bound() { 0 } else { ABILITIES };
            a != b && node.cda() == nodes[pending[b]].cda() && reads & writes[b] != 0
        };
        let independent = !(0..pending.len()).any(|a| (0..pending.len()).any(|b| plausible(a, b)))
            && !instructions
                .iter()
                .flatten()
                .any(|i| matches!(i.operation, Operation::Modify(Modification::GrantStatic(_))));
        let ordered = || {
            let mut order: Vec<_> = (0..pending.len()).collect();
            order.sort_by_key(timestamp_order);
            order
        };
        if independent && !cfg!(debug_assertions) {
            return ordered();
        }
        let mut dependencies = vec![vec![false; pending.len()]; pending.len()];
        #[cfg(feature = "benchmark")]
        let _timer = crate::benchmark_layers::Timer::new(|m, ns| {
            m.dependency_passes += 1;
            m.dependency_ns += ns;
        });
        for (b, &index) in pending.iter().enumerate() {
            if !cfg!(debug_assertions) && !(0..pending.len()).any(|a| plausible(a, b)) {
                continue;
            }
            let mut after = {
                #[cfg(feature = "benchmark")]
                let _timer = crate::benchmark_layers::Timer::new(|m, ns| {
                    m.contexts_cloned += 1;
                    m.clone_ns += ns;
                });
                self.clone()
            };
            after.apply(&nodes[index], &instructions[b]);
            for (a, &other) in pending.iter().enumerate() {
                if a != b && nodes[other].cda() == nodes[index].cda() {
                    if !cfg!(debug_assertions) && !plausible(a, b) {
                        continue;
                    }
                    dependencies[a][b] = self.exists(&nodes[other]) != after.exists(&nodes[other])
                        || instructions[a] != after.instructions(&nodes[other], layer);
                    debug_assert!(
                        plausible(a, b) || !dependencies[a][b],
                        "layer read/write summary missed a dependency"
                    );
                    #[cfg(feature = "benchmark")]
                    crate::benchmark_layers::record(|m| {
                        m.comparisons += 1;
                        m.edges += u64::from(dependencies[a][b]);
                    });
                }
            }
        }
        if independent {
            // Keep the complete pairwise oracle in debug, once per independent
            // batch rather than cloning the same board for every suffix.
            debug_assert!(dependencies.iter().flatten().all(|edge| !edge));
            return ordered();
        }
        // Transitive reachability identifies only edges within dependency
        // loops; external prerequisites still precede the loop (CR 613.8b).
        let mut reach = dependencies.clone();
        for k in 0..pending.len() {
            for a in 0..pending.len() {
                for b in 0..pending.len() {
                    reach[a][b] |= reach[a][k] && reach[k][b];
                }
            }
        }
        vec![(0..pending.len())
            .filter(|&a| !(0..pending.len()).any(|b| dependencies[a][b] && !reach[b][a]))
            .min_by_key(timestamp_order)
            .expect("dependency graph has a source component")]
    }

    fn exists(&self, node: &Node) -> bool {
        node.is_bound()
            || self
                .get(node.source)
                .is_some_and(|o| o.statics.iter().any(|s| s.key == node.key))
    }

    fn targets(&self, source: PermanentId, plan: &SourcePlan) -> Vec<PermanentId> {
        if plan
            .condition
            .as_ref()
            .is_some_and(|c| !self.condition(c, source))
        {
            return Vec::new();
        }
        match &plan.scope {
            StaticScope::This => self.get(source).map_or_else(Vec::new, |_| vec![source]),
            StaticScope::Attached => self
                .game
                .attached_permanent(source)
                .filter(|&id| self.get(id).is_some())
                .map_or_else(Vec::new, |id| vec![id]),
            StaticScope::SpellsYouControl(_) => Vec::new(),
            scope => {
                let source_controller = self.controller(source);
                self.live
                    .iter()
                    .copied()
                    .filter(|&id| {
                        matches_static_scope(
                            scope,
                            source,
                            source_controller,
                            None,
                            id,
                            self.controller(id),
                            |predicate| self.matches(id, predicate, Some(source)),
                        )
                    })
                    .collect()
            }
        }
    }

    fn instructions(&self, node: &Node, layer: Layer) -> Vec<Instruction> {
        match &node.plan {
            NodePlan::Resolved(part_layer, op) => {
                if *part_layer != layer {
                    return Vec::new();
                }
                vec![Instruction {
                    target: node.source,
                    operation: op.clone(),
                }]
            }
            NodePlan::Static { plan, targets } => {
                if !self.exists(node) {
                    return Vec::new();
                }
                let targets = match targets {
                    Some(targets) => targets.clone(),
                    None => self.targets(node.source, plan),
                };
                targets
                    .into_iter()
                    .flat_map(|id| {
                        plan.parts
                            .iter()
                            .filter(move |(part, _)| *part == layer)
                            .map(move |(_, op)| {
                                let operation = match op {
                                    Operation::Modify(m) => Operation::Modify(
                                        m.map_amount(|n| LayerValue::Fixed(self.value(n, id))),
                                    ),
                                    Operation::Cda {
                                        cda,
                                        sets_toughness,
                                    } => {
                                        let value = LayerValue::Fixed(
                                            self.cda(cda, self.controller(node.source)),
                                        );
                                        Operation::Modify(Modification::SetBasePt {
                                            power: Some(value.clone()),
                                            toughness: sets_toughness.then_some(value),
                                        })
                                    }
                                    Operation::Buff(buff) => self.buff(buff, node.source, id),
                                    op => op.clone(),
                                };
                                Instruction {
                                    target: id,
                                    operation,
                                }
                            })
                    })
                    .collect()
            }
        }
    }

    fn buff(&self, buff: &StaticPtBuff, source: PermanentId, id: PermanentId) -> Operation {
        let per_type = buff.per_creature_type_max.map_or(1, |max| {
            let view = &self.object(id).view;
            if view.has_all_creature_types() {
                max
            } else {
                (view.subtypes.iter().filter(|s| is_creature_type(s)).count() as i32).min(max)
            }
        });
        let controller = self.controller(source);
        let per_color = buff.per_color_among.as_ref().map_or(1, |p| {
            self.live
                .iter()
                .copied()
                .filter(|&id| self.controller(id) == controller && self.matches(id, p, None))
                .flat_map(|id| self.object(id).view.colors.iter().copied())
                .collect::<BTreeSet<_>>()
                .len() as i32
        });
        let per_card = buff
            .per_graveyard_matching
            .as_ref()
            .map_or(1, |p| self.graveyard_count(controller, p));
        let times = per_type * per_color * per_card;
        Operation::Modify(Modification::ModifyPt {
            power: LayerValue::Fixed(buff.power * times),
            toughness: LayerValue::Fixed(buff.toughness * times),
        })
    }

    fn controller(&self, id: PermanentId) -> PlayerId {
        self.game.live_permanent(id).controller
    }

    fn value(&self, value: &LayerValue, id: PermanentId) -> i32 {
        match value {
            LayerValue::Fixed(n) => *n,
            LayerValue::ManaValue => i32::from(
                self.game
                    .object_mana_value(self.game.live_permanent(id).object),
            ),
            LayerValue::GraveyardCardTypes { offset } => {
                [PlayerId(0), PlayerId(1)]
                    .into_iter()
                    .flat_map(|player| self.game.state.zone_cards(ZoneType::Graveyard, player))
                    .flat_map(|card| {
                        self.game.state[self.game.state.object_of(card)]
                            .types
                            .types
                            .iter()
                    })
                    .collect::<BTreeSet<_>>()
                    .len() as i32
                    + offset
            }
        }
    }

    fn creature_count(&self, controller: PlayerId) -> usize {
        self.live
            .iter()
            .copied()
            .filter(|&id| self.controller(id) == controller && self.object(id).view.is_creature())
            .count()
    }

    fn cda(&self, cda: &PowerCda, controller: PlayerId) -> i32 {
        match cda {
            PowerCda::CountersOnLandsYouControl => self
                .live
                .iter()
                .copied()
                .filter(|&id| self.controller(id) == controller && self.object(id).view.is_land())
                .map(|id| {
                    self.game
                        .live_permanent(id)
                        .counter_count(CounterKind::PlusOnePlusOne)
                })
                .sum(),
            PowerCda::CreaturesYouControl => self.creature_count(controller) as i32,
            PowerCda::GraveyardMatching(p) => self.graveyard_count(controller, p),
            PowerCda::GraveyardMatchingBy { who, predicate } => (0..self.game.state.players.len())
                .map(PlayerId)
                .filter(|player| who.includes(controller, *player))
                .map(|player| self.graveyard_count(player, predicate))
                .sum(),
        }
    }

    fn graveyard_count(&self, owner: PlayerId, predicate: &CardPredicate) -> i32 {
        self.game
            .state
            .zone_cards(ZoneType::Graveyard, owner)
            .filter(|card| {
                let object = self.game.state.object_of(*card);
                self.game.off_battlefield_matches(object, predicate, || {
                    let card = &self.game.state[object];
                    card.pt_cda().map_or_else(
                        || card.power.unwrap_or(0),
                        |(cda, _)| self.cda(cda, card.owner),
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

    fn select_permanents(
        &self,
        selector: &ObjectSelector,
        source: PermanentId,
    ) -> Vec<PermanentId> {
        let controller = self.controller(source);
        match selector {
            ObjectSelector::This => vec![source],
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
                self.live
                    .iter()
                    .copied()
                    .filter(|&id| {
                        !self.game.is_phased_out(id)
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
                    })
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    /// Static comparisons see the characteristics at this point in the layer system.
    fn comparison_value(&self, value: &EffectValue, source: PermanentId) -> i32 {
        let controller = self.controller(source);
        let controlled = || self.game.iter_battlefield_permanents(controller);
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
            } => aggregate.evaluate(
                self.select_permanents(of, source)
                    .into_iter()
                    .map(|id| self.property(id, *property)),
            ),
            EffectValue::Objects { of, predicate, .. } => self
                .select_permanents(of, source)
                .into_iter()
                .filter(|&id| {
                    predicate
                        .as_ref()
                        .is_none_or(|p| self.matches(id, p, Some(source)))
                })
                .count() as i32,
            EffectValue::DistinctManaValues { of } => self
                .select_permanents(of, source)
                .into_iter()
                .map(|id| self.game.permanent_mana_value(id))
                .collect::<BTreeSet<_>>()
                .len() as i32,
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
        )
    }

    fn apply(&mut self, node: &Node, instructions: &[Instruction]) {
        let clock = |ability: &Ability, index| {
            let Ability::Triggered { condition, .. } = ability;
            let static_slot = node.key.as_slice()[1];
            condition.once_each_turn().then(|| GrantClock {
                timestamp: node.timestamp,
                grantor: node.source,
                static_slot: (static_slot != usize::MAX).then_some(static_slot),
                ability_index: index,
            })
        };
        for (ordinal, instruction) in instructions.iter().enumerate() {
            let id = instruction.target;
            let origin = if id == node.source {
                node.origin
            } else {
                AbilityOrigin::Permanent(node.source)
            };
            let object = self.objects[id.0].as_mut().unwrap();
            match &instruction.operation {
                Operation::LandType(subtype) => {
                    if !object.view.is_land() {
                        continue;
                    }
                    {
                        let details = Arc::make_mut(&mut object.view).details_mut();
                        details
                            .subtypes
                            .retain(|s| !BASIC_LANDS.iter().any(|(name, _)| s == name));
                        details.subtypes.push(subtype.clone());
                        details.added_subtypes.push(subtype.clone());
                    }
                    lose_all_abilities(object);
                    Arc::make_mut(&mut object.view)
                        .details_mut()
                        .activated_abilities
                        .extend(intrinsic_mana_ability(|name| name == subtype));
                    object.dirty |= DIRTY_TYPES_SUBTYPES | DIRTY_ACTIVATIONS;
                }
                Operation::Inherit => {
                    if let Some(definition) = self.game.inherited_definition(id) {
                        let details = Arc::make_mut(&mut object.view).details_mut();
                        if !definition.activated_abilities.is_empty() {
                            details
                                .activated_abilities
                                .extend(definition.activated_abilities.clone());
                            object.dirty |= DIRTY_ACTIVATIONS;
                        }
                        details.triggered_abilities.extend(
                            definition.abilities.iter().cloned().enumerate().map(
                                |(index, ability)| EffectiveTrigger {
                                    origin,
                                    clock: clock(&ability, index),
                                    ability,
                                    program: false,
                                },
                            ),
                        );
                    }
                }
                Operation::CatalogTrigger(ability) => {
                    Arc::make_mut(&mut object.view)
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
                        let view = Arc::make_mut(&mut object.view);
                        view.name_set = true;
                        if view.name != *name {
                            view.details_mut().name = name.clone();
                        }
                    }
                    Modification::AllCreatureTypes => {
                        // A later grant of every type overrides an earlier set list.
                        let view = Arc::make_mut(&mut object.view);
                        view.creature_types_set = false;
                        if !view.creature_types.is_all() {
                            view.details_mut().creature_types = CreatureTypes::All;
                            object.dirty |= DIRTY_TYPES_SUBTYPES;
                        }
                    }
                    Modification::SetCreatureTypes(types) => {
                        let view = Arc::make_mut(&mut object.view);
                        view.creature_types_set = true;
                        let details = view.details_mut();
                        details.creature_types = CreatureTypes::default();
                        details.subtypes.retain(|s| !is_creature_type(s));
                        details.subtypes.extend(types.iter().cloned());
                        details.added_subtypes.retain(|s| !is_creature_type(s));
                        object.dirty |= DIRTY_TYPES_SUBTYPES;
                    }
                    Modification::AddSubtypes(types) => {
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
                            object.dirty |= DIRTY_TYPES_SUBTYPES;
                        }
                    }
                    Modification::AddTypes(types) => {
                        if types.iter().any(|t| !object.view.types.types.contains(t)) {
                            Arc::make_mut(&mut object.view)
                                .details_mut()
                                .types
                                .types
                                .extend(types.iter().copied());
                            object.dirty |= DIRTY_TYPES_SUBTYPES;
                        }
                    }
                    Modification::SetTypes { types, subtypes } => {
                        // CR 205.1a, 205.3d: a subtype survives while it
                        // corresponds to a remaining type and the effect sets
                        // no subtype of its kind (a basic land imprisoned as a
                        // land stays a Forest; a Citizen loses other creature
                        // types).
                        let view = Arc::make_mut(&mut object.view);
                        let keeps = |subtype: &String| {
                            types.iter().any(|t| subtype_corresponds_to(subtype, *t))
                                && !subtypes
                                    .iter()
                                    .any(|set| subtype_category(set) == subtype_category(subtype))
                        };
                        let sets_creature_types = subtypes.iter().any(|s| is_creature_type(s));
                        if !types.contains(&CardType::Creature) || sets_creature_types {
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
                        object.dirty |= DIRTY_TYPES_SUBTYPES;
                    }
                    Modification::BecomeCreature => {
                        if !object.view.types.types.contains(&CardType::Creature) {
                            Arc::make_mut(&mut object.view)
                                .details_mut()
                                .types
                                .types
                                .insert(CardType::Creature);
                            object.dirty |= DIRTY_TYPES_SUBTYPES;
                        }
                    }
                    Modification::Nonlegendary => {
                        if object.view.is_legendary {
                            Arc::make_mut(&mut object.view).is_legendary = false;
                        }
                    }
                    Modification::SetColors(colors) => {
                        if object.view.colors != *colors {
                            Arc::make_mut(&mut object.view).details_mut().colors = colors.clone();
                        }
                    }
                    Modification::AddColors(colors) => {
                        if colors.iter().any(|c| !object.view.colors.contains(c)) {
                            Arc::make_mut(&mut object.view)
                                .details_mut()
                                .colors
                                .extend(colors);
                        }
                    }
                    Modification::AddKeywords(keywords) => {
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
                    Modification::LoseNonManaAbilities => {
                        lose_nonmana_abilities(object);
                    }
                    Modification::LoseAllAbilities => {
                        lose_all_abilities(object);
                    }
                    Modification::Firebending(n) => {
                        let view = Arc::make_mut(&mut object.view);
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
                        Arc::make_mut(&mut object.view)
                            .details_mut()
                            .activated_abilities
                            .push((**ability).clone());
                        object.dirty |= DIRTY_ACTIVATIONS;
                    }
                    Modification::GrantTriggered(ability) => {
                        Arc::make_mut(&mut object.view)
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
                        if let StaticAbility::Ward(cost) = ability.as_ref() {
                            let details = Arc::make_mut(&mut object.view).details_mut();
                            details.ward = Some(cost.clone());
                            details.triggered_abilities.push(EffectiveTrigger {
                                origin,
                                clock: None,
                                ability: Ability::ward(cost.clone()),
                                program: true,
                            });
                        }
                        let key = node.key.extended(id.0, ordinal);
                        Arc::make_mut(&mut object.statics).push(StaticEntry {
                            origin,
                            key,
                            timestamp: node.timestamp.max(self.game.live_permanent(id).timestamp),
                            ability: (**ability).clone(),
                        });
                        object.dirty |= DIRTY_STATICS;
                    }
                    Modification::SetBasePt { power, toughness } => {
                        let new_power = match power {
                            Some(LayerValue::Fixed(p)) => *p,
                            _ => object.view.power,
                        };
                        let new_toughness = match toughness {
                            Some(LayerValue::Fixed(t)) => *t,
                            _ => object.view.toughness,
                        };
                        if new_power != object.view.power || new_toughness != object.view.toughness
                        {
                            let view = Arc::make_mut(&mut object.view);
                            view.power = new_power;
                            view.toughness = new_toughness;
                        }
                    }
                    Modification::ModifyPt {
                        power: LayerValue::Fixed(p),
                        toughness: LayerValue::Fixed(t),
                    } => {
                        if *p != 0 || *t != 0 {
                            let view = Arc::make_mut(&mut object.view);
                            view.power += p;
                            view.toughness += t;
                        }
                    }
                    Modification::SwitchPt => {
                        if object.view.power != object.view.toughness {
                            let view = Arc::make_mut(&mut object.view);
                            std::mem::swap(&mut view.power, &mut view.toughness);
                        }
                    }
                    _ => unreachable!("layer instruction is resolved"),
                },
                _ => unreachable!("layer instruction is resolved"),
            }
        }
    }
}

fn build_node(game: &Game, source: PermanentId, entry: &StaticEntry) -> Option<Node> {
    let mut plan = SourcePlan::on_source(Vec::new());
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
        StaticAbility::PtBuff(b) => {
            plan.scope = b.scope.clone();
            plan.condition = b.condition.clone();
            plan.parts
                .push((Layer::ModifyPower, Operation::Buff(Box::new(b.clone()))));
        }
        StaticAbility::CharacteristicPower(cda) | StaticAbility::CharacteristicPt(cda) => {
            plan.cda = true;
            plan.parts.push((
                Layer::CharacteristicPower,
                Operation::Cda {
                    cda: Box::new(cda.clone()),
                    sets_toughness: matches!(entry.ability, StaticAbility::CharacteristicPt(_)),
                },
            ));
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
                    Operation::Modify(Modification::SetCreatureTypes(
                        set.creature_types.clone(),
                    )),
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
    Some(Node {
        origin: entry.origin,
        source,
        key: entry.key.clone(),
        timestamp: entry.timestamp,
        plan: NodePlan::Static {
            plan: Arc::new(plan.summarize()),
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

fn lose_nonmana_abilities(object: &mut LayerObject) {
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
    if !object.statics.is_empty() {
        object.statics = empty_statics();
        object.dirty |= DIRTY_STATICS;
    }
}

fn lose_all_abilities(object: &mut LayerObject) {
    if !object.view.mana_abilities.is_empty()
        || !object.view.triggered_mana_abilities.is_empty()
        || !object.view.activated_abilities.is_empty()
    {
        let details = Arc::make_mut(&mut object.view).details_mut();
        details.mana_abilities.clear();
        details.triggered_mana_abilities.clear();
        details.activated_abilities.clear();
    }
    lose_nonmana_abilities(object);
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
    .matches(p, source_name)
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
    .matches(p, None)
}

impl CharacteristicView<'_> {
    fn has_subtype(&self, subtype: &str) -> bool {
        self.subtypes.iter().any(|s| s == subtype)
            || (self.all_creature_types && is_creature_type(subtype))
    }

    fn matches(&self, p: &CardPredicate, source_name: Option<&str>) -> bool {
        // A permanent is on the battlefield, never a graveyard card.
        !p.from_battlefield_this_turn
            && p.accepts_name(self.name)
            && (!p.namesake || source_name == Some(self.name))
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
                    .any(|alternative| self.matches(alternative, source_name)))
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
